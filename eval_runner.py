import asyncio
import csv
import argparse
import time
import json
import os
import logging
from typing import Dict, Any, List, Union
from datetime import datetime

from ReAct_Workflow import ReActAgent
from custom_dashscope_llm import customDashscopeLLM
from workflow_strategy import DefaultWorkflowStrategy
from config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eval_runner.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EvalRunner")

def _build_agent() -> ReActAgent:
    """构建ReAct智能体实例"""
    llm = customDashscopeLLM(
        model_code=config.DEFAULT_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        context_window=config.DEFAULT_CONTEXT_WINDOW,
        max_tokens=config.DEFAULT_NUM_OUTPUT
    )
    planning_llm = customDashscopeLLM(
        model_code=config.PLANNING_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DEFAULT_TEMPERATURE,
        top_p=config.DEFAULT_TOP_P,
        context_window=config.DEFAULT_CONTEXT_WINDOW,
        max_tokens=config.DEFAULT_NUM_OUTPUT
    )
    conclusion_llm = customDashscopeLLM(
        model_code=config.CONCLUSION_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P
    )
    filter_llm = customDashscopeLLM(
        model_code=config.FILTER_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P
    )
    plan_update_llm = customDashscopeLLM(
        model_code=config.PLANNING_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P
    )
    workflow_strategy = DefaultWorkflowStrategy()
    agent = ReActAgent(
        llm=llm,
        planning_llm=planning_llm,
        conclusion_llm=conclusion_llm,
        filter_llm=filter_llm,
        plan_update_llm=plan_update_llm,
        tools=[],
        workflow_strategy=workflow_strategy,
        timeout=config.WORKFLOW_TIMEOUT,
        verbose=config.WORKFLOW_VERBOSE
    )
    return agent

def _normalize_eval_item(item: dict) -> dict:
    """解析单个输入项，提取输入文本"""
    if isinstance(item, str):
        return {"input": item.strip(), "raw": item}
    if isinstance(item, dict):
        # 通用格式: {"input": "...", "metadata": {...}, "attachments": [...]}}
        if isinstance(item, dict) and ("input" in item or "metadata" in item or "attachments" in item):
            try:
                payload = {
                    "input": item.get("input", ""),
                    "metadata": item.get("metadata", {}),
                    "attachments": item.get("attachments", []),
                }
                return {
                    "input": json.dumps(payload, ensure_ascii=False),
                    "raw": item,
                }
            except Exception:
                return {
                    "input": str(item.get("input", "")) or "",
                    "raw": item,
                }
        return {
            "input": str(item.get("input") or item.get("query") or item.get("text") or ""),
            "raw": item,
        }
    return {"input": str(item), "raw": item}

def _read_inputs(path: str) -> List[Dict[str, Any]]:
    """
    读取输入文件，支持自动格式检测
    支持格式：
    1. JSON Array: [{"input": "..."}, ...] 或 ["...", ...]
    2. JSON Lines: 每行一个JSON对象
    3. Plain Text: 每行一个文本
    4. CSV: 尝试解析第一列或特定列头
    """
    rows: List[Dict[str, Any]] = []
    
    if not os.path.exists(path):
        logger.error(f"输入文件不存在: {path}")
        return rows

    ext = os.path.splitext(path)[1].lower()
    
    try:
        # 尝试作为整个JSON读取
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    logger.info(f"检测到JSON数组格式，包含 {len(data)} 条记录")
                    for i, item in enumerate(data):
                        text = _normalize_eval_item(item)
                        if text:
                            rows.append({"id": str(i + 1), "input": text["input"], "raw": text["raw"]})
                    return rows
            except json.JSONDecodeError:
                pass # 不是标准JSON数组，继续尝试其他格式

        # 按行处理
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 检测是否为CSV (简单的启发式：检查首行是否有逗号且扩展名为csv)
        if ext == '.csv':
            import io
            f_csv = io.StringIO("".join(lines))
            reader = csv.DictReader(f_csv)
            if reader.fieldnames:
                logger.info("检测到CSV格式")
                for i, row in enumerate(reader):
                    # 优先查找特定列
                    text = ""
                    for key in ["input", "query", "content"]:
                        if key in row and row[key]:
                            text = row[key]
                            break
                    if not text and reader.fieldnames:
                        # 默认取第一列
                        text = row[reader.fieldnames[0]]
                    
                    rid = row.get("id") or row.get("case_id") or str(i + 1)
                    if text:
                        rows.append({"id": rid, "input": text, "raw": row})
                return rows

        # 处理 JSON Lines 或 纯文本
        logger.info("尝试按行解析 (JSON Lines 或 纯文本)")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 尝试解析为JSON对象
                item = json.loads(line)
                text = _parse_input_item(item)
                rows.append({"id": str(i + 1), "input": text, "raw": item})
            except json.JSONDecodeError:
                # 解析失败，视为纯文本
                rows.append({"id": str(i + 1), "input": line, "raw": line})
                
    except Exception as e:
        logger.error(f"读取输入文件失败: {e}")
        
    logger.info(f"共读取 {len(rows)} 条有效输入")
    return rows

def _write_outputs(path: str, results: List[Dict[str, Any]]):
    """将结果写入CSV文件"""
    headers = [
        "id",
        "input",
        "status",
        "elapsed_ms",
        "categories",
        "responses",
        "error",
        "timestamp"
    ]
    
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in results:
                cats = []
                pr = r.get("parallel_results", {})
                if isinstance(pr, dict):
                    cats = list(pr.keys())
                
                resp_texts = []
                for a in r.get("answers", []):
                    cat = a.get('workflow_category', 'Unknown')
                    content = a.get('content', '')
                    resp_texts.append(f"[{cat}]\n{content}")
                
                writer.writerow([
                    r.get("id"),
                    r.get("input"),
                    r.get("status"),
                    int((r.get("elapsed") or 0) * 1000),
                    ",".join(cats),
                    "\n\n".join(resp_texts),
                    r.get("error", ""),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
        logger.info(f"结果已保存至: {path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

async def _run_single_safe(agent: ReActAgent, input_text: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    """带并发控制的单条执行"""
    async with sem:
        start = time.time()
        logger.info(f"开始执行: {input_text[:20]}...")
        try:
            # 重新构建agent以确保状态隔离（如果agent是有状态的）
            # 注意：ReActAgent目前的设计是单个实例可以多次运行，但为了避免上下文污染，最好每次重建
            # 或者调用 agent.reset() 如果有的话。这里我们复用agent但ReActAgent.run通常会初始化新上下文
            # 为了最大安全性，这里我们还是为每个请求构建新agent，或者确保agent是无状态的
            # 鉴于 _build_agent 开销不大，我们在外部构建，但在并发场景下，
            # 如果 agent 内部有共享状态可能导致问题。
            # 检查 ReActAgent 实现，run 方法会创建新的 Context，所以应该是安全的。
            # 但为了保险起见，建议每个任务独立的 agent 实例
            local_agent = _build_agent()
            
            result = await local_agent.run(input=input_text)
            
            normalized: Dict[str, Any] = {}
            answers: List[Dict[str, Any]] = []
            
            if isinstance(result, dict):
                pr = result.get("parallel_results")
                if isinstance(pr, dict):
                    normalized["parallel_results"] = pr
                    for cat, r in pr.items():
                        try:
                            content = getattr(r, "response", None)
                            if not content and isinstance(r, dict):
                                content = r.get("response")
                            if content:
                                answers.append({"workflow_category": cat, "content": str(content)})
                        except Exception:
                            pass
                else:
                    wc = result.get("workflow_category", "")
                    normalized["parallel_results"] = {wc or "综合": result}
                    r = result
                    content = getattr(r, "response", None)
                    if not content and isinstance(r, dict):
                        content = r.get("response")
                    if content:
                        answers.append({"workflow_category": wc or "综合", "content": str(content)})
            else:
                normalized["parallel_results"] = {"综合": {"response": str(result), "status": "completed"}}
                answers.append({"workflow_category": "综合", "content": str(result)})
            
            elapsed = (time.time() - start)
            logger.info(f"执行完成: {input_text[:20]}... 耗时: {elapsed:.2f}s")
            
            return {
                "status": "completed",
                "elapsed": elapsed,
                "answers": answers,
                "parallel_results": normalized.get("parallel_results", {})
            }
        except Exception as e:
            elapsed = (time.time() - start)
            logger.error(f"执行失败: {input_text[:20]}... 错误: {str(e)}")
            return {
                "status": "failed", 
                "error": str(e), 
                "elapsed": elapsed, 
                "answers": [], 
                "parallel_results": {}
            }

def _append_output(path: str, result: Dict[str, Any]):
    """将单条结果追加写入CSV文件"""
    headers = [
        "id",
        "input",
        "status",
        "elapsed_ms",
        "categories",
        "responses",
        "error",
        "timestamp"
    ]
    
    file_exists = os.path.exists(path)
    
    try:
        with open(path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(headers)
                
            cats = []
            pr = result.get("parallel_results", {})
            if isinstance(pr, dict):
                cats = list(pr.keys())
            
            resp_texts = []
            for a in result.get("answers", []):
                cat = a.get('workflow_category', 'Unknown')
                content = a.get('content', '')
                resp_texts.append(f"[{cat}]\n{content}")
            
            writer.writerow([
                result.get("id"),
                result.get("input"),
                result.get("status"),
                int((result.get("elapsed") or 0) * 1000),
                ",".join(cats),
                "\n\n".join(resp_texts),
                result.get("error", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
        logger.info(f"结果已追加至: {path}")
    except Exception as e:
        logger.error(f"追加结果失败: {e}")

async def _run_all(input_path: str, output_path: str, limit: int | None = None, concurrency: int = 1, wait_interval: int = 300):
    """批量运行所有用例"""
    logger.info(f"开始批量评测任务，输入: {input_path}, 并发数: {concurrency}, 间隔等待: {wait_interval}秒")
    
    cases = _read_inputs(input_path)
    if not cases:
        logger.warning("没有找到有效的输入用例")
        return

    if limit:
        logger.info(f"限制运行前 {limit} 条用例")
        cases = cases[:limit]

    # 并发控制信号量
    sem = asyncio.Semaphore(concurrency)
    
    # 串行或并行执行
    if concurrency <= 1:
        logger.info("使用串行模式执行...")
        for i, c in enumerate(cases):
            logger.info(f"正在处理第 {i+1}/{len(cases)} 条用例...")
            result = await _run_single_safe(None, c["input"], sem)
            
            # 补充ID和Input信息以便落表
            result["id"] = c["id"]
            result["input"] = c["input"]
            
            # 立即落表
            _append_output(output_path, result)
            
            # 如果不是最后一条，等待指定间隔
            if i < len(cases) - 1:
                logger.info(f"等待 {wait_interval} 秒以减轻模型压力...")
                await asyncio.sleep(wait_interval)
                
    else:
        logger.info(f"使用并行模式执行，并发数: {concurrency}...")
        # 并行模式暂不支持单条即时落表和间隔等待的完美结合
        # 但可以在每个任务完成后落表
        tasks = []
        
        async def run_and_save(case_item):
            res = await _run_single_safe(None, case_item["input"], sem)
            res["id"] = case_item["id"]
            res["input"] = case_item["input"]
            _append_output(output_path, res)
            return res

        for c in cases:
            task = asyncio.create_task(run_and_save(c))
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
    
    logger.info("批量评测任务完成")

def main():
    parser = argparse.ArgumentParser(description="自动化评测脚本")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径 (支持 .json, .txt, .csv)")
    parser.add_argument("--output", type=str, default="eval_output.csv", help="输出CSV文件路径")
    parser.add_argument("--limit", type=int, default=None, help="限制运行的用例数量")
    parser.add_argument("--concurrency", type=int, default=1, help="并发执行数量 (默认: 1，即串行执行)")
    parser.add_argument("--wait", type=int, default=300, help="单条用例执行后的等待间隔(秒)，默认300秒")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        return

    try:
        asyncio.run(_run_all(args.input, args.output, args.limit, args.concurrency, args.wait))
    except KeyboardInterrupt:
        print("\n任务被用户中断")
    except Exception as e:
        logger.exception("程序运行发生未捕获异常")
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
