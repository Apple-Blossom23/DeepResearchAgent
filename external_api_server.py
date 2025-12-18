"""
外部API服务
提供SSE流式接口供外部调用
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import os

from external_config import external_config
from task_manager import task_manager, TaskStatus
from sse_manager import sse_manager
from ReAct_Workflow import ReActAgent
from custom_dashscope_llm import customDashscopeLLM
from config import config
from workflow_strategy import StreamingWorkflowStrategy
from log_config import get_external_api_logger
from results_repository import init_results_table, insert_answers, query_answers


# 配置日志
logger = get_external_api_logger()

# 创建FastAPI应用
app = FastAPI(
    title="DeepResearchAgent External API",
    description="提供SSE流式接口的深度研究代理服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=external_config.ALLOWED_ORIGINS,
    allow_credentials=external_config.ALLOW_CREDENTIALS,
    allow_methods=external_config.ALLOW_METHODS,
    allow_headers=external_config.ALLOW_HEADERS,
)

# 挂载静态文件
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

# 请求模型
class AnalyzeRequest(BaseModel):
    input: str = Field(..., description="问题/需求描述", min_length=1, max_length=10000)
    options: Optional[Dict[str, Any]] = Field(default=None, description="分析选项")

# UserResponseRequest 已删除，简化架构不需要用户交互

# 认证中间件
async def authenticate_api_key(request: Request):
    """验证API密钥"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少有效的Authorization头")
    
    api_key = auth_header.split(" ")[1]
    if api_key not in external_config.API_KEYS:
        raise HTTPException(status_code=401, detail="无效的API密钥")
    
    return api_key

# 速率限制中间件
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_rate_limited(self, client_ip: str) -> bool:
        now = datetime.now().timestamp()
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # 清理超过1分钟的请求记录
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if now - req_time < 60
        ]
        
        # 检查是否超过限制
        if len(self.requests[client_ip]) >= external_config.RATE_LIMIT_PER_MINUTE:
            return True
        
        # 记录当前请求
        self.requests[client_ip].append(now)
        return False

rate_limiter = RateLimiter()

async def run_react_with_streaming(input_text: str, task_id: str, options: Optional[Dict[str, Any]] = None):
    """直接运行ReAct工作流并支持流式输出"""
    try:
        # 创建LLM实例
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
        
        # 创建流式策略
        streaming_strategy = StreamingWorkflowStrategy(task_id, sse_manager, task_manager)
        
        # 创建更多LLM实例
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
        
        # 创建ReAct代理
        from ReAct_Tools import tools
        react_agent = ReActAgent(
            llm=llm,
            planning_llm=planning_llm,
            conclusion_llm=conclusion_llm,
            filter_llm=filter_llm,
            plan_update_llm=plan_update_llm,
            tools=tools,
            workflow_strategy=streaming_strategy,
            timeout=config.WORKFLOW_TIMEOUT,
            verbose=config.WORKFLOW_VERBOSE
        )
        
        # 更新任务状态
        await task_manager.update_task_status(
            task_id, TaskStatus.RUNNING, 
            progress=10.0, current_step="开始分析"
        )
        
        # 发送任务开始事件
        # 为外部API调用设置默认分类
        event_metadata = {
            "input": input_text,
            "timestamp": datetime.now().isoformat(),
            "category": "research-general"  # 外部API默认分类
        }
        await streaming_strategy.on_workflow_event("task_start", "开始分析任务", event_metadata)
        
        # 直接运行ReAct工作流
        result = await react_agent.run(input=input_text)
        normalized_result: Dict[str, Any] = {}
        try:
            answers = []
            if isinstance(result, dict):
                pr = result.get("parallel_results")
                if isinstance(pr, dict):
                    normalized_result["parallel_results"] = pr
                    for cat, r in pr.items():
                        try:
                            content = getattr(r, 'response', None)
                            if not content and isinstance(r, dict):
                                content = r.get('response')
                            if content:
                                answers.append({"workflow_category": cat, "answer_type": "final_response", "content": str(content)})
                        except Exception:
                            pass
                else:
                    wc = result.get("workflow_category", "")
                    normalized_result["parallel_results"] = {wc or "综合": result}
                    r = result
                    content = getattr(r, 'response', None)
                    if not content and isinstance(r, dict):
                        content = r.get('response')
                    if content:
                        answers.append({"workflow_category": wc or "综合", "answer_type": "final_response", "content": str(content)})
            else:
                normalized_result["parallel_results"] = {"综合": {"response": str(result), "status": "completed"}}
                answers.append({"workflow_category": "综合", "answer_type": "final_response", "content": str(result)})
            if answers:
                insert_answers(task_id, answers)
        except Exception as e:
            logger.error(f"结果入库失败: {e}", exc_info=True)
        
        # 更新任务状态为完成
        await task_manager.update_task_status(
            task_id, TaskStatus.COMPLETED,
            progress=100.0, current_step="完成",
            result=normalized_result
        )
        
        # 发送任务完成事件
        event_metadata = {
            "result": normalized_result,
            "timestamp": datetime.now().isoformat(),
            "category": "research-general"  # 外部API默认分类
        }
        await streaming_strategy.on_workflow_event("task_complete", "任务分析完成", event_metadata)
        
    except Exception as e:
        logger.error(f"ReAct工作流执行失败: {e}", exc_info=True)
        
        # 更新任务状态为失败
        await task_manager.update_task_status(
            task_id, TaskStatus.FAILED,
            error=str(e)
        )
        
        # 发送错误事件
        await sse_manager.send_message(task_id, "task_error", {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        raise

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """速率限制中间件"""
    client_ip = request.client.host
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="请求频率过高，请稍后再试")
    return await call_next(request)

@app.middleware("http")
async def request_size_middleware(request: Request, call_next):
    """请求大小限制中间件"""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > external_config.MAX_REQUEST_SIZE:
        raise HTTPException(status_code=413, detail="请求体过大")
    return await call_next(request)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """日志中间件"""
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"外部调用: {request.method} {request.url.path} - {response.status_code} - {duration:.2f}s")
    return response

# API端点
@app.post("/api/v1/external/analyze/stream")
async def external_analyze_stream(
    request: AnalyzeRequest,
    api_key: str = Depends(authenticate_api_key)
):
    """直接启动分析任务并返回SSE流"""
    try:
        # 创建任务
        task_id = await task_manager.create_task(
            input_text=request.input,
            created_by=api_key,
            options=request.options
        )
        
        # 创建SSE连接
        await sse_manager.create_connection(task_id)
        
        # 异步启动分析任务
        asyncio.create_task(
            run_react_with_streaming(
                input_text=request.input,
                task_id=task_id,
                options=request.options
            )
        )
        
        # 直接返回SSE流
        return StreamingResponse(
            sse_manager.generate_sse_stream(task_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "X-Task-ID": task_id,
            }
        )
        
    except Exception as e:
        logger.error(f"创建分析任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

@app.get("/api/v1/external/status/{task_id}")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(authenticate_api_key)
):
    """获取任务状态"""
    try:
        task = await task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.created_by != api_key:
            raise HTTPException(status_code=403, detail="无权访问此任务")
        
        return task.to_dict()
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@app.delete("/api/v1/external/cancel/{task_id}")
async def cancel_task(
    task_id: str,
    api_key: str = Depends(authenticate_api_key)
):
    """取消任务"""
    try:
        task = await task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.created_by != api_key:
            raise HTTPException(status_code=403, detail="无权访问此任务")
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="任务已完成，无法取消")
        
        success = await task_manager.cancel_task(task_id)
        if success:
            await sse_manager.send_message(task_id, "task_cancelled", {
                "message": "任务已取消",
                "timestamp": datetime.now().isoformat()
            })
        
        return {"message": "任务已取消", "task_id": task_id}
        
    except Exception as e:
        logger.error(f"取消任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")

@app.get("/api/v1/external/tasks")
async def list_tasks(
    api_key: str = Depends(authenticate_api_key),
    limit: int = 10
):
    """列出用户的任务"""
    try:
        tasks = await task_manager.list_tasks(created_by=api_key)
        return {
            "tasks": [task.to_dict() for task in tasks[:limit]],
            "total": len(tasks)
        }
        
    except Exception as e:
        logger.error(f"列出任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"列出任务失败: {str(e)}")

# 用户交互功能已移除，简化架构不需要用户确认
# @app.post("/api/v1/external/user/response")
# async def submit_user_response(...): 
#     功能已移除

@app.get("/")
async def root():
    """重定向到静态文件"""
    return RedirectResponse(url="/static/index.html")

@app.get("/api/v1/external/health")
async def health_check():
    """健康检查"""
    task_stats = task_manager.get_task_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": sse_manager.get_active_connections_count(),
        "task_stats": task_stats
    }

@app.get("/api/v1/results")
async def list_results(
    task_id: Optional[str] = None,
    category: Optional[str] = None,
    answer_type: Optional[str] = None,
    limit: int = 50,
    api_key: str = Depends(authenticate_api_key)
):
    try:
        items = query_answers(task_id=task_id, category=category, answer_type=answer_type, limit=limit)
        return {"results": items}
    except Exception as e:
        logger.error(f"查询结果失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="查询结果失败")

# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("外部API服务启动")
    
    await sse_manager.start_heartbeat()
    await task_manager.start_cleanup_scheduler()
    await task_manager.start_timeout_checker()
    try:
        from db_pool_manager import initialize_db_pool
        initialize_db_pool()
        logger.info("数据库连接池初始化完成")
        init_results_table()
        logger.info("结果表初始化完成")
    except Exception as e:
        logger.error(f"结果表初始化失败: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("外部API服务关闭")
    
    # 停止SSE心跳
    await sse_manager.stop_heartbeat()
    
    # 停止任务清理调度器
    await task_manager.stop_cleanup_scheduler()

if __name__ == "__main__":
    uvicorn.run(
        "external_api_server:app",
        host=external_config.HOST,
        port=external_config.PORT,
        reload=True,
        log_level=external_config.LOG_LEVEL.lower()
    )
