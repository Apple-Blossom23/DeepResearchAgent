import httpx
from fastmcp import FastMCP

from typing import Dict, Any, List, Optional, Callable
import psycopg2
import json
import re

# 导入ReAct_Tools中的依赖
from custom_dashscope_llm import customDashscopeLLM
from config import config, VECTOR_STORE_CONFIG, EMBEDDING_CONFIG
from custom_react_system_prompt import CONCLUSION_PROMPT_TEMPLATE
from log_config import get_mcp_logger

# 初始化 FastMCP 服务器
mcp = FastMCP('tools')
logger = get_mcp_logger()

# 工具超时时间配置 - 通过装饰器自动填充
_TOOL_TIMEOUT_CONFIG = {}

def tool_config(timeout: Optional[float] = None):
    """
    工具配置装饰器

    用于为工具函数添加超时时间配置。会自动将超时时间注册到配置字典中。

    Args:
        timeout: 工具执行超时时间（秒），不提供则使用默认300秒

    Usage:
        @mcp.tool()
        @tool_config(timeout=300.0)
        async def my_tool():
            pass
    """
    def decorator(func: Callable) -> Callable:
        # 设置超时时间属性
        timeout_value = timeout if timeout is not None else 60.0
        func._tool_timeout = timeout_value

        # 自动注册到配置字典
        if func.__name__ not in _TOOL_TIMEOUT_CONFIG:
            _TOOL_TIMEOUT_CONFIG[func.__name__] = timeout_value

        return func
    return decorator

def get_tool_timeout(func_name: str) -> float:
    """
    获取工具的超时时间

    Args:
        func_name: 工具函数名

    Returns:
        超时时间（秒）
    """
    return _TOOL_TIMEOUT_CONFIG.get(func_name, 60.0)

def _get_dynamic_es_index(category: str = None) -> str:
    """
    根据category动态获取ES索引名称
    
    Args:
        category: 工作流程分类
        
    Returns:
        对应的ES索引名称
    """
    if category and hasattr(config, 'ES_INDEX_MAPPING'):
        return config.ES_INDEX_MAPPING.get(category, config.ES_INDEX)
    return config.ES_INDEX

def _extract_final_answer(content: str) -> str:
    """
    从包含<think>标签的内容中提取最终答案
    
    Args:
        content: 包含思考过程和最终答案的完整内容
        
    Returns:
        最终答案内容
    """
    if not content:
        return ""
    
    # 检查是否包含<think>标签
    if "<think>" in content and "</think>" in content:
        # 找到</think>标签的位置
        think_end = content.find("</think>")
        if think_end != -1:
            # 提取</think>之后的内容
            final_answer = content[think_end + 8:].strip()
            return final_answer
    
    # 如果没有<think>标签，直接返回原内容
    return content.strip()

@mcp.tool(exclude_args=['category'])
async def search_documents(query: str = "", category: str = "") -> List[Dict[str, Any]]:
    """
    文档信息检索工具。

    功能：基于查询语句进行检索，返回与问题相关的文档片段。

    Args:
        query: 检索关键词
        category: 可选分类，用于选择检索数据源

    Returns:
        返回包含 doc_name、chunk、score 等字段的字典列表
    """
    if not query.strip():
        return []
    return await _es_full_text_search(query, category)

def _manual_rrf_fusion(
    vector_results: List[Dict[str, Any]], 
    text_results: List[Dict[str, Any]], 
    k: Optional[int] = None, 
    top_n: Optional[int] = None,
    only_text: bool = False
) -> List[Dict[str, Any]]:
    """
    手动实现RRF融合算法
    
    RRF公式: score(d) = Σ [1 / (k + rank_i(d))]
    
    Args:
        vector_results: 向量检索结果
        text_results: 全文检索结果
        k: RRF平滑常数 (默认使用config.ES_RRF_K)
        top_n: 返回Top-N结果 (默认使用config.ES_SEARCH_SIZE)
    
    Returns:
        融合后的结果列表
    """
    # 使用配置文件中的默认值
    if k is None:
        k = config.ES_RRF_K
    if top_n is None:
        top_n = config.ES_SEARCH_SIZE
    
    # 构建文档字典
    doc_dict = {}
    
    if not only_text:
        for item in vector_results:
            doc_id = item['id']
            if doc_id not in doc_dict:
                doc_dict[doc_id] = {
                    'doc_name': item['doc_name'],
                    'chunk': item['chunk'],
                    'vector_rank': item['vector_rank'],
                    'text_rank': None,
                    'rrf_score': 0.0
                }
            doc_dict[doc_id]['rrf_score'] += 1.0 / (k + item['vector_rank'])
    
    # 处理全文检索结果
    for item in text_results:
        doc_id = item['id']
        if doc_id not in doc_dict:
            doc_dict[doc_id] = {
                'doc_name': item['doc_name'],
                'chunk': item['chunk'],
                'vector_rank': None,
                'text_rank': item['text_rank'],
                'rrf_score': 0.0
            }
        else:
            doc_dict[doc_id]['text_rank'] = item['text_rank']
        # RRF贡献: 1 / (k + rank)
        doc_dict[doc_id]['rrf_score'] += 1.0 / (k + item['text_rank'])
    
    # 按RRF分数排序
    sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
    
    # 返回Top-N
    final_results = []
    for rank, (doc_id, doc_info) in enumerate(sorted_docs[:top_n], 1):
        final_results.append({
            'doc_name': doc_info['doc_name'],
            'chunk': doc_info['chunk'],
            'score': doc_info['rrf_score'],  # 使用rrf_score作为score
            'rrf_score': doc_info['rrf_score'],
            'rank': rank,  # RRF融合后的排名
            'vector_rank': doc_info['vector_rank'],
            'text_rank': doc_info['text_rank']
        })
    
    return final_results

async def _es_full_text_search(query: str, category: str = None) -> List[Dict[str, Any]]:
    """
    ES混合检索实现（向量+全文，手动RRF融合）

    Args:
        query: 用户查询语句
        
    Returns:
        包含doc_name、chunk、score、rank的字典列表
    """
    try:
        if not config.ES_HOST or not config.ES_PORT or not config.ES_INDEX:
            logger.warning("ES未配置，search_documents将返回空结果")
            return []
        if not EMBEDDING_CONFIG.get("online_url"):
            logger.warning("Embedding服务未配置，search_documents将返回空结果")
            return []
        # 1. 使用 embedding 模型将查询转换为向量
        from custom_embedding import OnlineQwen3Embedding
        embedder = OnlineQwen3Embedding(online_url=EMBEDDING_CONFIG["online_url"])
        query_vector = embedder._get_query_embedding(query)
        logger.info(f"查询向量维度: {len(query_vector)}")
        
        # ES配置 - 动态获取ES索引
        es_index = _get_dynamic_es_index(category)
        logger.info(f"查询索引库: {es_index}")
        es_url = f"http://{config.ES_HOST}:{config.ES_PORT}/{es_index}/_search"
        headers = {
            'Content-Type': 'application/json',
            'authorization': config.ES_AUTH
        }
        
        # 2. 执行向量检索（召回更多候选，用于RRF融合）
        vector_candidates = config.ES_VECTOR_CANDIDATES  # 从配置文件读取向量检索候选数
        vector_search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": vector_candidates,
                "num_candidates": vector_candidates * 3
            },
            "_source": ["doc_name", "chunk"],
            "size": vector_candidates
        }
        
        logger.info(f"执行向量检索 (召回Top-{vector_candidates})...")
        async with httpx.AsyncClient() as client:
            vector_response = await client.post(
                es_url,
                headers=headers,
                json=vector_search_body,
                timeout=config.ES_TIMEOUT
            )
            
            if vector_response.status_code != 200:
                logger.error(f"向量检索失败: {vector_response.status_code}")
                logger.error(f"错误响应: {vector_response.text}")
                vector_results = []
            else:
                vector_data = vector_response.json()
                vector_hits = vector_data.get('hits', {}).get('hits', [])
                vector_results = [
                    {
                        'id': hit.get('_id'),
                        'doc_name': hit['_source'].get('doc_name', ''),
                        'chunk': hit['_source'].get('chunk', ''),
                        'vector_score': hit.get('_score', 0.0),
                        'vector_rank': rank
                    }
                    for rank, hit in enumerate(vector_hits, 1)
                ]
                logger.info(f"✅ 向量检索完成，召回 {len(vector_results)} 条")
        
        # 3. 执行全文检索（召回更多候选，用于RRF融合）
        text_candidates = config.ES_TEXT_CANDIDATES  # 从配置文件读取全文检索候选数
        text_search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "chunk": query
                            }
                        }
                    ]
                }
            },
            "_source": ["doc_name", "chunk"],
            "size": text_candidates
        }
        
        logger.info(f"执行全文检索 (召回Top-{text_candidates})...")
        async with httpx.AsyncClient() as client:
            text_response = await client.post(
                es_url,
                headers=headers,
                json=text_search_body,
                timeout=config.ES_TIMEOUT
            )
            
            if text_response.status_code != 200:
                logger.error(f"全文检索失败: {text_response.status_code}")
                text_results = []
            else:
                text_data = text_response.json()
                text_hits = text_data.get('hits', {}).get('hits', [])
                text_results = [
                    {
                        'id': hit.get('_id'),
                        'doc_name': hit['_source'].get('doc_name', ''),
                        'chunk': hit['_source'].get('chunk', ''),
                        'text_score': hit.get('_score', 0.0),
                        'text_rank': rank
                    }
                    for rank, hit in enumerate(text_hits, 1)
                ]
                logger.info(f"✅ 全文检索完成，召回 {len(text_results)} 条")
        
        # 4. 手动RRF融合
        logger.info(f"执行手动RRF融合...")
        final_results = _manual_rrf_fusion(
            vector_results, 
            text_results,
            # k和top_n使用配置文件中的默认值
            only_text=False
        )
        
        logger.info(f"✅ ES混合检索成功! 使用向量+全文+手动RRF融合，返回 {len(final_results)} 条结果")
        if final_results:
            logger.info(f"📊 Top 3 结果: {[(r['doc_name'], round(r['rrf_score'], 4)) for r in final_results[:3]]}")
        
        return final_results
            
    except Exception as e:
        logger.error(f"❌ ES混合检索错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return []

async def retrieve_plans_from_db(query: str = "", category: Optional[str] = None) -> List[str]:
    """
    从数据库精确检索计划的工具（完全匹配，不使用嵌入/相似度）。
    支持分类匹配：如果query格式为"[分类]原始查询"，则只返回匹配该分类的计划。
 
    Args:
        query: 查询语句。支持格式：
               - 普通查询："如何设计可插拔工具系统"
               - 分类查询："[research-planning]如何设计可插拔工具系统"
 
    Returns:
        检索到的计划列表（按 query 字段完全匹配，支持分类过滤）
    """
    if not query.strip():
        return []
    if not category or not str(category).strip():
        return []

    try:
        # 使用连接池获取数据库连接
        from db_pool_manager import db_pool_manager

        with db_pool_manager.get_cursor(commit=False) as cursor:
            from config import config
            allowed = set(getattr(config, 'ES_INDEX_MAPPING', {}).keys())
            use_category = str(category).strip()
            if allowed and use_category not in allowed:
                return []
            sql = f"SELECT plan FROM {VECTOR_STORE_CONFIG['example_plans_table']} WHERE query = %s AND category = %s"
            cursor.execute(sql, (query.strip(), use_category))
            rows = cursor.fetchall()
            plans = [row[0] for row in rows]
            
            return plans

    except psycopg2.Error as e:
        logger.error(f"向量检索错误: {e}")
        return []

async def write_plans_to_db(query: str = "", plan: str = "", category: Optional[str] = None) -> Dict[str, str]:
    """
    将查询和计划对写入数据库的工具。
    如果查询已存在，将会被更新。
    仅写入关系型库中的示例计划表（完全匹配，不写入向量库）。
    支持分类验证：如果query包含分类标识"[分类]"，会验证分类一致性。
 
    Args:
        query: 要写入的查询。支持格式：
               - 普通查询："如何设计可插拔工具系统"
               - 分类查询："[research-planning]如何设计可插拔工具系统"
        plan: 要写入的计划
 
    Returns:
        操作结果信息
    """
    if not query.strip() or not plan.strip():
        return {"status": "error", "message": "Query and/or plan cannot be empty"}
    if not category or not str(category).strip():
        return {"status": "error", "message": "Category is required"}

    result = {"status": "success", "message": ""}

    # update the plan table
    logger.info("更新计划表")
    try:
        # 使用连接池获取数据库连接
        from db_pool_manager import db_pool_manager

        with db_pool_manager.get_cursor(commit=True) as cursor:
            from config import config
            allowed = set(getattr(config, 'ES_INDEX_MAPPING', {}).keys())
            use_category = str(category).strip()
            if allowed and use_category not in allowed:
                return {"status": "error", "message": f"无效分类: {use_category}"}
            cursor.execute(f"SELECT query, plan FROM {VECTOR_STORE_CONFIG['example_plans_table']} WHERE query = %s AND category = %s", (query.strip(), use_category))
            exact = cursor.fetchone()
            if exact:
                sql = f"UPDATE {VECTOR_STORE_CONFIG['example_plans_table']} SET plan = %s WHERE query = %s AND category = %s"
                cursor.execute(sql, (plan, query.strip(), use_category))
                result["message"] = "Query updated successfully"
            else:
                sql = f"INSERT INTO {VECTOR_STORE_CONFIG['example_plans_table']} (query, plan, category) VALUES (%s, %s, %s)"
                cursor.execute(sql, (query.strip(), plan, use_category))
                result["message"] = "Query inserted successfully"

    except psycopg2.Error as e:
        logger.error(f"数据库写入错误: {e}")
        result["status"] = "error"
        result["message"] = str(e)
        return result

@mcp.tool(exclude_args=["doc_chunks"])
@tool_config(timeout=300.0)
async def conclude_document_chunks(query: str = "", doc_chunks: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    文档块总结工具
    
    功能：对检索到的文档块进行总结，生成针对查询问题的精炼回答
    
    调用约定（重要）：
    - query 尽量包含已知的关键事实与约束条件，避免使用占位符

    Args:
        query: 用户的查询问题或总结目标
        
    Returns:
        包含总结结果的字典
        {
            "status": "success/error",
            "result": "总结的内容",
            "message": "处理信息"
        }
        
    示例:
        输入: 
        query="请基于给定资料，总结某主题的关键结论与依据。"

        输出: 
        {
            "status": "success", 
            "result": "结论要点：1. ... 2. ...", 
            "message": "成功总结2个文档块"
        }
    """
    # 检查参数
    if not query.strip() or not doc_chunks:
        return {
            "status": "error",
            "message": "查询问题和文档块不能为空（doc_chunks 由系统自动注入，调用方无需传入）",
            "result": ""
        }
    
    try:
        # 导入所需的依赖
        from custom_dashscope_llm import customDashscopeLLM
        from custom_formatter import ConclusionChatFormatter
        from config import config
        from log_config import get_mcp_logger
        
        logger = get_mcp_logger()

        # 创建LLM实例
        llm = customDashscopeLLM(
            api_key=config.DASHSCOPE_API_KEY,
            temperature=config.DETERMINISTIC_TEMPERATURE,
            top_p=config.DETERMINISTIC_TOP_P,
            context_window=config.DEFAULT_CONTEXT_WINDOW,
            max_tokens=4096  # 增加max_tokens以容纳更长的总结
        )
        
        logger.info(f"开始总结{len(doc_chunks)}个文档块，查询问题: {query}")

        content = ''
        try:
            # 格式化输入
            llm_input = CONCLUSION_PROMPT_TEMPLATE.format(query=query, doc_chunks=doc_chunks)

            response = llm.answer_gen(llm_input, streamed=False)

            # 提取回答内容
            if isinstance(response, dict) and "choices" in response:
                raw_content = response["choices"][0]["message"].get("content", "").strip()

                # 处理包含<think>标签的情况，只保留最终回复
                content = _extract_final_answer(raw_content)

        except Exception as e:
            logger.error(f"总结文档块时发生错误: {e}")
            return {
                "status": "error",
                "message": f"总结文档块时发生错误: {str(e)}",
                "result": ""
            }

        # 去除末尾多余的换行符
        conclusion = content.strip()
        logger.info(f"总结完成，最终结果: {conclusion}")
        
        return {
            "status": "success",
            "result": conclusion,
            "message": f"成功总结{len(doc_chunks)}个文档块"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"总结文档块时发生错误: {str(e)}",
            "result": ""
        }

if __name__ == "__main__":
    # 初始化数据库连接池
    from db_pool_manager import initialize_db_pool
    initialize_db_pool()
    
    mcp.run(transport="streamable-http", host=config.MCP_SERVER_HOST, port=config.MCP_SERVER_PORT)
