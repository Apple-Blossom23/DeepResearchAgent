"""
统一配置文件
支持开发环境和生产环境的配置管理
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Config:
    """基础配置类"""
    
    # 环境配置
    ENV = os.getenv("ENV", "local")  # 默认使用本地环境

    # API配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

    # 模型配置
    DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME")
    PLANNING_MODEL_NAME = os.getenv("PLANNING_MODEL_NAME")
    CONCLUSION_MODEL_NAME = os.getenv("CONCLUSION_MODEL_NAME")
    FILTER_MODEL_NAME = os.getenv("FILTER_MODEL_NAME")

    # 嵌入模型配置
    EMBEDDING_ONLINE_URL = os.getenv("EMBEDDING_ONLINE_URL")

    # 数据库配置
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "0"))
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    # 数据库连接池配置
    DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
    DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "20"))

    # MCP服务器配置
    MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST")
    MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "0"))
    MCP_SERVER_PROTOCOL = os.getenv("MCP_SERVER_PROTOCOL", "http")
    
    # MCP客户端调用配置
    MCP_CLIENT_HOST = os.getenv("MCP_CLIENT_HOST")
    MCP_CLIENT_PORT = int(os.getenv("MCP_CLIENT_PORT", "0"))
    MCP_CLIENT_PROTOCOL = os.getenv("MCP_CLIENT_PROTOCOL", "http")

    # ES检索配置
    ES_HOST = os.getenv("ES_HOST")
    ES_PORT = int(os.getenv("ES_PORT", "0"))
    ES_INDEX = os.getenv("ES_INDEX")
    ES_AUTH = os.getenv("ES_AUTH")
    ES_TIMEOUT = int(os.getenv("ES_TIMEOUT", "60"))
    ES_SEARCH_SIZE = int(os.getenv("ES_SEARCH_SIZE", "15"))  # 召回得分最高的前N个知识片段
    
    # 混合检索RRF融合参数配置
    # RRF (Reciprocal Rank Fusion) 是一种融合多个检索结果的算法
    # 公式: score(d) = Σ [1 / (k + rank_i(d))]
    # 其中 k 是平滑常数，rank_i(d) 是文档d在第i个检索结果中的排名
    
    ES_RRF_K = 10  # RRF平滑常数，推荐范围：10-100
                   # - 较小的k值(如10-30)：更重视排名靠前的结果，融合效果更激进
                   # - 较大的k值(如60-100)：排名靠后的结果也有较大贡献，融合更平滑
                   # - 典型值60：平衡了前排和后排结果的影响，是经验最佳值
    
    ES_VECTOR_CANDIDATES = 20  # 向量检索召回候选数量
                               # - 从ES索引中召回的向量检索Top-N结果数
                               # - 推荐范围：20-100，取决于索引规模和性能要求
                               # - 较大的值可以提高召回率，但增加计算开销
    
    ES_TEXT_CANDIDATES = 20    # 全文检索召回候选数量
                               # - 从ES索引中召回的BM25全文检索Top-N结果数
                               # - 推荐范围：20-100，应与向量检索候选数保持接近
                               # - 两者候选数相同可以平衡向量和文本的影响权重
    
    ES_INDEX_MAPPING = {}
    
    # 向量存储配置
    VECTOR_STORE_EMBED_DIM = 1024
    VECTOR_STORE_TABLE_NAME = "embeddings"
    STABLE_RULES_TABLE = "stable_rules"
    EXAMPLE_PLANS_TABLE = "example_plans"
    DATA_EXAMPLE_QUERIES_TABLE = "data_example_queries"

    # LLM参数配置
    DEFAULT_TEMPERATURE = 0.01
    DEFAULT_TOP_P = 0.01
    DEFAULT_CONTEXT_WINDOW = 32678
    DEFAULT_NUM_OUTPUT = 8192

    # 确定性LLM参数（用于结论、过滤等）
    DETERMINISTIC_TEMPERATURE = 0.01
    DETERMINISTIC_TOP_P = 0.01

    # 工作流程配置
    WORKFLOW_TIMEOUT = 3600  # 增加到1小时
    WORKFLOW_VERBOSE = True
    
    # 并发工作流程配置
    MAX_PARALLEL_WORKFLOWS = 3  # 最大并发工作流程数
    WORKFLOW_EXECUTION_TIMEOUT = 3600  # 单个工作流程超时时间（秒）
    ENABLE_WORKFLOW_CACHING = True  # 是否启用工作流程级别的缓存
    
    TOOL_WHITELIST_MAPPING = {
        "research-general": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "research-document": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "research-problem": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "research-planning": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "research-decision": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "research-analysis": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "technical-troubleshooting": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "technical-evaluation": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "creative-brainstorming": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "creative-design": [
            "search_documents",
            "conclude_document_chunks",
        ],
        "default": [
            "search_documents",
            "conclude_document_chunks",
        ],
    }

class DevelopmentConfig(Config):
    """开发环境配置"""
    pass

class ProductionConfig(Config):
    """生产环境配置"""
    pass

def get_config() -> Config:
    """根据环境变量获取配置"""
    env = os.getenv("ENV", "")
    
    if env == "prod":
        return ProductionConfig()
    elif env == "dev":
        return DevelopmentConfig()

    return Config()

# 全局配置实例
config = get_config()

# 为了向后兼容，提供原有的配置字典格式
DB_CONFIG = {
    "host": config.DB_HOST,
    "port": config.DB_PORT,
    "database": config.DB_NAME,
    "user": config.DB_USER,
    "password": config.DB_PASSWORD
}

VECTOR_STORE_CONFIG = {
    "embed_dim": config.VECTOR_STORE_EMBED_DIM,
    "table_name": config.VECTOR_STORE_TABLE_NAME,
    "stable_rules_table": config.STABLE_RULES_TABLE,
    "example_plans_table": config.EXAMPLE_PLANS_TABLE,
    "data_example_queries_table": config.DATA_EXAMPLE_QUERIES_TABLE
}

EMBEDDING_CONFIG = {
    "online_url": config.EMBEDDING_ONLINE_URL
}
