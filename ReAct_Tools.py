from llama_index.core.tools import FunctionTool
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

from custom_dashscope_llm import customDashscopeLLM
from custom_react_system_prompt import NER_EXTRACT_TEMPLATE
from custom_formatter import NERExtractChatFormatter
from custom_embedding import OnlineQwen3Embedding

from typing import List
import json
import psycopg2
import uuid
import requests

# 导入统一配置
from config import config, DB_CONFIG, VECTOR_STORE_CONFIG, EMBEDDING_CONFIG


# def add(x: int, y: int) -> int:
#     """Useful function to add two numbers."""
#     return x + y


# def multiply(x: int, y: int) -> int:
#     """Useful function to multiply two numbers."""
#     return x * y

# 这些函数已迁移到MCP工具中(tools.py)，这里只保留用于向后兼容的导入功能函数
# 实际的工具调用现在通过MCP协议进行

# 用于向后兼容的辅助函数
def write_plans_to_md(file_path: str, plan: str) -> None:
    """将计划写入Markdown文件的辅助函数（向后兼容）"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(plan)
    return



# 工具列表现在为空，因为所有工具都通过MCP调用
# 保留这个列表是为了向后兼容
tools = [
    # 所有工具已迁移到MCP (tools.py)
    # 现在通过MCPClient调用工具
]


