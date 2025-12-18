"""
外部API配置文件
配置SSE接口的相关参数
"""

import os

class ExternalAPIConfig:
    """外部API配置类"""
    
    # 服务器配置
    HOST = os.getenv("EXTERNAL_API_HOST", "0.0.0.0")
    PORT = int(os.getenv("EXTERNAL_API_PORT", "8989"))
    
    # CORS配置
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]
    
    # 认证配置
    API_KEYS = os.getenv("EXTERNAL_API_KEYS", "").split(",")
    
    # 速率限制
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # 请求限制
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
    
    # SSE配置
    SSE_TIMEOUT = int(os.getenv("SSE_TIMEOUT", "36000"))  # 1小时
    SSE_HEARTBEAT_INTERVAL = int(os.getenv("SSE_HEARTBEAT_INTERVAL", "5"))  # 5秒
    
    # 任务配置
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
    TASK_CLEANUP_INTERVAL = int(os.getenv("TASK_CLEANUP_INTERVAL", "3600"))  # 1小时
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 全局配置实例
external_config = ExternalAPIConfig()

