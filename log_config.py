"""
日志配置模块
提供统一的日志配置功能，支持将不同服务的日志输出到不同的文件
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

class LogConfig:
    """日志配置类"""
    
    def __init__(self):
        # 创建logs目录
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # 日志格式
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.date_format = "%Y-%m-%d %H:%M:%S"
        
        # 日志级别
        self.default_level = logging.INFO
        
    def setup_logger(self, name: str, log_file: str, level: int = None) -> logging.Logger:
        """
        设置日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
            level: 日志级别
            
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
            
        logger.setLevel(level or self.default_level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            self.log_format,
            datefmt=self.date_format
        )
        
        # 文件处理器 - 使用RotatingFileHandler进行日志轮转
        log_path = self.logs_dir / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level or self.default_level)
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level or self.default_level)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_mcp_logger(self) -> logging.Logger:
        """获取MCP代理服务器的日志记录器"""
        return self.setup_logger("mcp_proxy", "mcp.log")
    
    def get_external_api_logger(self) -> logging.Logger:
        """获取外部API服务器的日志记录器"""
        return self.setup_logger("external_api", "server.log")
    
    def get_task_manager_logger(self) -> logging.Logger:
        """获取任务管理器的日志记录器"""
        return self.setup_logger("task_manager", "server.log")
    
    def get_sse_manager_logger(self) -> logging.Logger:
        """获取SSE管理器的日志记录器"""
        return self.setup_logger("sse_manager", "server.log")
    
    def get_react_adapter_logger(self) -> logging.Logger:
        """获取ReAct适配器的日志记录器"""
        return self.setup_logger("react_adapter", "server.log")

# 全局日志配置实例
log_config = LogConfig()

# 便捷函数
def get_mcp_logger():
    """获取MCP代理服务器的日志记录器"""
    return log_config.get_mcp_logger()

def get_external_api_logger():
    """获取外部API服务器的日志记录器"""
    return log_config.get_external_api_logger()

def get_task_manager_logger():
    """获取任务管理器的日志记录器"""
    return log_config.get_task_manager_logger()

def get_sse_manager_logger():
    """获取SSE管理器的日志记录器"""
    return log_config.get_sse_manager_logger()

def get_react_adapter_logger():
    """获取ReAct适配器的日志记录器"""
    return log_config.get_react_adapter_logger()
