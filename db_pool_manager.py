"""
数据库连接池管理器
使用psycopg2的ThreadedConnectionPool实现线程安全的连接池
"""

import psycopg2
from psycopg2 import pool
from typing import Optional
import threading
from contextlib import contextmanager
from log_config import get_mcp_logger

logger = get_mcp_logger()


class DatabasePoolManager:
    """数据库连接池管理器 - 单例模式"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._connection_pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialized = False
        logger.info("DatabasePoolManager 初始化")

    def initialize_pool(self, db_config: dict, min_connections: int = 5, max_connections: int = 20):
        """
        初始化连接池

        Args:
            db_config: 数据库配置字典 {"host": ..., "port": ..., "database": ..., "user": ..., "password": ...}
            min_connections: 最小连接数
            max_connections: 最大连接数
        """
        if self._connection_pool is not None:
            logger.warning("连接池已存在，跳过初始化")
            return

        try:
            self._connection_pool = pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                host=db_config["host"],
                port=db_config.get("port", 5432),
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"]
            )
            self._initialized = True
            logger.info(f"✅ 数据库连接池初始化成功: min={min_connections}, max={max_connections}")
        except Exception as e:
            logger.error(f"❌ 数据库连接池初始化失败: {e}", exc_info=True)
            raise

    @contextmanager
    def get_connection(self):
        """
        从连接池获取连接（上下文管理器）

        使用示例:
            with db_pool_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()
                cursor.close()

        Yields:
            psycopg2连接对象
        """
        if self._connection_pool is None:
            raise RuntimeError("连接池未初始化，请先调用 initialize_pool()")

        conn = None
        try:
            conn = self._connection_pool.getconn()
            if conn:
                logger.debug(f"从连接池获取连接: {id(conn)}")
                yield conn
            else:
                raise RuntimeError("无法从连接池获取连接")
        except Exception as e:
            logger.error(f"数据库连接错误: {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                logger.debug(f"归还连接到连接池: {id(conn)}")
                self._connection_pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = True):
        """
        获取游标的便捷方法（自动管理连接和游标）

        Args:
            commit: 是否在操作成功后自动提交事务（默认True）

        使用示例:
            with db_pool_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()

        Yields:
            psycopg2游标对象
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def close_all_connections(self):
        """关闭所有连接池中的连接"""
        if self._connection_pool:
            self._connection_pool.closeall()
            self._connection_pool = None
            self._initialized = False
            logger.info("✅ 数据库连接池已关闭")

    def get_pool_status(self) -> dict:
        """
        获取连接池状态信息

        Returns:
            包含连接池状态的字典
        """
        if self._connection_pool is None:
            return {
                "initialized": False,
                "error": "连接池未初始化"
            }

        # psycopg2的ThreadedConnectionPool没有直接的状态查询方法
        # 这里返回基本信息
        return {
            "initialized": True,
            "min_connections": self._connection_pool.minconn,
            "max_connections": self._connection_pool.maxconn
        }


# 全局连接池管理器实例
db_pool_manager = DatabasePoolManager()


def initialize_db_pool():
    """
    初始化数据库连接池（应在应用启动时调用）

    使用示例:
        from config import DB_CONFIG, config
        initialize_db_pool()
    """
    from config import DB_CONFIG, config

    min_size = getattr(config, 'DB_POOL_MIN_SIZE', 5)
    max_size = getattr(config, 'DB_POOL_MAX_SIZE', 20)

    db_pool_manager.initialize_pool(
        db_config=DB_CONFIG,
        min_connections=min_size,
        max_connections=max_size
    )
    logger.info(f"数据库连接池已初始化: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")


def close_db_pool():
    """
    关闭数据库连接池（应在应用关闭时调用）
    """
    db_pool_manager.close_all_connections()
    logger.info("数据库连接池已关闭")