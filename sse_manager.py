"""
SSE流管理器
管理SSE连接和消息推送
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import logging

from log_config import get_sse_manager_logger

logger = get_sse_manager_logger()

# 尝试导入推理步骤类型，便于定制序列化
try:
    from llama_index.core.agent.react.types import (
        ActionReasoningStep,
        ObservationReasoningStep,
    )
except Exception:
    ActionReasoningStep = None
    ObservationReasoningStep = None
try:
    from custom_reasoning_step import MilestoneReasoningStep
except Exception:
    MilestoneReasoningStep = None

def _safe_json_default(obj: Any):
    """自定义JSON序列化器，处理不可直接序列化的对象"""
    # datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    # dataclass → dict
    try:
        from dataclasses import is_dataclass
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    # ActionReasoningStep
    if ActionReasoningStep and isinstance(obj, ActionReasoningStep):
        return {
            "type": "ActionReasoningStep",
            "content": getattr(obj, "get_content", lambda: None)(),
            "action": getattr(obj, "action", None),
            "action_input": getattr(obj, "action_input", None),
            "is_done": getattr(obj, "is_done", None),
            "response": getattr(obj, "response", None),
        }
    # ObservationReasoningStep
    if ObservationReasoningStep and isinstance(obj, ObservationReasoningStep):
        return {
            "type": "ObservationReasoningStep",
            "content": getattr(obj, "get_content", lambda: None)(),
            "observation": getattr(obj, "observation", None),
        }
    # MilestoneReasoningStep
    if MilestoneReasoningStep and isinstance(obj, MilestoneReasoningStep):
        return {
            "type": "MilestoneReasoningStep",
            "content": getattr(obj, "get_content", lambda: None)(),
            "milestone": getattr(obj, "milestone", None),
        }
    # 对象字典的安全转换
    if hasattr(obj, "__dict__"):
        try:
            safe = {}
            for k, v in obj.__dict__.items():
                try:
                    json.dumps(v, default=_safe_json_default)
                    safe[k] = v
                except Exception:
                    safe[k] = str(v)
            safe["type"] = obj.__class__.__name__
            return safe
        except Exception:
            pass
    # 最后降级为字符串
    return str(obj)

@dataclass
class SSEMessage:
    """SSE消息"""
    event: str
    data: Dict[str, Any]
    timestamp: datetime
    task_id: Optional[str] = None

    def to_sse_format(self) -> str:
        """转换为SSE格式"""
        try:
            message_dict = asdict(self)
            # 确保时间戳是字符串格式
            timestamp_str = message_dict['timestamp'].isoformat() if isinstance(message_dict['timestamp'], datetime) else str(message_dict['timestamp'])
            # 如果data中已经包含timestamp，则不再重复添加
            if "timestamp" not in message_dict['data']:
                message_dict['data']['timestamp'] = timestamp_str
            # 移除外层的timestamp，避免重复
            del message_dict['timestamp']
            # 返回正确的SSE格式：data: {json}\n\n
            return f"data: {json.dumps(message_dict, ensure_ascii=False, default=_safe_json_default)}\n\n"
        except Exception as e:
            # 序列化失败时优雅降级
            logger.error(f"SSE消息序列化失败: {e} [to_sse_format][UI:div tool-calls][Terminal:4/5]", exc_info=True)
            fallback = {
                "event": self.event,
                "data": {
                    "status": "error",
                    "message": "serialization failed",
                    "raw": str(self.data),
                },
                "task_id": self.task_id,
            }
            return f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"

class SSEConnection:
    """SSE连接"""
    
    def __init__(self, task_id: str, queue: asyncio.Queue):
        self.task_id = task_id
        self.queue = queue
        self.is_active = True
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
# 移除断开连接回调，简化架构
    
    async def send_message(self, message: SSEMessage):
        """发送消息"""
        if self.is_active:
            try:
                await self.queue.put(message)
                self.last_activity = datetime.now()
            except Exception as e:
                logger.error(f"发送SSE消息失败: {e}", exc_info=True)
                await self.disconnect()
    
    async def disconnect(self):
        """断开连接"""
        if self.is_active:
            self.is_active = False
            logger.info(f"SSE连接断开: {self.task_id}")
    
    def close(self):
        """关闭连接"""
        asyncio.create_task(self.disconnect())

class SSEManager:
    """SSE流管理器"""
    
    def __init__(self):
        self._connections: Dict[str, SSEConnection] = {}
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def create_connection(self, task_id: str) -> SSEConnection:
        """创建SSE连接"""
        async with self._lock:
            if task_id in self._connections:
                # 如果已存在连接，先关闭旧的
                await self.remove_connection(task_id)
            
            queue = asyncio.Queue()
            connection = SSEConnection(task_id, queue)
            
            self._connections[task_id] = connection
            self._message_queues[task_id] = queue
            
            logger.info(f"创建SSE连接: {task_id}")
            return connection
    
    async def remove_connection(self, task_id: str):
        """移除SSE连接"""
        async with self._lock:
            if task_id in self._connections:
                connection = self._connections[task_id]
                await connection.disconnect()
                del self._connections[task_id]
                
            if task_id in self._message_queues:
                del self._message_queues[task_id]
            
            logger.info(f"移除SSE连接: {task_id}")
    
# 移除连接断开处理，简化逻辑
    
    async def send_message(self, task_id: str, event: str, data: Dict[str, Any]):
        """发送消息到指定任务"""
        connection = self._connections.get(task_id)
        if connection and connection.is_active:
            message = SSEMessage(
                event=event,
                data=data,
                timestamp=datetime.now(),
                task_id=task_id
            )
            await connection.send_message(message)
            logger.debug(f"发送SSE事件: task_id={task_id}, event={event}")
        else:
            logger.warning(f"无法发送SSE消息: task_id={task_id}, event={event}, 连接不存在或已断开")
    
# 移除广播功能，简化架构只需要点对点消息
    
    async def generate_sse_stream(self, task_id: str):
        """生成SSE流"""
        connection = self._connections.get(task_id)
        if not connection:
            raise ValueError(f"任务 {task_id} 的SSE连接不存在")
        
        queue = self._message_queues[task_id]
        
        try:
            while connection.is_active:
                try:
                    # 等待消息，不设置超时，依赖心跳保持连接
                    message = await queue.get()
                    # 使用正确的SSE格式
                    try:
                        yield message.to_sse_format()
                    except Exception as se:
                        logger.error(f"SSE流序列化降级: {se} [generate_sse_stream]", exc_info=True)
                        degrade = SSEMessage(
                            event="serialization_error",
                            data={"status": "error", "message": "SSE serialization error"},
                            timestamp=datetime.now(),
                            task_id=task_id,
                        )
                        yield degrade.to_sse_format()
                except Exception as e:
                    logger.error(f"SSE流生成错误: {e}", exc_info=True)
                    break
        finally:
            # 连接断开时清理资源
            await self.remove_connection(task_id)
    
    async def start_heartbeat(self):
        """启动心跳任务"""
        async def heartbeat_loop():
            while True:
                try:
                    await asyncio.sleep(5)  # 5秒发送一次心跳
                    # 给所有活跃连接发送心跳
                    heartbeat_message = SSEMessage(
                        event="heartbeat",
                        data={"timestamp": datetime.now().isoformat()},
                        timestamp=datetime.now()
                    )
                    for connection in self._connections.values():
                        if connection.is_active:
                            await connection.send_message(heartbeat_message)
                except Exception as e:
                    logger.error(f"心跳任务错误: {e}", exc_info=True)
        
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    async def stop_heartbeat(self):
        """停止心跳任务"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
    
    def get_active_connections_count(self) -> int:
        """获取活跃连接数"""
        return len([conn for conn in self._connections.values() if conn.is_active])
    
# 移除不活跃连接清理，简化管理逻辑

# 全局SSE管理器实例
sse_manager = SSEManager()
