"""
工作流策略接口
定义工作流执行过程中的回调策略，实现关注点分离
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class WorkflowStrategy(ABC):
    """统一的工作流策略基类 - 定义工作流执行过程中的回调接口"""
    
    @abstractmethod
    async def on_step_start(self, step_name: str, context: Dict[str, Any]) -> None:
        """步骤开始时的回调"""
        pass
    
    @abstractmethod 
    async def on_step_complete(self, step_name: str, context: Dict[str, Any]) -> None:
        """步骤完成时的回调"""
        pass
    
    @abstractmethod
    async def on_streaming_content(self, content: str, event: str, content_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """流式内容输出 - 统一处理所有流式内容
        
        Args:
            content: 流式内容
            event: 执行阶段 (intent, entity, planning, reasoning, conclusion)
            content_type: 内容类型 (thinking, output)
            metadata: 额外的元数据
        """
        pass
    
    @abstractmethod
    async def on_tool_call_start(self, tool_name: str, arguments: Dict[str, Any], category: str = "research-general") -> None:
        """工具调用开始时的回调"""
        pass
    
    @abstractmethod
    async def on_tool_call_complete(self, tool_name: str, result: Any, category: str = "research-general") -> None:
        """工具调用完成时的回调"""
        pass
    
    @abstractmethod
    async def on_workflow_event(self, event_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """通用工作流事件 - 处理各种工作流阶段的事件"""
        pass
    
    @abstractmethod
    async def on_filter_start(self, total_chunks: int, query: str, category: str = "research-general") -> None:
        """过滤开始时的回调"""
        pass
    
    @abstractmethod
    async def on_filter_progress(self, batch_id: int, chunk_index: int, chunk_content: str, 
                                is_relevant: bool, thinking_process: str, category: str = "research-general", score: int | None = None) -> None:
        """过滤进度回调"""
        pass
    
    @abstractmethod
    async def on_filter_complete(self, total_chunks: int, relevant_count: int, 
                                filtered_count: int, category: str = "research-general") -> None:
        """过滤完成时的回调"""
        pass

class DefaultWorkflowStrategy(WorkflowStrategy):
    """默认工作流策略 - 不执行任何特殊操作，保持原有行为"""
    
    async def on_step_start(self, step_name: str, context: Dict[str, Any]) -> None:
        pass
    
    async def on_step_complete(self, step_name: str, context: Dict[str, Any]) -> None:
        pass
    
    async def on_streaming_content(self, content: str, event: str, content_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    async def on_tool_call_start(self, tool_name: str, arguments: Dict[str, Any], category: str = "research-general") -> None:
        pass
    
    async def on_tool_call_complete(self, tool_name: str, result: Any, category: str = "research-general") -> None:
        pass
    
    async def on_workflow_event(self, event_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    async def on_filter_start(self, total_chunks: int, query: str, category: str = "research-general") -> None:
        pass
    
    async def on_filter_progress(self, batch_id: int, chunk_index: int, chunk_content: str, 
                                is_relevant: bool, thinking_process: str, category: str = "research-general", score: int | None = None) -> None:
        pass
    
    async def on_filter_complete(self, total_chunks: int, relevant_count: int, 
                                filtered_count: int, category: str = "research-general") -> None:
        pass


class StreamingWorkflowStrategy(WorkflowStrategy):
    """统一的流式工作流策略 - 处理所有 SSE 事件发送"""
    
    def __init__(self, task_id: str, sse_manager, task_manager):
        self.task_id = task_id
        self.sse_manager = sse_manager
        self.task_manager = task_manager
        self.step_counter = 0
    
    async def on_step_start(self, step_name: str, context: Dict[str, Any]) -> None:
        await self.sse_manager.send_message(self.task_id, "step_start", {
            "step": step_name,
            "context": context,
            "timestamp": self._get_timestamp()
        })
    
    async def on_step_complete(self, step_name: str, context: Dict[str, Any]) -> None:
        await self.sse_manager.send_message(self.task_id, "step_complete", {
            "step": step_name,
            "context": context,
            "timestamp": self._get_timestamp()
        })
    
    async def on_streaming_content(self, content: str, event: str, content_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """统一处理所有流式内容输出
        
        Args:
            content: 流式内容
            event: 执行阶段 (intent, entity, planning, reasoning, conclusion)
            content_type: 内容类型 (thinking, output)
            metadata: 额外的元数据
        """
        event_data = {
            "content": content,
            "content_type": content_type,
            "streaming": True,
            "timestamp": self._get_timestamp()
        }
        if metadata:
            event_data.update(metadata)
        
        # 检查是否有分类信息，如果有则在event_data中包含category
        category = metadata.get('category') if metadata else None
        if category:
            # 对于并行模式，在原有事件类型中包含category信息
            event_data["category"] = category
        
        # 统一使用原有的事件类型
        await self.sse_manager.send_message(self.task_id, event, event_data)
        
        # 为流式效果添加小延迟
        import asyncio
        await asyncio.sleep(0.01)
    
    async def on_tool_call_start(self, tool_name: str, arguments: Dict[str, Any], category: str = "research-general") -> None:
        await self.sse_manager.send_message(self.task_id, "tool_call_start", {
            "tool_name": tool_name,
            "arguments": arguments,
            "category": category,
            "timestamp": self._get_timestamp()
        })
    
    async def on_tool_call_complete(self, tool_name: str, result: Any, category: str = "research-general") -> None:
        await self.sse_manager.send_message(self.task_id, "tool_call_complete", {
            "tool_name": tool_name,
            "result": str(result),
            "category": category,
            "timestamp": self._get_timestamp()
        })
    
    async def on_workflow_event(self, event_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """通用工作流事件处理"""
        event_data = {
            "content": content,
            "timestamp": self._get_timestamp()
        }
        if metadata:
            event_data.update(metadata)
        
        await self.sse_manager.send_message(self.task_id, event_type, event_data)
    
    async def on_filter_start(self, total_chunks: int, query: str, category: str = "research-general") -> None:
        """过滤开始事件"""
        await self.sse_manager.send_message(self.task_id, "filter_start", {
            "total_chunks": total_chunks,
            "query": query,
            "category": category,
            "timestamp": self._get_timestamp()
        })
    
    async def on_filter_progress(self, batch_id: int, chunk_index: int, chunk_content: str, 
                                is_relevant: bool, thinking_process: str, category: str = "research-general", score: int | None = None) -> None:
        """过滤进度事件"""
        await self.sse_manager.send_message(self.task_id, "filter_progress", {
            "batch_id": batch_id,
            "chunk_index": chunk_index,
            "chunk_content": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
            "is_relevant": is_relevant,
            "thinking_process": thinking_process,
            "category": category,
            "score": score,
            "timestamp": self._get_timestamp()
        })
    
    async def on_filter_complete(self, total_chunks: int, relevant_count: int, 
                                filtered_count: int, category: str = "research-general") -> None:
        """过滤完成事件"""
        await self.sse_manager.send_message(self.task_id, "filter_complete", {
            "total_chunks": total_chunks,
            "relevant_count": relevant_count,
            "filtered_count": filtered_count,
            "category": category,
            "timestamp": self._get_timestamp()
        })
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
