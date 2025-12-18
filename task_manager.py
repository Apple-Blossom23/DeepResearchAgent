"""
统一任务管理器
管理分析任务的创建、状态跟踪、结果存储和上下文管理
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

from log_config import get_task_manager_logger

logger = get_task_manager_logger()

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    status: TaskStatus
    input_text: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    current_step: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    timeout_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.timeout_at:
            data['timeout_at'] = self.timeout_at.isoformat()
        return data
    
    def is_cancelled(self) -> bool:
        """检查任务是否已取消"""
        return self.status == TaskStatus.CANCELLED
    
    def is_timeout(self) -> bool:
        """检查任务是否超时"""
        if self.timeout_at:
            return datetime.now() > self.timeout_at
        return False

class TaskManager:
    """统一任务管理器"""
    
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._timeout_check_task: Optional[asyncio.Task] = None
        
    async def create_task(self, input_text: str, created_by: str, options: Optional[Dict[str, Any]] = None) -> str:
        """创建新任务"""
        async with self._lock:
            task_id = str(uuid.uuid4())
            now = datetime.now()
            
            # 设置任务超时时间（默认30分钟）
            timeout_minutes = options.get('timeout', 30) if options else 30
            timeout_at = now + timedelta(minutes=timeout_minutes)
            
            task = TaskInfo(
                task_id=task_id,
                status=TaskStatus.PENDING,
                input_text=input_text,
                created_by=created_by,
                created_at=now,
                updated_at=now,
                options=options or {},
                timeout_at=timeout_at
            )
            
            self._tasks[task_id] = task
            logger.info(f"创建任务: {task_id}, 超时时间: {timeout_at}")
            return task_id
    
    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def update_task_status(self, task_id: str, status: TaskStatus, 
                                progress: Optional[float] = None, 
                                current_step: Optional[str] = None,
                                result: Optional[Dict[str, Any]] = None,
                                error: Optional[str] = None) -> bool:
        """更新任务状态"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            # 如果任务已经被取消，不允许更新状态
            if task.status == TaskStatus.CANCELLED and status != TaskStatus.CANCELLED:
                logger.warning(f"任务 {task_id} 已被取消，忽略状态更新: {status.value}")
                return False
            
            task.status = status
            task.updated_at = datetime.now()
            
            if progress is not None:
                task.progress = progress
            if current_step is not None:
                task.current_step = current_step
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            
            logger.info(f"更新任务状态: {task_id} -> {status.value} (进度: {task.progress}%)")
            return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                logger.warning(f"任务 {task_id} 已完成，无法取消")
                return False
            
            task.status = TaskStatus.CANCELLED
            task.updated_at = datetime.now()
            logger.info(f"取消任务: {task_id}")
            return True
    
    async def check_task_cancelled(self, task_id: str) -> bool:
        """检查任务是否被取消"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return True  # 任务不存在，视为已取消
            
            return task.is_cancelled()
    
    async def list_tasks(self, created_by: Optional[str] = None) -> List[TaskInfo]:
        """列出任务"""
        async with self._lock:
            tasks = list(self._tasks.values())
            if created_by:
                tasks = [task for task in tasks if task.created_by == created_by]
            return sorted(tasks, key=lambda x: x.created_at, reverse=True)
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧任务"""
        async with self._lock:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            tasks_to_remove = []
            
            for task_id, task in self._tasks.items():
                if task.created_at.timestamp() < cutoff_time:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                logger.info(f"清理旧任务: {task_id}")
    
    async def check_timeout_tasks(self):
        """检查超时任务"""
        async with self._lock:
            now = datetime.now()
            timeout_tasks = []
            
            for task_id, task in self._tasks.items():
                if task.is_timeout() and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    timeout_tasks.append(task_id)
            
            for task_id in timeout_tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.FAILED
                task.error = "任务超时"
                task.updated_at = now
                logger.warning(f"任务超时: {task_id}")
    
    async def start_cleanup_scheduler(self):
        """启动清理调度器"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # 每小时清理一次
                    await self.cleanup_old_tasks()
                except Exception as e:
                    logger.error(f"清理任务时出错: {e}", exc_info=True)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def start_timeout_checker(self):
        """启动超时检查器"""
        async def timeout_check_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # 每分钟检查一次超时
                    await self.check_timeout_tasks()
                except Exception as e:
                    logger.error(f"检查超时任务时出错: {e}", exc_info=True)
        
        self._timeout_check_task = asyncio.create_task(timeout_check_loop())
    
    async def stop_cleanup_scheduler(self):
        """停止清理调度器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._timeout_check_task:
            self._timeout_check_task.cancel()
            try:
                await self._timeout_check_task
            except asyncio.CancelledError:
                pass

    
    def get_task_stats(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        stats = {
            "total_tasks": len(self._tasks),
            "tasks_by_status": {}
        }
        
        for task in self._tasks.values():
            status = task.status.value
            stats["tasks_by_status"][status] = stats["tasks_by_status"].get(status, 0) + 1
        
        return stats

# 全局任务管理器实例
task_manager = TaskManager()

