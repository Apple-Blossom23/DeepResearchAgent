"""
工作流状态定义
定义任务状态、交互类型等枚举和数据结构
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class TaskState(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"




class TaskContext:
    """任务上下文"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.state = TaskState.PENDING
        self.memory = None
        self.current_reasoning = []
        self.current_plan = ""
        self.sources = []
        self.plan_example = ""
        self.has_retrieved_plan_example = False
        self.workflow_position = "start"  # 工作流位置标记
        self.user_input = ""
        self.additional_input = []
        self.current_interaction_id = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.error = None
        
    def update_state(self, new_state: TaskState):
        """更新状态"""
        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now()
        logger.info(f"任务 {self.task_id} 状态从 {old_state.value} 更新为 {new_state.value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "current_plan": self.current_plan,
            "workflow_position": self.workflow_position,
            "user_input": self.user_input,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error": self.error
        }
