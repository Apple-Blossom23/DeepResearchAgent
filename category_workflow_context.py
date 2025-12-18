"""
分类工作流程上下文
为每个工作流程分类提供独立的上下文存储，确保数据隔离
"""

from typing import Any, Dict, Optional, List
from llama_index.core.workflow import Context
import asyncio
import threading


class CategoryWorkflowContext:
    """
    分类工作流程上下文
    为每个工作流程分类提供独立的数据存储空间
    """
    
    def __init__(self, category: str, base_context: Optional[Context] = None):
        self.category = category
        self.base_context = base_context  # 可以为None
        self._category_data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    async def set(self, key: str, value: Any):
        """设置分类专用的键值对"""
        # 先保存到内部存储
        with self._lock:
            self._category_data[key] = value
        
        # 如果base_context存在且有store，则同步到base_context
        if self.base_context and hasattr(self.base_context, 'store'):
            category_key = f"{self.category}_{key}"
            try:
                await self.base_context.store.set(category_key, value)
            except Exception:
                pass  # 忽略store操作失败
    
    async def get(self, key: str, default: Any = None) -> Any:
        """获取分类专用的值"""
        # 先尝试从本地缓存获取
        with self._lock:
            if key in self._category_data:
                return self._category_data[key]
        
        # 如果base_context存在且有store，尝试从base_context获取
        if self.base_context and hasattr(self.base_context, 'store'):
            category_key = f"{self.category}_{key}"
            try:
                value = await self.base_context.store.get(category_key, default)
                # 更新本地缓存
                with self._lock:
                    self._category_data[key] = value
                return value
            except Exception:
                pass
        
        return default
    
    async def has(self, key: str) -> bool:
        """检查是否存在指定的键"""
        category_key = f"{self.category}_{key}"
        try:
            value = await self.base_context.store.get(category_key)
            return value is not None
        except Exception:
            return False
    
    async def delete(self, key: str):
        """删除指定的键值对"""
        category_key = f"{self.category}_{key}"
        with self._lock:
            self._category_data.pop(key, None)
        # 注意：llama_index的Context可能没有delete方法，这里做保护性处理
        try:
            if hasattr(self.base_context.store, 'delete'):
                await self.base_context.store.delete(category_key)
        except Exception:
            pass
    
    async def get_all_keys(self) -> List[str]:
        """获取当前分类的所有键"""
        with self._lock:
            return list(self._category_data.keys())
    
    async def clear_category_data(self):
        """清理当前分类的所有数据"""
        keys_to_clear = await self.get_all_keys()
        for key in keys_to_clear:
            await self.delete(key)
        
        with self._lock:
            self._category_data.clear()
        
        print(f"🗑️ 已清理分类 '{self.category}' 的所有上下文数据")
    
    def get_category(self) -> str:
        """获取当前分类"""
        return self.category
    
    def get_base_context(self) -> Context:
        """获取基础上下文（用于兼容现有代码）"""
        return self.base_context
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取上下文统计信息"""
        keys = await self.get_all_keys()
        return {
            "category": self.category,
            "total_keys": len(keys),
            "keys": keys
        }


class CategoryContextManager:
    """分类上下文管理器"""
    
    def __init__(self):
        self.category_contexts: Dict[str, CategoryWorkflowContext] = {}
        self._lock = threading.Lock()
    
    def get_category_context(self, category: str, base_context: Optional[Context] = None) -> CategoryWorkflowContext:
        """
        获取或创建指定分类的上下文
        
        Args:
            category: 工作流程分类
            base_context: 基础上下文
            
        Returns:
            分类专用的工作流程上下文
        """
        with self._lock:
            if category not in self.category_contexts:
                self.category_contexts[category] = CategoryWorkflowContext(category, base_context)
                print(f"✅ 为分类 '{category}' 创建了独立的工作流程上下文")
            
            return self.category_contexts[category]
    
    async def initialize_category_context(self, category: str, user_input: str, 
                                        additional_data: Optional[Dict[str, Any]] = None) -> CategoryWorkflowContext:
        """
        初始化分类上下文的基础数据
        
        Args:
            category: 工作流程分类
            user_input: 用户输入
            additional_data: 额外的初始化数据
            
        Returns:
            初始化后的分类上下文
        """
        ctx = self.get_category_context(category)
        
        # 设置基础数据
        await ctx.set("user_input", user_input)
        await ctx.set("current_workflow_category", category)
        await ctx.set("initialization_time", asyncio.get_event_loop().time())
        
        # 设置额外数据
        if additional_data:
            for key, value in additional_data.items():
                await ctx.set(key, value)
        
        print(f"🚀 已初始化分类 '{category}' 的工作流程上下文")
        return ctx
    
    async def clear_category_context(self, category: str):
        """清理指定分类的上下文"""
        with self._lock:
            if category in self.category_contexts:
                await self.category_contexts[category].clear_category_data()
                del self.category_contexts[category]
    
    async def clear_all_contexts(self):
        """清理所有分类的上下文"""
        categories = list(self.category_contexts.keys())
        for category in categories:
            await self.clear_category_context(category)
    
    def get_all_categories(self) -> List[str]:
        """获取所有已创建的分类"""
        with self._lock:
            return list(self.category_contexts.keys())
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        categories = self.get_all_categories()
        stats = {
            "total_categories": len(categories),
            "categories": categories,
            "category_stats": {}
        }
        
        for category in categories:
            if category in self.category_contexts:
                category_stats = await self.category_contexts[category].get_stats()
                stats["category_stats"][category] = category_stats
        
        return stats


# 全局上下文管理器实例
_global_context_manager = None

def get_global_context_manager() -> CategoryContextManager:
    """获取全局上下文管理器实例（单例模式）"""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = CategoryContextManager()
    return _global_context_manager