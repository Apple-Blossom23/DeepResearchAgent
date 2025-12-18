"""
分类LLM管理器
为每个工作流程分类创建和管理独立的LLM实例，确保并行执行时的完全隔离
"""

from typing import Dict, Any, Optional
from custom_dashscope_llm import customDashscopeLLM
import threading


class CategoryLLMManager:
    """分类LLM管理器 - 为每个工作流程分类创建独立的LLM实例"""
    
    def __init__(self):
        self.llm_pools: Dict[str, Dict[str, Any]] = {}  # {category: {llm_type: llm_instance}}
        self._lock = threading.Lock()  # 线程安全锁
        
    def get_category_llm(self, category: str, llm_type: str, base_llm_config: Optional[Dict] = None):
        """
        获取特定分类的LLM实例
        
        Args:
            category: 工作流程分类 (如 "research-general", "technical-troubleshooting")
            llm_type: LLM类型 (如 "main", "planning", "filter" 等)
            base_llm_config: 基础LLM配置
            
        Returns:
            对应分类和类型的LLM实例
        """
        with self._lock:
            if category not in self.llm_pools:
                self.llm_pools[category] = {}
                
            if llm_type not in self.llm_pools[category]:
                # 为每个分类创建独立的LLM实例
                self.llm_pools[category][llm_type] = self._create_llm_instance(
                    llm_type, category, base_llm_config
                )
                print(f"✅ 为分类 '{category}' 创建了独立的 {llm_type} LLM实例")
                
            return self.llm_pools[category][llm_type]
    
    def _create_llm_instance(self, llm_type: str, category: str, config: Optional[Dict] = None):
        """
        创建新的LLM实例
        
        Args:
            llm_type: LLM类型
            category: 工作流程分类
            config: 配置参数
            
        Returns:
            新的LLM实例
        """
        # 根据不同类型创建相应的LLM实例
        # 这里可以根据需要为不同类型配置不同的参数
        base_config = config or {}
        
        # 为不同分类添加标识，便于调试和监控
        instance_config = {
            **base_config,
            "category": category,
            "llm_type": llm_type
        }
        
        # 目前所有类型都使用customDashscopeLLM，但可以根据需要扩展
        if llm_type in ["main", "conclusion", "filter", "planning", "planning_judge", 
                       "plan_modify", "plan_update", "entity_recognition", "intent_recognition"]:
            return customDashscopeLLM()
        else:
            # 默认返回主LLM
            return customDashscopeLLM()
    
    def get_category_llm_pool(self, category: str) -> Dict[str, Any]:
        """
        获取特定分类的完整LLM池
        
        Args:
            category: 工作流程分类
            
        Returns:
            该分类的所有LLM实例字典
        """
        with self._lock:
            if category not in self.llm_pools:
                self.llm_pools[category] = {}
            return self.llm_pools[category].copy()
    
    def create_full_llm_set(self, category: str, base_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        为指定分类创建完整的LLM实例集合
        
        Args:
            category: 工作流程分类
            base_config: 基础配置
            
        Returns:
            包含所有必需LLM类型的字典
        """
        llm_types = [
            "main", "conclusion", "filter", "planning", "planning_judge",
            "plan_modify", "plan_update", "entity_recognition", "intent_recognition"
        ]
        
        llm_set = {}
        for llm_type in llm_types:
            llm_set[f"{llm_type}_llm"] = self.get_category_llm(category, llm_type, base_config)
        
        print(f"✅ 为分类 '{category}' 创建了完整的LLM实例集合 ({len(llm_types)} 个实例)")
        return llm_set
    
    def clear_category_llms(self, category: str):
        """
        清理指定分类的所有LLM实例
        
        Args:
            category: 要清理的工作流程分类
        """
        with self._lock:
            if category in self.llm_pools:
                del self.llm_pools[category]
                print(f"🗑️ 已清理分类 '{category}' 的所有LLM实例")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取LLM管理器的统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = {
                "total_categories": len(self.llm_pools),
                "categories": list(self.llm_pools.keys()),
                "category_details": {}
            }
            
            for category, llm_pool in self.llm_pools.items():
                stats["category_details"][category] = {
                    "llm_count": len(llm_pool),
                    "llm_types": list(llm_pool.keys())
                }
            
            return stats


# 全局LLM管理器实例
_global_llm_manager = None

def get_global_llm_manager() -> CategoryLLMManager:
    """获取全局LLM管理器实例（单例模式）"""
    global _global_llm_manager
    if _global_llm_manager is None:
        _global_llm_manager = CategoryLLMManager()
    return _global_llm_manager
