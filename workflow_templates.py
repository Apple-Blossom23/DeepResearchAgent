#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用工作流程模板管理模块"""

from __future__ import annotations

from typing import Dict, Optional


_TEMPLATES: Dict[str, str] = {
    "research-general": """
## 通用研究分析工作流程
1. 明确问题与目标
2. 收集与筛选关键信息
3. 进行结构化分析与推理
4. 形成结论与可执行建议
""",
    "research-document": """
## 文档分析工作流程
1. 快速浏览与结构梳理
2. 提取关键信息与证据
3. 对比核验与补充缺口
4. 输出摘要、要点与引用
""",
    "research-problem": """
## 问题解决工作流程
1. 复述问题与约束
2. 枚举可能方案与权衡
3. 选择最优方案并细化步骤
4. 给出验证方式与回滚策略
""",
    "research-planning": """
## 规划制定工作流程
1. 明确目标、范围与里程碑
2. 识别资源、依赖与风险
3. 产出计划与执行路径
4. 定义验收标准与交付物
""",
    "research-decision": """
## 决策分析工作流程
1. 定义决策问题与评价指标
2. 收集备选方案与证据
3. 评估、打分与敏感性分析
4. 给出决策结论与行动建议
""",
    "research-analysis": """
## 深度分析工作流程
1. 收集数据与上下文
2. 识别模式、异常与关键驱动
3. 形成因果假设并验证
4. 输出洞察、结论与建议
""",
    "technical-troubleshooting": """
## 技术故障排除工作流程
1. 复现问题并收集日志/症状
2. 缩小范围并定位根因
3. 提出修复方案并验证
4. 总结经验与预防措施
""",
    "technical-evaluation": """
## 技术评估工作流程
1. 定义评估目标与范围
2. 设定指标与评价标准
3. 现状测量与对比分析
4. 输出改进建议与落地计划
""",
    "creative-brainstorming": """
## 创意头脑风暴工作流程
1. 明确需求与受众
2. 发散想法并收集候选
3. 归类筛选与组合优化
4. 输出可落地方案
""",
    "creative-design": """
## 创意设计工作流程
1. 需求与用户研究
2. 概念方案与设计原则
3. 原型迭代与评审
4. 最终方案与交付说明
""",
}


def get_workflow_template(key: str) -> Optional[str]:
    return _TEMPLATES.get(key)


def get_workflow_template_by_device_type(device_type: str, category: str = "research-general") -> str:
    return get_workflow_template(category) or get_workflow_template("research-general") or ""

