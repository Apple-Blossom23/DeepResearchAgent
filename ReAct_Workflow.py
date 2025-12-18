from typing import Any, List, Optional, Callable
from config import config

from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from workflow_templates import get_workflow_template_by_device_type
from ReAct_Events import *
from ReAct_Tools import *
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from custom_react_system_prompt import (CUSTOM_REACT_CHAT_SYSTEM_HEADER,
                                        CUSTOM_CONTEXT_REACT_CHAT_SYSTEM_HEADER,
                                        CONCLUSION_PROMPT_TEMPLATE,
                                        FILTER_PROMPT_TEMPLATE,
                                        PLANNING_JUDGE_TEMPLATE,
                                        PLANNING_TEMPLATE,
                                        PLAN_MODIFY_TEMPLATE,
                                        PLAN_UPDATE_TEMPLATE,
                                        ENTITY_RECOGNITION_TEMPLATE,
                                        INTENT_RECOGNITION_TEMPLATE,
                                        )
from custom_formatter import (customReActChatFormatter,
                              ConclusionChatFormatter,
                              FilterChatFormatter,
                              PlanningJudgeFormatter,
                              PlanningFormatter,
                              PlanModifyFormatter,
                              PlanUpdateFormatter,
                              EntityRecognitionFormatter,
                              IntentRecognitionFormatter,
                              )
from custom_output_parser import customReActOutputParser
from custom_reasoning_step import MilestoneReasoningStep
from custom_dashscope_llm import customDashscopeLLM
from fast_mcp_client import MCPClient
from workflow_strategy import WorkflowStrategy, DefaultWorkflowStrategy
from streaming_response_parser import StreamingResponseParser
from category_llm_manager import get_global_llm_manager
from category_workflow_context import get_global_context_manager
import json, re, traceback
import asyncio
import concurrent.futures
from typing import List, Dict, Any
import copy
from dataclasses import dataclass


@dataclass
class WorkflowResult:
    """工作流程执行结果"""
    category: str
    status: str  # "completed", "failed", "timeout"
    reasoning: List[Any] = None
    sources: List[Any] = None
    error: str = None
    execution_time: float = 0.0
    response: str | None = None


class ParallelWorkflowManager:
    """增强的并行工作流程管理器，支持独立Agent实例"""
    
    def __init__(self, agent_instance):
        self.base_agent = agent_instance
        self.workflow_tasks = {}  # {workflow_category: asyncio.Task}
        self.workflow_results = {}  # {workflow_category: WorkflowResult}
        self.workflow_contexts = {}  # {workflow_category: Context}
        self.category_agents = {}  # {workflow_category: ReActAgent}
        self.llm_manager = get_global_llm_manager()
        self.context_manager = get_global_context_manager()
    
    def get_or_create_category_agent(self, category: str):
        """获取或创建指定分类的独立Agent实例"""
        if category not in self.category_agents:
            # 获取该分类的独立LLM实例
            category_llms = self.llm_manager.create_full_llm_set(category)
            
            # 创建独立的Agent实例，共享MCP客户端
            category_agent = ReActAgent(
                llm=category_llms['main_llm'],
                tools=self.base_agent.tools,
                workflow_strategy=self.base_agent.workflow_strategy,
                workflow_category=category
            )
            
            # 关键修改：共享基础Agent的MCP客户端实例，避免重复连接
            if hasattr(self.base_agent, 'mcp_client'):
                category_agent.mcp_client = self.base_agent.mcp_client
            
            # 设置分类特定的LLM实例
            category_agent.conclusion_llm = category_llms['conclusion_llm']
            category_agent.filter_llm = category_llms['filter_llm']
            category_agent.planning_llm = category_llms['planning_llm']
            category_agent.entity_recognition_llm = category_llms['entity_recognition_llm']
            category_agent.intent_recognition_llm = category_llms['intent_recognition_llm']
            # 设置其他可能缺失的LLM实例，避免NoneType错误
            category_agent.planning_judge_llm = category_llms.get('planning_judge_llm', category_llms['main_llm'])
            category_agent.plan_modify_llm = category_llms.get('plan_modify_llm', category_llms['main_llm'])
            category_agent.plan_update_llm = category_llms.get('plan_update_llm', category_llms['main_llm'])
            
            self.category_agents[category] = category_agent
            print(f"✅ 为分类 '{category}' 创建独立Agent实例 (共享MCP客户端)")
        
        return self.category_agents[category]
        
    async def create_workflow_context(self, base_ctx: Context, category: str) -> Context:
        """为每个工作流程创建独立的上下文"""
        # 获取该分类的独立Agent实例
        category_agent = self.get_or_create_category_agent(category)
        
        # 创建新的上下文实例，使用分类特定的代理实例
        workflow_ctx = Context(workflow=category_agent)
        
        # 复制基础数据
        base_data = {
            "recognized_entities": await base_ctx.store.get("recognized_entities"),
            "user_input": await base_ctx.store.get("user_input"),
            "memory": await base_ctx.store.get("memory"),
            "input_metadata": await base_ctx.store.get("input_metadata"),
            "workflow_categories": await base_ctx.store.get("workflow_categories", default=[]),
        }
        
        for key, value in base_data.items():
            if value is not None:
                # 深拷贝以避免上下文间的数据污染
                copied_value = copy.deepcopy(value) if isinstance(value, (list, dict)) else value
                await workflow_ctx.store.set(key, copied_value)
        
        # 设置工作流程特定的标识
        await workflow_ctx.store.set("current_workflow_category", category)
        await workflow_ctx.store.set("workflow_id", f"{category}_{id(workflow_ctx)}")
        await workflow_ctx.store.set("current_reasoning", [])
        await workflow_ctx.store.set("sources", [])
        
        # 使用分类上下文管理器初始化上下文
        category_context = self.context_manager.get_category_context(category, workflow_ctx)
        # 设置分类特定的初始化数据
        await category_context.set("workflow_initialized", True)
        await category_context.set("workflow_start_time", asyncio.get_event_loop().time())
        
        return workflow_ctx
    
    async def execute_single_workflow(self, workflow_ctx: Context, ev: EntityAnalysisEvent, category: str) -> WorkflowResult:
        """执行单个工作流程"""
        import time
        start_time = time.time()
        
        try:
            print(f"\n🚀 启动并行工作流程: {category}")
            
            # 获取分类特定的Agent实例
            category_agent = self.get_or_create_category_agent(category)
            
            # 选择工作流模板
            try:
                selected_template = category_agent._select_workflow_template(ev.recognized_entities, category)
                await workflow_ctx.store.set("selected_workflow_template", selected_template)
                print(f"📄 {category} 选择的工作流模板: {selected_template[:50]}...")
            except Exception as e:
                print(f"⚠️ {category} 选择模板失败，使用默认模板: {e}")
                await workflow_ctx.store.set("selected_workflow_template", "通用故障处理流程")
            
            # 执行工作流程的完整生命周期
            # 增加超时控制，避免单个工作流卡死整个并行组
            # 注意：这里的timeout应该比总的并行超时时间短一些，或者由外层控制
            # 这里我们不加wait_for，让外层的gather统一控制超时
            result = await self._run_workflow_lifecycle(workflow_ctx, ev, category)
            
            execution_time = time.time() - start_time
            print(f"✅ {category} 工作流程完成，耗时: {execution_time:.2f}秒")
            
            return WorkflowResult(
                category=category,
                status="completed",
                reasoning=result.get("reasoning", []),
                sources=result.get("sources", []),
                response=result.get("response"),
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏰ {category} 工作流程超时，耗时: {execution_time:.2f}秒")
            return WorkflowResult(
                category=category,
                status="timeout",
                error="工作流程执行超时",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {category} 工作流程失败: {str(e)}")
            return WorkflowResult(
                category=category,
                status="failed",
                error=str(e),
                execution_time=execution_time
            )
    
    async def _run_workflow_lifecycle(self, workflow_ctx: Context, ev: EntityAnalysisEvent, category: str) -> Dict[str, Any]:
        """运行单个工作流程的完整生命周期"""
        try:
            # 获取分类特定的Agent实例
            category_agent = self.get_or_create_category_agent(category)
            
            # 1. 生成计划
            user_input = await workflow_ctx.store.get("user_input")
            planning_event = PlanningEvent(input=user_input, additional_input=[])
            
            # 调用分类特定agent的generate_plan方法
            input_event = await category_agent.generate_plan(workflow_ctx, planning_event)
            
            # 2. 准备聊天历史
            prep_event = PrepEvent(input=user_input)
            input_event = await category_agent.prepare_chat_history(workflow_ctx, prep_event)
            
            # 3. 处理LLM输入，使用动态终止条件
            current_reasoning = []
            sources = []
            iteration = 0
            final_response = None
            while True:
                try:
                    # 处理LLM输入
                    result = await category_agent.handle_llm_input(workflow_ctx, input_event)
                    
                    if isinstance(result, StopEvent):
                        # 工作流程完成，提取最终响应
                        try:
                            res_obj = getattr(result, "result", {}) or {}
                            final_response = res_obj.get("response")
                        except Exception:
                            final_response = None
                        break
                    elif isinstance(result, ToolCallEvent):
                        # 处理工具调用
                        prep_event = await category_agent.handle_tool_calls(workflow_ctx, result)
                        input_event = await category_agent.prepare_chat_history(workflow_ctx, prep_event)
                        
                        # 收集推理步骤
                        reasoning_steps = await workflow_ctx.store.get("current_reasoning", default=[])
                        if reasoning_steps:
                            current_reasoning.extend(reasoning_steps)
                        
                        # 收集数据源
                        current_sources = await workflow_ctx.store.get("sources", default=[])
                        if current_sources:
                            sources.extend(current_sources)
                    iteration += 1
                
                except Exception as e:
                    print(f"⚠️ {category} 工作流程第 {iteration} 次迭代出错: {str(e)}")
                    break
            
            # 获取最终结果
            final_reasoning = await workflow_ctx.store.get("current_reasoning", default=current_reasoning)
            final_sources = await workflow_ctx.store.get("sources", default=sources)
            
            # 如果没有获取到结果，提供默认结果
            if not final_reasoning:
                final_reasoning = [f"{category} 工作流程执行完成"]
            if not final_sources:
                final_sources = [f"{category} 相关数据源"]
            
            return {
                "reasoning": final_reasoning,
                "sources": final_sources,
                "response": final_response,
            }
            
        except Exception as e:
            print(f"❌ {category} 工作流程生命周期执行失败: {str(e)}")
            # 返回错误信息作为结果
            return {
                "reasoning": [f"{category} 工作流程执行失败: {str(e)}"],
                "sources": []
            }
    
    async def execute_parallel_workflows(
        self, 
        base_ctx: Context, 
        ev: EntityAnalysisEvent, 
        categories: List[str], 
        timeout: float = 30.0,
        on_thinking: Optional[Callable] = None,
        on_content: Optional[Callable] = None
    ) -> Dict[str, WorkflowResult]:
        """
        并行执行多个工作流程分类
        
        Args:
            base_ctx: 基础上下文
            ev: 实体分析事件
            categories: 工作流程分类列表
            timeout: 超时时间（秒）
            on_thinking: 思考过程回调函数
            on_content: 内容回调函数
        
        Returns:
            Dict[str, WorkflowResult]: 各分类的工作流程结果
        """
        print(f"\n🔄 开始并行执行工作流程，分类: {categories}")
        
        # 获取基础Agent的流式响应解析器
        response_parser = self.base_agent.response_parser
        
        # 为每个分类创建独立的上下文和任务
        tasks = {}
        streaming_responses = {}
        
        for category in categories:
            try:
                # 创建分类特定的上下文
                workflow_ctx = await self.create_workflow_context(base_ctx, category)
                self.workflow_contexts[category] = workflow_ctx
                
                # 获取分类特定的Agent
                category_agent = self.get_or_create_category_agent(category)
                
                # 创建异步任务
                task = asyncio.create_task(
                    self.execute_single_workflow(workflow_ctx, ev, category)
                )
                tasks[category] = task
                self.workflow_tasks[category] = task
                
                # 存储流式响应
                if hasattr(category_agent, 'streaming_response'):
                    streaming_responses[category] = category_agent.streaming_response
                
                print(f"📋 为分类 '{category}' 创建执行任务")
                
            except Exception as e:
                print(f"❌ 创建分类 '{category}' 的任务失败: {str(e)}")
                self.workflow_results[category] = WorkflowResult(
                    category=category,
                    status="failed",
                    error=f"任务创建失败: {str(e)}"
                )
        
        # 处理流式响应
        if streaming_responses and on_thinking and on_content:
            try:
                # 创建流式响应解析任务
                streaming_task = asyncio.create_task(
                    response_parser.parse_parallel_streaming_response(
                        streaming_responses,
                        on_thinking,
                        on_content
                    )
                )
            except Exception as e:
                print(f"⚠️ 创建流式响应解析任务失败: {str(e)}")
                streaming_task = None
        else:
            streaming_task = None
        
        # 等待所有任务完成或超时
        if tasks:
            try:
                print(f"⏱️ 等待所有工作流程完成，超时时间: {timeout}秒")
                # 使用shield防止外部取消影响内部任务，同时确保wait_for生效
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
                
                # 处理结果
                for i, (category, task) in enumerate(tasks.items()):
                    if i < len(results):
                        result = results[i]
                        if isinstance(result, Exception):
                            # 区分取消异常和其他异常
                            if isinstance(result, asyncio.CancelledError):
                                print(f"⚠️ 分类 '{category}' 任务被取消")
                                self.workflow_results[category] = WorkflowResult(
                                    category=category,
                                    status="cancelled",
                                    error="任务被取消"
                                )
                            else:
                                print(f"❌ 分类 '{category}' 执行异常: {str(result)}")
                                self.workflow_results[category] = WorkflowResult(
                                    category=category,
                                    status="failed",
                                    error=str(result)
                                )
                        else:
                            self.workflow_results[category] = result
                    else:
                        self.workflow_results[category] = WorkflowResult(
                            category=category,
                            status="failed",
                            error="未获取到执行结果"
                        )
                        
            except asyncio.TimeoutError:
                print(f"⏰ 并行执行超时，取消未完成的任务")
                # 取消未完成的任务
                for category, task in tasks.items():
                    if not task.done():
                        task.cancel()
                        # 尝试等待任务取消完成，避免悬挂
                        try:
                            # 给一点时间让任务响应取消
                            # await asyncio.wait_for(task, timeout=2.0) 
                            # 注意：这里不能await task，因为task可能因为无法响应取消而卡住
                            pass
                        except:
                            pass
                            
                        self.workflow_results[category] = WorkflowResult(
                            category=category,
                            status="timeout",
                            error="执行超时"
                        )
            except Exception as e:
                print(f"❌ 并行执行发生未预期错误: {str(e)}")
                for category in categories:
                    if category not in self.workflow_results:
                        self.workflow_results[category] = WorkflowResult(
                            category=category,
                            status="failed",
                            error=f"执行器错误: {str(e)}"
                        )
            
            # 等待流式响应解析任务完成
            if streaming_task:
                try:
                    # 激进的清理策略：无论状态如何，强制取消
                    if not streaming_task.done():
                        streaming_task.cancel()
                        # 尝试等待取消完成，但设置极短超时
                        try:
                            # 增加一个极短的wait，让EventLoop有机会处理cancel信号
                            await asyncio.wait_for(streaming_task, timeout=0.1)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                        except Exception as e:
                            print(f"⚠️ 流式任务取消时发生异常: {str(e)}")
                            
                    # 确保完全释放引用
                    streaming_task = None
                except Exception as e:
                    print(f"⚠️ 流式响应解析清理异常: {str(e)}")
                        
        # 打印执行摘要
        print(f"\n📊 并行执行完成摘要:")
        for category, result in self.workflow_results.items():
            status_emoji = "✅" if result.status == "completed" else "❌" if result.status == "failed" else "⏰"
            print(f"  {status_emoji} {category}: {result.status} ({result.execution_time:.2f}s)")
            
        return self.workflow_results
    
    def clear_results(self):
        """清理执行结果和上下文"""
        self.workflow_tasks.clear()
        self.workflow_results.clear()
        self.workflow_contexts.clear()
        # 清理分类上下文
        self.context_manager.clear_all_contexts()
        print("🧹 已清理所有工作流程执行结果和上下文")


class ReActAgent(Workflow):
    def __init__(
            self,
            *args: Any,
            llm: LLM | None = None,
            tools: list[BaseTool] | None = None,
            extra_context: str | None = None,
            react_chat_system_header: str | None = CUSTOM_REACT_CHAT_SYSTEM_HEADER,
            context_react_chat_system_header: str | None = CUSTOM_CONTEXT_REACT_CHAT_SYSTEM_HEADER,
            conclusion_prompt: str | None = CONCLUSION_PROMPT_TEMPLATE,
            filter_prompt: str | None = FILTER_PROMPT_TEMPLATE,
            planning_judge_prompt: str | None = PLANNING_JUDGE_TEMPLATE,
            planning_prompt: str | None = PLANNING_TEMPLATE,
            plan_modify_prompt=PLAN_MODIFY_TEMPLATE,
            plan_update_prompt=PLAN_UPDATE_TEMPLATE,
            conclusion_llm: LLM | None = None,
            filter_llm: LLM | None = None,
            planning_llm: LLM | None = None,
            planning_jugde_llm: LLM | None = None,
            plan_modify_llm: LLM | None = None,
            plan_update_llm: LLM | None = None,
            entity_recognition_llm: LLM | None = None,
            intent_recognition_llm: LLM | None = None,
            workflow_strategy: WorkflowStrategy | None = None,
            workflow_category: str | None = None,  # 新增：工作流程分类参数
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or customDashscopeLLM()  # 使用自定义LLM替代OpenAI
        self.mcp_client = MCPClient()  # 添加MCP客户端
        self.workflow_strategy = workflow_strategy or DefaultWorkflowStrategy()  # 默认策略
        self.step_counter = 0  # 步骤计数器
        
        # 设置工作流程分类（将在意图识别中动态调整）
        self.workflow_category = workflow_category
        
        # self.formatter = ReActChatFormatter.from_defaults(
        #     context=extra_context or ""
        # )
        self.formatter = customReActChatFormatter(
            react_chat_system_header=react_chat_system_header,
            context_react_chat_system_header=context_react_chat_system_header).from_custom(
            context=extra_context or "",
        )
        self.output_parser = customReActOutputParser()

        # define formatters
        self.filter_formatter = FilterChatFormatter()
        self.conclusion_formatter = ConclusionChatFormatter()
        self.planning_judge_formatter = PlanningJudgeFormatter()
        self.planning_formatter = PlanningFormatter()
        self.plan_modify_formatter = PlanModifyFormatter()
        self.plan_update_formatter = PlanUpdateFormatter()
        self.entity_recognition_formatter = EntityRecognitionFormatter()
        self.intent_recognition_formatter = IntentRecognitionFormatter()

        # define llms - 确保所有LLM实例都有默认值，避免NoneType错误
        self.conclusion_llm = conclusion_llm or self.llm
        self.filter_llm = filter_llm or self.llm
        self.planning_llm = planning_llm or self.llm
        self.planning_judge_llm = planning_jugde_llm or self.llm
        self.plan_modify_llm = plan_modify_llm or self.llm
        self.plan_update_llm = plan_update_llm or self.llm
        self.entity_recognition_llm = entity_recognition_llm or self.llm  # 如果没有指定，使用默认LLM
        self.intent_recognition_llm = intent_recognition_llm or self.llm  # 如果没有指定，使用默认LLM
        
        # 创建流式响应解析器
        self.response_parser = StreamingResponseParser()

        # 线程安全：为filter_llm调用添加互斥锁
        import threading
        self._filter_llm_lock = threading.Lock()
        
        # 初始化并行工作流程管理器
        self.parallel_manager = ParallelWorkflowManager(self)
    
    def _filter_single_chunk_sync(self, chunk: str, query: str, chunk_index: int, batch_id: int, local_filter_llm) -> tuple[str, bool, int]:
        """
        同步方式过滤单个文档块（在线程池中运行）
        
        Args:
            chunk: 要过滤的文档块
            query: 用户查询
            chunk_index: 块索引
            batch_id: 批次ID
            
        Returns:
            (chunk, is_relevant) 元组
        """
        import datetime
        import asyncio
        
        try:
            # 格式化输入 - 检查返回类型并转换为字符串
            llm_input = self.filter_formatter.format(query=query, doc_chunks=chunk)
            
            # 如果返回的是ChatMessage列表，转换为字符串
            if isinstance(llm_input, list):
                # 将ChatMessage列表转换为字符串
                formatted_text = ""
                for msg in llm_input:
                    if hasattr(msg, 'content'):
                        formatted_text += f"{msg.role}: {msg.content}\n"
                    else:
                        formatted_text += str(msg) + "\n"
                llm_input_text = formatted_text.strip()
            else:
                llm_input_text = str(llm_input)
                
            print(f"  📤 线程{batch_id}-块{chunk_index+1} 开始LLM调用 [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]")
                
            # 使用同步方式调用LLM - 直接传递字符串（使用线程专属实例）
            response = local_filter_llm.complete(llm_input_text)
            response_content = response.text
            final_content = self.response_parser.extract_answer(response_content)
            thinking_process = self.response_parser.extract_thinking(response_content)


            # 判断相关性
            is_relevant = "相关" in final_content and "无关" not in final_content
            import re
            score_match = re.search(r"SCORE\s*:\s*(\d{1,3})", final_content)
            score_json_match = re.search(r"\{\s*\"score\"\s*:\s*(\d{1,3})\s*\}", final_content)
            score_val = None
            if score_match:
                try:
                    score_val = int(score_match.group(1))
                except Exception:
                    score_val = None
            elif score_json_match:
                try:
                    score_val = int(score_json_match.group(1))
                except Exception:
                    score_val = None
            if score_val is None:
                score_val = 80 if is_relevant else 20
            if score_val < 0:
                score_val = 0
            if score_val > 100:
                score_val = 100
                
            if is_relevant:
                print(f"✅线程 {batch_id} - 块{chunk_index+1}: 相关 ，score：{score_val} [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]")
                print(f"  🧠 思考过程: {thinking_process}")
                print(f"  📚 知识片段: {chunk}")
            else:
                print(f"❌线程 {batch_id} - 块{chunk_index+1}: 无关 ，score：{score_val} [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]")
                print(f"  🧠 思考过程: {thinking_process}")
                print(f"  📚 知识片段: {chunk}")
                
            try:
                if hasattr(self, "_event_loop") and hasattr(self, "workflow_strategy"):
                    asyncio.run_coroutine_threadsafe(
                        self.workflow_strategy.on_filter_progress(
                            batch_id,
                            chunk_index,
                            chunk,
                            is_relevant,
                            thinking_process,
                            getattr(self, "_current_category", ""),
                            score_val
                        ),
                        self._event_loop
                    )
            except Exception:
                pass
            return chunk, is_relevant, score_val
                
        except Exception as e:
            print(f"⚠️ 线程 {batch_id} - 块{chunk_index+1} 处理失败: {e}")
            return chunk, True, 50
    
    def _filter_batch_in_thread_sync(self, chunks: List[str], query: str, batch_id: int) -> List[tuple[str, int]]:
        """
        在单独线程中同步处理一批文档块的相关性过滤
        这个方法会在独立的线程中运行，真正实现批次间的并行
        
        Args:
            chunks: 要过滤的文档块列表
            query: 用户查询
            batch_id: 批次ID，用于日志
            
        Returns:
            相关的文档块列表
        """
        # 检查filter_llm是否可用
        if not self.filter_llm:
            print(f"线程 {batch_id}: filter_llm 未配置，返回所有块")
            return chunks
        
        import datetime, threading
        from custom_dashscope_llm import customDashscopeLLM
        from config import config

        # 为该线程创建独立的LLM实例，避免共享客户端带来的线程安全问题
        local_filter_llm = customDashscopeLLM(
            model_code=getattr(config, 'FILTER_MODEL_NAME', config.DEFAULT_MODEL_NAME),
            api_key=config.DASHSCOPE_API_KEY,
            temperature=getattr(config, 'DEFAULT_TEMPERATURE', 0.01),
            top_p=getattr(config, 'DEFAULT_TOP_P', 0.01),
            context_window=getattr(config, 'DEFAULT_CONTEXT_WINDOW', 16384),
            max_tokens=getattr(config, 'DEFAULT_NUM_OUTPUT', 4096)
        )

        thread_name = threading.current_thread().name
        print(f"🔍 线程 {batch_id} ({thread_name}) 开始处理 {len(chunks)} 个文档块... [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]")
        
        relevant_chunks_with_scores = []
        
        # 在这个线程中顺序处理每个chunk
        for i, chunk in enumerate(chunks):
            try:
                chunk_result, is_relevant, score_val = self._filter_single_chunk_sync(chunk, query, i, batch_id, local_filter_llm)
                if is_relevant:
                    relevant_chunks_with_scores.append((chunk_result, score_val))
                    print(f"  ✓ 线程 {batch_id} - 块{i+1}: 相关")
                else:
                    print(f"  ✗ 线程 {batch_id} - 块{i+1}: 无关")
            except Exception as e:
                print(f"⚠️ 线程 {batch_id} - 块{i+1} 处理失败: {e}")
                relevant_chunks_with_scores.append((chunk, 50))
        
        print(f"✅ 线程 {batch_id} ({thread_name}) 完成，筛选出 {len(relevant_chunks_with_scores)}/{len(chunks)} 个相关块")
        relevant_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        return relevant_chunks_with_scores
    
    async def _filter_chunk_batch_with_threads(self, chunks: List[str], query: str, batch_id: int) -> List[str]:
        """
        异步包装器，用于在线程池中执行同步批次处理
        """
        loop = asyncio.get_event_loop()
        self._event_loop = loop
        self._current_category = ""
        return await loop.run_in_executor(
            None,  # 使用默认线程池
            self._filter_batch_in_thread_sync,
            chunks, query, batch_id
        )
    
    async def _filter_chunks_parallel(self, doc_chunks: List[str], query: str, category: str = "research-general") -> List[str]:
        """
        真正并行过滤文档块的相关性
        每个批次在独立的线程中运行，实现真正的线程级并行
        
        Args:
            doc_chunks: 所有文档块
            query: 用户查询
            category: 工作流分类
            
        Returns:
            过滤后的相关文档块列表
        """
        if not doc_chunks:
            return []
        
        import datetime
        start_time = datetime.datetime.now()
        
        # 发送过滤开始事件
        await self.workflow_strategy.on_filter_start(len(doc_chunks), query, category)
        
        # 每个线程处理的chunk数量
        chunks_per_thread = 3
        # 最多使用的线程数
        max_threads = 3
        
        # 计算实际需要的线程数
        total_chunks = len(doc_chunks)
        needed_threads = min(max_threads, (total_chunks + chunks_per_thread - 1) // chunks_per_thread)
        
        # 将文档块均匀分配给线程
        batches = []
        if needed_threads == 1:
            # 如果只需要一个线程，直接处理所有块
            batches.append(doc_chunks)
        else:
            # 计算每个线程的实际负载
            base_size = total_chunks // needed_threads
            remainder = total_chunks % needed_threads
            
            start_idx = 0
            for i in range(needed_threads):
                # 前面的线程多处理一个chunk（如果有余数）
                current_size = base_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_size
                if start_idx < total_chunks:
                    batches.append(doc_chunks[start_idx:end_idx])
                start_idx = end_idx
        
        print(f"📊 文档块总数: {total_chunks}, 使用 {len(batches)} 个线程并行处理")
        for i, batch in enumerate(batches):
            print(f"  线程 {i+1}: {len(batch)} 个文档块")
        
        # 使用asyncio的线程池执行器实现真正的线程并行
        loop = asyncio.get_event_loop()
        
        # 创建线程任务
        tasks = []
        for i, batch in enumerate(batches):
            if batch:  # 确保批次不为空
                # 每个批次在独立线程中运行
                task = loop.run_in_executor(
                    None,  # 使用默认线程池
                    self._filter_batch_in_thread_sync,
                    batch, query, i + 1
                )
                tasks.append(task)
        
        # 等待所有线程完成
        if tasks:
            print(f"🚀 启动 {len(tasks)} 个并行线程...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 合并结果
            all_relevant_chunks_with_scores = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ 线程 {i+1} 执行失败: {result}")
                    if i < len(batches):
                        for c in batches[i]:
                            all_relevant_chunks_with_scores.append((c, 50))
                else:
                    all_relevant_chunks_with_scores.extend(result)
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            filtered_count = total_chunks - len(all_relevant_chunks_with_scores)
            print(f"✅ 所有线程完成！总耗时: {duration:.2f}秒, 筛选出 {len(all_relevant_chunks_with_scores)}/{total_chunks} 个相关块")
            
            # 发送过滤完成事件
            await self.workflow_strategy.on_filter_complete(
                total_chunks, 
                len(all_relevant_chunks_with_scores), 
                filtered_count, 
                category
            )
            all_relevant_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
            return [c for c, s in all_relevant_chunks_with_scores]
        else:
            # 没有任务，返回空列表
            await self.workflow_strategy.on_filter_complete(0, 0, 0, category)
            return []
    
    async def get_mcp_tool_descriptions(self, category: str = None) -> str:
        """获取MCP工具描述，支持基于category的工具过滤"""
        try:
            # 确保MCP客户端已连接
            await self.mcp_client.ensure_connected()
            
            # 获取工具列表
            response = await self.mcp_client.list_tools()
            
            # 获取当前category对应的白名单工具列表（仅展示允许的工具）
            allowed_tools = self._get_allowed_tools_for_category(category)

            # 格式化工具描述，仅包含白名单中的工具；若未配置白名单则展示全部
            tool_descriptions = []
            for tool in response:
                tool_name = tool['name']
                
                # 白名单生效：仅当在白名单中时展示；未配置白名单（None）则展示全部
                if (isinstance(allowed_tools, list) and tool_name in allowed_tools) or (allowed_tools is None):
                    desc = f"Tool Name: {tool_name}\n"
                    desc += f"Description: {tool['description']}\n"
                    if 'inputSchema' in tool and tool['inputSchema']:
                        desc += f"Parameters: {tool['inputSchema']}\n"
                    tool_descriptions.append(desc)

            return "\n".join(tool_descriptions)
        except Exception as e:
            print(f"Error getting MCP tool descriptions: {e}")
            return "No tools available"
    
    def _get_allowed_tools_for_category(self, category: str = None) -> list | None:
        """根据category获取允许使用的工具列表（白名单模式）。未配置则返回None代表允许全部。"""
        from config import config
        
        mapping = getattr(config, 'TOOL_WHITELIST_MAPPING', {})
        if not category or category not in mapping:
            return mapping.get("default", None)
        return mapping[category]

    async def format_with_mcp_tools(self, chat_history, current_reasoning, current_plan, tool_descriptions, ctx=None, category=None):
        """使用MCP工具描述格式化聊天历史，支持基于category的工具过滤"""
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        
        # 从上下文获取元数据信息
        metadata = {}
        if ctx:
            metadata = await ctx.store.get("input_metadata", default={})
        
        # 构建格式化参数，直接使用tool_descriptions
        format_args = {
            "tool_desc": tool_descriptions,
            "tool_names": "MCP工具",  # 占位符，因为我们使用MCP
            "current_plan": current_plan,
        }
        
        # 如果有元数据，添加到上下文中
        if metadata:
            format_args["metadata_context"] = metadata
        else:
            format_args["metadata_context"] = "无额外信息"
        
        if self.formatter.context:
            format_args["context"] = self.formatter.context

        fmt_sys_header = self.formatter.system_header.format(**format_args)

        # 格式化推理历史
        reasoning_history = []
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ChatMessage(
                    role=MessageRole.TOOL,
                    content=reasoning_step.get_content(),
                )
            else:
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=reasoning_step.get_content(),
                )
            reasoning_history.append(message)

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *chat_history,
            *reasoning_history,
        ]

    
    async def _merge_entities_to_metadata(self, ctx: Context, recognized_entities: list[dict]) -> None:
        """
        将识别出的实体信息合并到元数据中
        如果实体识别的结果包含更准确的信息，会覆盖原有的空值或null值
        """
        if not recognized_entities:
            return
            
        # 获取现有的元数据
        current_metadata = await ctx.store.get("input_metadata", default={})
        
        # 创建实体信息列表
        entity_info_list = []
        enhanced_metadata = current_metadata.copy()
        
        for entity in recognized_entities:
            device_name = entity.get("device_name", "")
            device_type = entity.get("device_type", "")
            fault_type = entity.get("fault_type", "")
            voltage_level = entity.get("voltage_level", "")
            
            # 构建实体信息字符串
            entity_info = []
            if device_name:
                entity_info.append(f"设备名称: {device_name}")
            if device_type:
                entity_info.append(f"设备类型: {device_type}")
            if fault_type:
                entity_info.append(f"故障类型: {fault_type}")
            if voltage_level:
                entity_info.append(f"电压等级: {voltage_level}")
                
            if entity_info:
                entity_info_list.append(", ".join(entity_info))
            
            # 如果识别出的设备名称不为空，且原元数据中的dev_name为空或null，则覆盖
            if device_name and (not enhanced_metadata.get("dev_name") or enhanced_metadata.get("dev_name") == "null"):
                enhanced_metadata["dev_name"] = device_name
                
            # 如果识别出了故障类型，且原元数据中没有或为空，则覆盖
            if fault_type and not enhanced_metadata.get("fault_type1"):
                enhanced_metadata["fault_type1"] = fault_type
                

        
        # 将识别出的实体信息添加到元数据中
        if entity_info_list:
            enhanced_metadata["recognized_entities"] = entity_info_list
            
        # 更新上下文中的元数据
        await ctx.store.set("input_metadata", enhanced_metadata)
        
        if enhanced_metadata.get("dev_name") != current_metadata.get("dev_name"):
            print(f"  设备名称已更新: {current_metadata.get('dev_name', 'null')} -> {enhanced_metadata.get('dev_name')}")

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> IntentRecognitionEvent:
        await self.workflow_strategy.on_step_start("new_user_msg", {
            "input": ev.input
        })
        
        # JSON输入解析
        parsed_data = self._parse_user_input(ev.input)
        user_input = parsed_data["input"]
        metadata = parsed_data.get("metadata", {})
        attachments = parsed_data.get("attachments", [])
        
        # 将解析后的元数据存储到上下文中，供后续推理使用
        await ctx.store.set("input_metadata", metadata)
        await ctx.store.set("input_attachments", attachments)
        await ctx.store.set("user_input", user_input)  # 存储用户输入到上下文

        
        # 正常的工作流处理逻辑
        # clear sources
        await ctx.store.set("sources", [])
        await ctx.store.set("plan_example", "")

        # init memory if needed
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # clear current reasoning and current plan
        await ctx.store.set("current_reasoning", [])
        await ctx.store.set("current_plan", "")

        # clear some indicators
        await ctx.store.set("has_retrieved_plan_example", False)

        # set memory
        await ctx.store.set("memory", memory)
        
        result = IntentRecognitionEvent(input=user_input)
        
        await self.workflow_strategy.on_step_complete("new_user_msg", {
            "user_input": user_input,
            "metadata": metadata
        })
        
        return result
    
    def _parse_user_input(self, user_input: str) -> dict:
        """
        解析用户输入，支持JSON格式和普通文本格式
        """
        # 尝试解析为JSON
        try:
            # 如果输入看起来像JSON字符串，尝试解析
            if user_input.strip().startswith('{') and user_input.strip().endswith('}'):
                input_data = json.loads(user_input)
                return self._process_json_data(input_data)
        except json.JSONDecodeError:
            pass
        
        # 如果不是JSON或解析失败，按普通文本处理
        return {
            "input": user_input,
            "metadata": {},
            "attachments": []
        }
    
    def _process_json_data(self, input_data: dict) -> dict:
        """
        处理JSON数据，转换时间戳并重新格式化
        支持三种格式：
        1. 通用格式: {"input": "...", "metadata": {...}, "attachments": [...]}
        2. 兼容格式: {"query": "...", "metadata": {...}, "attachments": [...]}
        3. 遗留格式兼容: 自动映射旧字段到通用metadata结构
        """
        import time
        from datetime import datetime
        
        def format_date(input_str):
            """格式化日期字符串"""
            if not input_str:
                return ""
            
            # 判断是否符合yyyy-MM-dd HH:mm:ss格式
            try:
                datetime.strptime(str(input_str), "%Y-%m-%d %H:%M:%S")
                return str(input_str)  # 如果符合日期格式，直接返回
            except ValueError:
                pass
            
            # 尝试解析为时间戳
            try:
                timestamp = int(input_str)
                # 判断时间戳长度（秒级还是毫秒级）
                if len(str(timestamp)) == 13:  # 毫秒级时间戳
                    timestamp = timestamp / 1000
                elif len(str(timestamp)) == 10:  # 秒级时间戳
                    pass  # 保持原样
                return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            except (ValueError, OSError):
                return str(input_str)  # 如果都失败，返回原字符串

        # 通用格式优先
        if isinstance(input_data, dict):
            if "input" in input_data or "metadata" in input_data or "attachments" in input_data:
                metadata = input_data.get("metadata", {})
                attachments = input_data.get("attachments", [])
                return {
                    "input": str(input_data.get("input") or ""),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "attachments": attachments if isinstance(attachments, list) else [],
                }
            if "query" in input_data and ("metadata" in input_data or "attachments" in input_data):
                metadata = input_data.get("metadata", {})
                attachments = input_data.get("attachments", [])
                return {
                    "input": str(input_data.get("query") or ""),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "attachments": attachments if isinstance(attachments, list) else [],
                }

        # 旧格式兼容：保留旧字段名映射，但不再绑定为“故障”语义
        legacy_metadata = {
            "event_time": format_date(input_data.get("occurTime", "")),
            "event_id": input_data.get("faultId", ""),
            "source_device_id": input_data.get("devId", ""),
            "source_device_name": input_data.get("devName", ""),
        }
        # 清理空值
        legacy_metadata = {k: v for k, v in legacy_metadata.items() if v}

        return {
            "input": str(input_data.get("faultDescr", "") or ""),
            "metadata": legacy_metadata,
            "attachments": input_data.get("attachments", []) if isinstance(input_data.get("attachments", []), list) else [],
        }
    
    @step
    async def intent_recognition(self, ctx: Context, ev: IntentRecognitionEvent) -> EntityRecognitionEvent | StopEvent:
        """
        意图识别步骤：使用大模型判断是否需要快速响应
        """
        await self.workflow_strategy.on_step_start("intent_recognition", {
            "input": ev.input
        })
        
        user_input = ev.input
        
        try:
            # 使用意图识别模板调用LLM
            llm_input = self.intent_recognition_formatter.format(user_input)
            
            print(f"⚡️请求意图识别模型响应...")
            response_gen = await self.intent_recognition_llm.astream_chat(messages=llm_input)
            
            # 定义异步回调函数
            async def on_intent_thinking(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                thinking_metadata = {"step": "意图识别思考"}
                if current_category:
                    thinking_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "intent", "thinking", thinking_metadata
                )
            
            async def on_intent_content(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                content_metadata = {"step": "意图识别输出"}
                if current_category:
                    content_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "intent", "output", content_metadata
                )
            
            # 使用StreamingResponseParser解析流式响应
            response_content = await self.response_parser.parse_streaming_response(
                response_gen,
                on_thinking=on_intent_thinking,
                on_content=on_intent_content,
                thinking_metadata={"step": "意图识别思考", "phase": "intent_recognition"},
                content_metadata={"step": "意图识别输出", "phase": "intent_recognition"}
            )


            final_content = self.response_parser.extract_final_content(response_content)
            
            # 解析JSON响应
            import re
            json_match = re.search(r'```json(.*?)```', final_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                # 如果没有找到JSON代码块，尝试直接解析
                json_content = final_content.strip()
            
            # 解析JSON
            try:
                intent_result = json.loads(json_content)
                is_quick_response = intent_result.get("is_quick_response", False)
                standard_answer = intent_result.get("standard_answer", "")
                workflow_categories = intent_result.get("workflow_categories", ["research-general"])
                
                # 将工作流程分类信息存储到上下文中
                await ctx.store.set("workflow_categories", workflow_categories)
                
                print(f"🔍 意图识别结果: 工作流程分类={workflow_categories}")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ 意图识别JSON解析失败: {e}")
                print(f"响应内容: {response_content}")
                is_quick_response = False
                standard_answer = ""
                workflow_categories = []  # 默认为空列表，不设置默认分类
                await ctx.store.set("workflow_categories", workflow_categories)
            
            # 如果需要快速响应，直接返回结果
            if is_quick_response and standard_answer:
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                event_metadata = {
                    "standard_answer": standard_answer,
                    "user_input": user_input
                }
                if current_category:
                    event_metadata["category"] = current_category
                
                await self.workflow_strategy.on_workflow_event("quick_response_triggered", 
                    f"触发快速响应 {standard_answer[:50]}...", event_metadata)
                
                await self.workflow_strategy.on_step_complete("intent_recognition", {
                    "quick_response_triggered": True,
                    "standard_answer": standard_answer
                })
                
                # 直接返回StopEvent，结束工作流
                return StopEvent(
                    result={
                        "response": standard_answer,
                        "sources": [],
                        "reasoning": [],
                        "quick_response": True
                    }
                )
            
            # 如果不需要快速响应，继续正常流程
            # 获取当前工作流分类
            current_category = await ctx.store.get("current_workflow_category", default=None)
            event_metadata = {
                "quick_response_triggered": False,
                "user_input": user_input
            }
            if current_category:
                event_metadata["category"] = current_category
                
            await self.workflow_strategy.on_workflow_event("intent_recognition_complete", 
                "意图识别完成，进入正常流程", event_metadata)
            
            await self.workflow_strategy.on_step_complete("intent_recognition", {
                "quick_response_triggered": False,
                "continue_normal_flow": True
            })
            
            return EntityRecognitionEvent(input=user_input)
            
        except Exception as e:
            print(f"⚠️ 意图识别失败: {e}")
            await self.workflow_strategy.on_step_complete("intent_recognition", {
                "error": str(e),
                "fallback_to_normal": True
            })
            
            return EntityRecognitionEvent(input=user_input)

    @step
    async def entity_recognition(self, ctx: Context, ev: EntityRecognitionEvent) -> EntityAnalysisEvent | StopEvent:
        """
        实体识别步骤：识别用户输入中的设备并选择合适的工作流模板
        """
        await self.workflow_strategy.on_step_start("entity_recognition", {
            "input": ev.input
        })
        
        # 正常的实体识别处理
        user_input = ev.input
        
        try:
            # 使用实体识别模板调用LLM
            llm_input = self.entity_recognition_formatter.format(user_input)
            
            print(f"⚡️请求实体识别模型响应...")
            response_gen = await self.entity_recognition_llm.astream_chat(messages=llm_input)
            
            # 定义异步回调函数
            async def on_entity_thinking(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                thinking_metadata = {"step": "实体识别思考"}
                if current_category:
                    thinking_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "entity", "thinking", thinking_metadata
                )
            
            async def on_entity_content(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                content_metadata = {"step": "实体识别输出"}
                if current_category:
                    content_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "entity", "output", content_metadata
                )
            
            # 使用StreamingResponseParser解析流式响应
            response_content = await self.response_parser.parse_streaming_response(
                response_gen,
                on_thinking=on_entity_thinking,
                on_content=on_entity_content,
                thinking_metadata={"step": "实体识别思考", "phase": "entity_recognition"},
                content_metadata={"step": "实体识别输出", "phase": "entity_recognition"}
            )
            
            # 解析JSON响应
            import re
            final_content = self.response_parser.extract_final_content(response_content)
            # 提取JSON代码块中的内容
            # 匹配带或不带```json标记的JSON内容
            json_match = re.search(r'```json\n?(.*?)\n?```|^\s*(\[.*\])\s*$', final_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                # 如果没有找到JSON代码块，尝试直接解析响应内容
                json_content = final_content.strip()

            
            # 解析JSON
            try:
                recognized_entities = json.loads(json_content)
                # 确保结果是列表类型
                if not isinstance(recognized_entities, list):
                    if isinstance(recognized_entities, dict):
                        recognized_entities = [recognized_entities]
                    else:
                        recognized_entities = []
            except json.JSONDecodeError as e:
                print(f"\n⚠️ JSON解析失败: {e}")
                print(f"预处理后的JSON内容: {json_content}")
                print(f"原始响应内容: {response_content}")
                recognized_entities = []

            workflow_categories = await ctx.store.get("workflow_categories", default=["research-general"])
            if not workflow_categories:
                # 即使没有识别出分类，也保留为空列表，由后续逻辑处理
                workflow_categories = []
            
            if workflow_categories:
                current_category = workflow_categories[0]
                await ctx.store.set("current_workflow_category", current_category)
                selected_workflow_template = self._select_workflow_template(recognized_entities, current_category)
            else:
                current_category = None
                await ctx.store.set("current_workflow_category", None)
                selected_workflow_template = "" # 不使用任何模板

            
            # 将识别出的实体信息合并到元数据中
            await self._merge_entities_to_metadata(ctx, recognized_entities)
            
            # 发送实体识别完成事件
            # 获取当前工作流分类
            current_category = await ctx.store.get("current_workflow_category", default=None)
            event_metadata = {
                "entities_count": len(recognized_entities),
                "recognized_entities": recognized_entities,
                "selected_workflow_template": selected_workflow_template
            }
            if current_category:
                event_metadata["category"] = current_category
                
            await self.workflow_strategy.on_workflow_event("entity_recognition_complete", 
                f"实体识别完成，识别出 {len(recognized_entities)} 个实体", event_metadata)
            
            await self.workflow_strategy.on_step_complete("entity_recognition", {
                "recognized_entities": recognized_entities,
                "selected_workflow_template": selected_workflow_template,
                "entities_count": len(recognized_entities)
            })
            
            return EntityAnalysisEvent(
                input=user_input,
                recognized_entities=recognized_entities,
                selected_workflow_template=selected_workflow_template
            )
            
        except Exception as e:
            print(f"\n⚠️ 实体识别失败: {e}")
            # 如果实体识别失败，使用默认模板
            default_template = get_workflow_template_by_device_type("其他设备", self.workflow_category)
            
            await self.workflow_strategy.on_step_complete("entity_recognition", {
                "error": str(e),
                "fallback_to_default": True,
                "selected_workflow_template": default_template
            })
            
            return EntityAnalysisEvent(
                input=user_input,
                recognized_entities=[],
                selected_workflow_template=default_template
            )
    
    def _select_workflow_template(self, recognized_entities: list[dict], current_workflow_category: str = None) -> str:
        """
        根据识别出的实体和当前工作流程分类选择合适的工作流模板
        """
        # 使用传入的工作流程分类，如果没有则使用默认的
        workflow_category = current_workflow_category or self.workflow_category
        
        if not recognized_entities:
            # 如果没有识别出实体，使用默认模板
            return get_workflow_template_by_device_type("其他设备", workflow_category)
        
        # 统计设备类型
        device_type_counts = {}
        for entity in recognized_entities:
            device_type = entity.get("device_type", "其他设备")
            device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1
        
        # 选择出现次数最多的设备类型
        if device_type_counts:
            primary_device_type = max(device_type_counts, key=device_type_counts.get)
            return get_workflow_template_by_device_type(primary_device_type, workflow_category)



    @step
    async def check_valid_plan(
            self, ctx: Context, ev: EntityAnalysisEvent
    ) -> PlanningEvent | StopEvent:
        '''
        In this function, we use the user query to judge whether the current plan is valid.
        We also store the entity analysis results for later use in planning.
        We support parallel execution of multiple workflow categories.
        '''

        # 将实体分析结果存储到上下文中
        await ctx.store.set("recognized_entities", ev.recognized_entities)
        await ctx.store.set("selected_workflow_template", ev.selected_workflow_template)

        print(f"\n🔍 识别出的设备: {len(ev.recognized_entities)} 个")
        for entity in ev.recognized_entities:
            print(f"  - {entity.get('device_name', '')}: {entity.get('device_type', '')}, {entity.get('fault_type', '无')}")
        print(f"\n📄 选择的工作流模板: {ev.selected_workflow_template}...")

        # 获取工作流程分类
        workflow_categories = await ctx.store.get("workflow_categories", default=["research-general"])
        if not workflow_categories:
            # 如果没有分类，也不默认回退到默认分类，保持为空
            workflow_categories = []
        
        # 检查是否需要并行执行多个工作流程分类
        if len(workflow_categories) > 1:
            print(f"\n🚀 检测到多个工作流程分类: {workflow_categories}")
            print("🔄 启动并行执行模式...")
            
            try:
                # 使用并行管理器执行多个工作流程
                # 注意：execute_parallel_workflows 内部使用 asyncio.create_task 创建任务
                # 但在此处我们需要等待所有任务完成才能返回结果
                # 因此这里实际上是"并发"执行，当前主流程会等待并发结果
                parallel_results = await self.parallel_manager.execute_parallel_workflows(
                    ctx, ev, workflow_categories, timeout=config.WORKFLOW_EXECUTION_TIMEOUT,
                    on_thinking=self._on_parallel_thinking,
                    on_content=self._on_parallel_content
                )
                
                # 合并并行执行的结果
                combined_reasoning = []
                combined_sources = []
                
                for category, result in parallel_results.items():
                    if result.status == "completed":
                        if result.reasoning:
                            combined_reasoning.extend([f"[{category}] {r}" for r in result.reasoning])
                        if result.sources:
                            combined_sources.extend([f"[{category}] {s}" for s in result.sources])
                    else:
                        # 处理失败或超时的情况
                        error_msg = f"[{category}] 执行{result.status}: {result.error or '未知错误'}"
                        combined_reasoning.append(error_msg)
                
                # 返回合并后的结果
                return StopEvent(
                    result={
                        "response": f"并行执行完成，处理了 {len(workflow_categories)} 个工作流程分类",
                        "reasoning": combined_reasoning,
                        "sources": combined_sources,
                        "parallel_results": parallel_results,
                        "workflow_category": "综合" # 标记为综合结果
                    }
                )
            except Exception as e:
                print(f"❌ 并行执行出错: {str(e)}")
                # 出错时返回部分结果或错误信息
                return StopEvent(
                    result={
                        "response": f"并行执行部分失败: {str(e)}",
                        "reasoning": [f"执行错误: {str(e)}"],
                        "sources": [],
                        "workflow_category": "综合"
                    }
                )

        # 单个工作流程的情况，继续执行原有逻辑
        # 设置当前工作流程分类
        if workflow_categories:
            current_category = workflow_categories[0]
            await ctx.store.set("current_workflow_category", current_category)
        else:
            # 如果没有分类，则不设置当前分类，也不回退到默认分类
            current_category = None
            await ctx.store.set("current_workflow_category", None)
            print("⚠️ 未识别到工作流程分类，将不使用特定流程模板")
        
        return PlanningEvent(input=ev.input, additional_input=[])


    @step
    async def generate_plan(
            self, ctx: Context, ev: PlanningEvent
    ) -> InputEvent:
        await self.workflow_strategy.on_step_start("generate_plan", {
            "input": ev.input,
            "additional_input": ev.additional_input
        })
        '''
        In this function, we are sure that currently we do not have a valid plan accepted by the user.
        So we need to generate a plan given the user query and the chat history.
        After gnerating the plan, we need to tell the user, and wait for their feedback.
        Return the user feedback to ConciergeEvent, where the feedback is judged. Also, store the current plan in the context for later use.
        '''

        # get the memory
        memory = await ctx.store.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        usr_query = memory.get("usr_msg", default=None)

        # Get the user feedback, could be empty which means either this is the first round, or the user didn't provide any feedback.
        # In both empty cases, we generate a new plan.
        user_feedback = ev.additional_input

        query = ev.input  # get the user query

        # TODO: generate the plan and ask the user for feedback.

        # To generate the plan, we need prepare the following components:
        # 1. query
        # 2. current plan, if any; together with the user feedback
        # 3. some plan stored in the database/history as example
        # 4. tool list, optional
        query = ev.input
        current_plan = await ctx.store.get("current_plan", default="")  # get the current plan from the context
        has_retrieved_plan_example = await ctx.store.get("has_retrieved_plan_example",
                                                   default=False)  # flag to avoid always no example is retrieved, but we still try to retrieve.

        # 获取计划示例，如果有的话
        plan_example = await ctx.store.get("plan_example", default="")
        if not plan_example and not has_retrieved_plan_example:
            try:
                from tools import retrieve_plans_from_db
                current_category = await ctx.store.get("current_workflow_category", default=None)
                plan_example = await retrieve_plans_from_db(query, category=current_category)
            except Exception as e:
                print(f"Error retrieving plan examples: {e}")
                plan_example = []
            await ctx.store.set("has_retrieved_plan_example", True)

            # if we have similar plan examples. Note there is situiation where we have no similar examples given threshold.
            # we store the plan example in the context, no need for further retrieval
            if len(plan_example) > 0:
                await ctx.store.set("plan_example", plan_example)

        # generate plan
        generated_plan = ""
        # if current_plan and user_feedback:
            # if there is current_plan, we just modify it according to the user feedback
            # modification_input = self.plan_modify_formatter.format(
            #     current_plan=current_plan,
            #     modify_suggestion=user_feedback,
            # )

            # print(f"⌛️请求{config.PLANNING_MODEL_NAME}模型响应...")
            # response_gen = await self.plan_modify_llm.astream_chat(messages=modification_input)

            # async for response in response_gen:
            #     if hasattr(response, 'delta') and response.delta:
            #         print(response.delta, end='', flush=True)
            #
            # # Extract the plan from the complete response
            # generated_plan = response.message.content.split("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")[-1]

        if current_plan:
            # 用户没有提供反馈，直接返回现有计划并开始执行
            # user does not provide feedback for current plan, just return it. Ask user to comment on current plan.
            # print("⬇️我现在的方案如下。你还没有对现在的方案作出反馈，如果你需要任何改动，请在开始研究前告诉我：\n\n",
            #       current_plan)
            # question = "是否开始研究？"
            # user_feedback = input(question + "\n\n>")
            # return ConciergeEvent(input=usr_query[0].content, additional_input=[user_feedback])
            
            print("✅使用现有计划，开始执行...")
            return InputEvent()

        else:
            # there is no current plan
            assert (usr_query is not None)

            # get tool description from MCP
            current_workflow_category = await ctx.store.get("current_workflow_category", default=None)
            tool_desc = await self.get_mcp_tool_descriptions(current_workflow_category)
            metadata_context = await ctx.store.get("input_metadata", "")
            # metadata_context
            # 使用从实体识别中获得的动态工作流模板
            selected_workflow_template = await ctx.store.get("selected_workflow_template", default="")

            
            llm_input = self.planning_formatter.format(
                query=usr_query,
                plan_examples=plan_example,
                tool_desc=tool_desc,
                workflow=selected_workflow_template,  # 使用动态选择的工作流程模板
                metadata_context=metadata_context,
            )

            response_gen = await self.planning_llm.astream_chat(
                messages=llm_input
            )

            # 定义异步回调函数
            async def on_planning_thinking(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                thinking_metadata = {"step": "计划生成思考"}
                if current_category:
                    thinking_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "planning", "thinking", thinking_metadata
                )
            
            async def on_planning_content(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                content_metadata = {"step": "计划生成输出"}
                if current_category:
                    content_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "planning", "output", content_metadata
                )
            
            # 使用StreamingResponseParser解析流式响应
            response_content = await self.response_parser.parse_streaming_response(
                response_gen,
                on_thinking=on_planning_thinking,
                on_content=on_planning_content,
                thinking_metadata={"step": "计划生成思考", "phase": "planning"},
                content_metadata={"step": "计划生成输出", "phase": "planning"}
            )

            # 从响应中提取计划
            generated_plan = self.response_parser.extract_final_content(response_content)

        # write current plan to context
        await ctx.store.set("current_plan", generated_plan)
        current_plan = generated_plan

        # 将生成的计划写入数据库，供后续检索使用
        if generated_plan.strip():
            try:
                from tools import write_plans_to_db
                current_category = await ctx.store.get("current_workflow_category", default=None)
                await write_plans_to_db(query, generated_plan, category=current_category)
                print("✅计划已写入数据库")
            except Exception as e:
                print(f"⚠️ 写入计划到数据库失败: {e}")

        # 发送计划生成完成事件
        # 获取当前工作流分类
        current_category = await ctx.store.get("current_workflow_category", default=None)
        event_metadata = {
            "plan_generated": bool(generated_plan),
            "plan_content": generated_plan[:200] + "..." if len(generated_plan) > 200 else generated_plan
        }
        if current_category:
            event_metadata["category"] = current_category
            
        await self.workflow_strategy.on_workflow_event("plan_generation_complete", 
            "计划生成完成", event_metadata)
        
        await self.workflow_strategy.on_step_complete("generate_plan", {
            "plan_generated": bool(generated_plan)
        })

        # 注释掉用户反馈逻辑，直接开始执行
        # TODO: ask user for feedback
        # print("⬇️这是我拟定的方案。如果你需要进行任何改动，请在我开始研究前告诉我。\n\n", current_plan)
        # question = "是否开始研究？"
        # usr_response = await ctx.wait_for_event(
        #     HumanResponseEvent,
        #     waiter_id=question,
        #     waiter_event=InputRequiredEvent(
        #         prefix=question,
        #     ),
        # )

        # # collect the user feedback
        # user_feedback += [usr_response.response]

        # user_feedback = input(question + "\n\n>")
        
        # 直接返回PrepEvent，跳过用户确认
        print("✅计划生成完成，开始执行...")
        return PrepEvent()

    # 注释掉handle_user_feedback方法，因为不再需要用户确认
    # @step
    # async def handle_user_feedback(
    #         self, ctx: Context, ev: ConciergeEvent
    # ) -> PrepEvent | PlanningEvent:
    #     """
    #     处理用户反馈，判断用户是否接受计划并决定下一步行动
    #     """
    #     await self.workflow_strategy.on_step_start("handle_user_feedback", {
    #         "input": ev.input,
    #         "additional_input": ev.additional_input
    #     })
    #     
    #     user_feedback = ev.additional_input[0] if ev.additional_input else ""
    #     
    #     # 使用planning judge来判断用户是否接受计划
    #     llm_input = self.planning_judge_formatter.format(query=user_feedback)
    #     
    #     print(f"⌛️请求{config.PLANNING_MODEL_NAME}模型判断用户反馈...")
    #     response_gen = await self.planning_judge_llm.astream_chat(messages=llm_input)
    #     
    #     async for response in response_gen:
    #         if hasattr(response, 'delta') and response.delta:
    #             print(response.delta, end='', flush=True)
    #     
    #     judge_result = response.message.content.strip()
    #     
    #     if "肯定" in judge_result or "开始" in user_feedback or "是" in user_feedback:
    #         # 用户接受计划，开始执行
    #         await self.workflow_strategy.on_step_complete("handle_user_feedback", {
    #             "decision": "accepted",
    #             "user_feedback": user_feedback
    #         })
    #         return PrepEvent()
    #     else:
    #         # 用户要求修改计划，重新生成
    #         await self.workflow_strategy.on_step_complete("handle_user_feedback", {
    #             "decision": "rejected",
    #             "user_feedback": user_feedback
    #         })
    #         return PlanningEvent(input=ev.input, additional_input=ev.additional_input)

    # @step
    # async def fetch_usr_feedback(
    #     self, ctx: Context, ev: InputRequiredEvent
    # ) -> HumanResponseEvent:
    #     usr_feedback = input(">" + ev.prefix)
    #     return HumanResponseEvent(response=usr_feedback)

    # @step
    # async def send_usr_feedback(
    #     self, ctx: Context, ev: HumanResponseEvent
    # ) -> ConciergeEvent:
    #     memory = await ctx.store.get("memory", default=None)
    #     query = memory.get("usr_msg", default=None)
    #     return ConciergeEvent(input=query, additional_input=[ev.response])

    @step
    async def prepare_chat_history(
            self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        # get chat history
        memory = await ctx.store.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        current_plan = await ctx.store.get("current_plan", default="")
        
        # 从上下文获取当前工作流程分类
        current_workflow_category = await ctx.store.get("current_workflow_category", default=None)

        # 如果current_plan为空，说明还没有生成计划，直接进入执行阶段
        if not current_plan.strip():
            print("⚠️ 当前没有执行计划，直接进入执行阶段")
            # format the prompt with react instructions
            # 传递上下文参数给formatter，包括category
            llm_input = await self.format_with_mcp_tools(chat_history, current_reasoning, current_plan,
                                                    await self.get_mcp_tool_descriptions(current_workflow_category), 
                                                    ctx, current_workflow_category)

            return InputEvent(input=llm_input)

        if current_reasoning:
            # call llm to update plan
            update_llm_input = self.plan_update_formatter.format(
                current_plan=current_plan,
                current_reasoning=current_reasoning,
                chat_history=chat_history,
            )

            print(f"⌛️请求{config.PLANNING_MODEL_NAME}模型响应...")
            response_gen = await self.plan_update_llm.astream_chat(
                messages=update_llm_input,
            )

            # 定义异步回调函数
            async def on_planning_thinking(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                thinking_metadata = {"step": "计划更新思考"}
                if current_category:
                    thinking_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "planning", "thinking", thinking_metadata
                )

            async def on_planning_content(content, metadata):
                # 获取当前工作流分类
                current_category = await ctx.store.get("current_workflow_category", default=None)
                content_metadata = {"step": "计划更新输出"}
                if current_category:
                    content_metadata["category"] = current_category
                await self.workflow_strategy.on_streaming_content(
                    content, "planning", "output", content_metadata
                )

            # 使用StreamingResponseParser解析流式响应
            response_content = await self.response_parser.parse_streaming_response(
                response_gen,
                on_thinking=on_planning_thinking,
                on_content=on_planning_content,
                thinking_metadata={"step": "计划更新思考", "phase": "planning"},
                content_metadata={"step": "计划更新输出", "phase": "planning"}
            )

            generated_plan = self.response_parser.extract_final_content(response_content)

            # Extract the plan from the complete response
            if generated_plan.strip():
                current_plan = generated_plan
            # update the plan according to current_reasoning steps
            print("✅Update plan.md")
            write_plans_to_md("./plan.md", current_plan)  # write the plan to a markdown file

        print("\n⬇️Current plan: \n\n", current_plan)
        # format the prompt with react instructions
        tool_descriptions = await self.get_mcp_tool_descriptions(current_workflow_category)
        # 由于formatter期望BaseTool列表，我们需要创建一个特殊的format调用
        llm_input = await self.format_with_mcp_tools(chat_history, current_reasoning, current_plan, tool_descriptions, ctx, current_workflow_category)

        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
            self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        await self.workflow_strategy.on_step_start("llm_reasoning", {
            "input_length": len(ev.input)
        })
        chat_history = ev.input
        # print("\n" + "=" * 20 + "chat history" + "=" * 20 + "\n")
        # print(chat_history)
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        memory = await ctx.store.get("memory")

        print(f"⌛️请求{config.DEFAULT_MODEL_NAME}模型响应...")
        
        # 获取当前工作流分类
        current_category = await ctx.store.get("current_workflow_category", default=None)
        event_metadata = {
            "step": "llm_request"
        }
        if current_category:
            event_metadata["category"] = current_category
            
        await self.workflow_strategy.on_workflow_event("llm_request", "正在向模型发送推理请求...", event_metadata)
        
        response_gen = await self.llm.astream_chat(chat_history)
        
        # 定义异步回调函数
        async def on_llm_thinking(content, metadata):
            # 获取当前工作流分类
            current_category = await ctx.store.get("current_workflow_category", default=None)
            thinking_metadata = {"step": "LLM推理思考"}
            if current_category:
                thinking_metadata["category"] = current_category
            await self.workflow_strategy.on_streaming_content(
                content, "llm", "thinking", thinking_metadata
            )
        
        async def on_llm_content(content, metadata):
            # 获取当前工作流分类
            current_category = await ctx.store.get("current_workflow_category", default=None)
            content_metadata = {"step": "LLM推理回复"}
            if current_category:
                content_metadata["category"] = current_category
            await self.workflow_strategy.on_streaming_content(
                content, "llm", "output", content_metadata
            )
        
        # 使用StreamingResponseParser解析流式响应
        response_content = await self.response_parser.parse_streaming_response(
            response_gen,
            on_thinking=on_llm_thinking,
            on_content=on_llm_content,
            thinking_metadata={"step": "LLM推理思考", "phase": "reasoning"},
            content_metadata={"step": "LLM推理回复", "phase": "reasoning"}
        )
        
        try:
            # 从响应中提取答案主体
            answer_body = self.response_parser.extract_final_content(response_content)

            reasoning_step = self.output_parser.parse(answer_body)
            current_reasoning.append(reasoning_step)
            print("\n" + "=" * 20 + "ReAct参数" + "=" * 20 + "\n")
            print(reasoning_step.get_content())

            if reasoning_step.is_done:
                memory.put(
                    ChatMessage(
                        role="assistant", content=reasoning_step.response
                    )
                )
                await ctx.store.set("memory", memory)
                await ctx.store.set("current_reasoning", current_reasoning)

                sources = await ctx.store.get("sources", default=[])

                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [sources],
                        "reasoning": current_reasoning,
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
            elif isinstance(reasoning_step, MilestoneReasoningStep):
                await ctx.store.set("current_reasoning", current_reasoning)
                return PrepEvent()
        except Exception as e:
            current_reasoning.append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
            await ctx.store.set("current_reasoning", current_reasoning)

        # if no tool calls or final response, iterate again
        return PrepEvent()

    @step
    async def handle_tool_calls(
            self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        tool_calls = ev.tool_calls
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        sources = await ctx.store.get("sources", default=[])

        # 确保MCP客户端已连接
        await self.mcp_client.ensure_connected()

        # 通过MCP调用工具
        for tool_call in tool_calls:
            # 对总结工具提前注入参数，避免由大模型构造大入参
            if tool_call.tool_name == "conclude_document_chunks":
                try:
                    cached_chunks = await ctx.store.get("relevant_doc_chunks", default=[])
                except Exception:
                    cached_chunks = []

                conclude_kwargs = dict(tool_call.tool_kwargs or {})
                if "query" not in conclude_kwargs:
                    memory = await ctx.store.get("memory")
                    user_msgs = [m for m in memory.get() if getattr(m, "role", "") == "user"]
                    conclude_kwargs["query"] = user_msgs[-1].content if user_msgs else ""
                
                # 注入doc_chunks参数
                if cached_chunks:
                    conclude_kwargs["doc_chunks"] = cached_chunks

                # 回写到当前工具调用参数，以便后续走统一调用流程
                tool_call.tool_kwargs = conclude_kwargs
            
            # 为文档检索工具注入category参数
            if tool_call.tool_name == "search_documents":
                tool_kwargs = dict(tool_call.tool_kwargs or {})
                if "category" not in tool_kwargs:
                    # 从上下文获取当前工作流分类
                    try:
                        current_workflow_category = await ctx.store.get(
                            "current_workflow_category", default="research-general"
                        )
                        tool_kwargs["category"] = current_workflow_category
                    except Exception:
                        # 如果获取失败，使用默认值
                        tool_kwargs["category"] = "research-general"

                # 回写到当前工具调用参数
                tool_call.tool_kwargs = tool_kwargs

            # 发送工具调用开始事件
            current_workflow_category = await ctx.store.get("current_workflow_category", default="research-general")
            print(f"🟢 发送tool_call_start事件: tool={tool_call.tool_name}, category={current_workflow_category}")
            await self.workflow_strategy.on_tool_call_start(tool_call.tool_name, tool_call.tool_kwargs, current_workflow_category)



            # 使用MCP调用工具
            result = await self.mcp_client.call_tool(
                tool_call.tool_name,
                tool_call.tool_kwargs
            )
            print(tool_call.tool_name + "工具返回结果：" + str(result))

            # 发送工具调用完成事件
            print(f"🟢 发送tool_call_complete事件: tool={tool_call.tool_name}, category={current_workflow_category}")
            await self.workflow_strategy.on_tool_call_complete(tool_call.tool_name, result, current_workflow_category)
            # 处理MCP返回结果
            if isinstance(result, dict) and "result" in result:
                tool_output_text = result["result"]
            else:
                tool_output_text = str(result)

            # 特殊处理：文档检索类工具
            if tool_call.tool_name == "search_documents":
                # 从字符串中提取所有被包裹的 JSON 文本
                json_strings = re.findall(
                    r"TextContent\(type='text',\s*text='((?:\\'|[^'])*)',\s*annotations=None(?:,\s*meta=None)?\)",
                    tool_output_text,
                    re.DOTALL,
                )

                doc_chunks: List[str] = []
                parsed = None
                if json_strings:
                    try:
                        raw_text = json_strings[0]
                        safe_text = raw_text.replace('\\', '\\\\')
                        safe_text = safe_text.replace('\\\\n', '\\n')
                        safe_text = safe_text.replace('\\\\t', '\\t')
                        safe_text = safe_text.replace('\\\\r', '\\r')
                        safe_text = safe_text.replace('\\\\"', '\\"')
                        parsed = json.loads(safe_text)
                    except Exception:
                        parsed = None
                else:
                    try:
                        parsed = json.loads(tool_output_text)
                    except Exception:
                        try:
                            import ast
                            parsed = ast.literal_eval(tool_output_text)
                        except Exception:
                            parsed = None

                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            chunks = item.get("chunk", [])
                            if isinstance(chunks, list):
                                doc_chunks.extend([c for c in chunks if isinstance(c, str)])
                            elif isinstance(chunks, str):
                                doc_chunks.append(chunks)
                elif isinstance(parsed, dict):
                    chunks = parsed.get("chunk", [])
                    if isinstance(chunks, list):
                        doc_chunks.extend([c for c in chunks if isinstance(c, str)])
                    elif isinstance(chunks, str):
                        doc_chunks.append(chunks)
                elif json_strings:
                    doc_chunks = json_strings

                # 如果有查询且启用了过滤器，则并行过滤相关 chunk
                query = tool_call.tool_kwargs.get("query")
                current_workflow_category = await ctx.store.get("current_workflow_category", default="research-general")
                if self.filter_llm and query:
                    relevant_chunks = await self._filter_chunks_parallel(doc_chunks, query, current_workflow_category)
                else:
                    relevant_chunks = doc_chunks

                # 更新上下文
                sources.append(relevant_chunks)
                current_reasoning.append(ObservationReasoningStep(observation=str(relevant_chunks)))
                # 将相关文档块缓存到上下文，供后续总结工具使用
                try:
                    await ctx.store.set("relevant_doc_chunks", relevant_chunks)
                except Exception:
                    pass
            else:
                # 通用处理：其他工具直接保存输出
                sources.append(tool_output_text)
                current_reasoning.append(
                    ObservationReasoningStep(observation=str(tool_output_text))
                )



        # 保存更新后的状态到上下文
        await ctx.store.set("sources", sources)
        await ctx.store.set("current_reasoning", current_reasoning)

        # 准备下一轮迭代
        return PrepEvent()
    # @step
    # async def conclude_doc_chunks(
    #         self, ctx: Context, ev: ConcludeEvent
    # ) -> PrepEvent:
    #     memory = await ctx.store.get("memory", default=None)
    #     query = await ctx.store.get("rag_input", default=None)
    #     current_reasoning = await ctx.store.get("current_reasoning", default=[])
    #     print("\n" + "=" * 20 + "筛选后doc内容" + "=" * 20 + "\n")
    #     print(ev.input)
    #
    #     conclusion = ""  # 每个chunk的总结，会被单独append到conclusion中。目前暂时不考虑跨chunk的关联关系
    #
    #     for chunk in ev.input:
    #         llm_input = self.conclusion_formatter.format(query=query, doc_chunks=chunk)
    #
    #         response_gen = await self.conclusion_llm.astream_chat(llm_input)
    #
    #         # 定义异步回调函数
    #         async def on_conclusion_thinking(content, metadata):
    #             await self.workflow_strategy.on_streaming_content(
    #                 content, "conclusion_thinking", {"step": "文档总结思考"}
    #             )
    #
    #         async def on_conclusion_content(content, metadata):
    #             await self.workflow_strategy.on_streaming_content(
    #                 content, "conclusion_output", {"step": "文档总结输出"}
    #             )
    #
    #         # 使用StreamingResponseParser解析流式响应
    #         response_content = await self.response_parser.parse_streaming_response(
    #             response_gen,
    #             on_thinking=on_conclusion_thinking,
    #             on_content=on_conclusion_content,
    #             thinking_metadata={"step": "文档总结思考", "phase": "conclusion"},
    #             content_metadata={"step": "文档总结输出", "phase": "conclusion"}
    #         )
    #
    #         # 提取最终内容
    #         final_content = self.response_parser.extract_final_content(response_content)
    #         conclusion += final_content + "\n\n"
    #         # print("\n" + "=" * 20 + "llm输出" + "=" * 20 + "\n")
    #         # print(response.message.content)
    #
    #     # put the llm conclusion to memory
    #     memory.put(
    #         ChatMessage(
    #             role="assistant", content=conclusion
    #         )
    #     )
    #
    #     current_reasoning.append(
    #         ObservationReasoningStep(observation=conclusion)
    #     )
    #
    #     await ctx.store.set("memory", memory)
    #     await ctx.store.set("current_reasoning", current_reasoning)
    #
    #     return PrepEvent()

    async def _on_parallel_thinking(self, content: str, metadata: dict):
        """并行工作流程思考过程回调"""
        category = metadata.get("category", "未知分类")
        await self.workflow_strategy.on_streaming_content(
            content, 
            "planning", 
            "thinking",
            {
                "step": f"[{category}] 并行思考过程",
                "category": category,
                "phase": "parallel_execution"
            }
        )

    async def _on_parallel_content(self, content: str, metadata: dict):
        """并行工作流程内容回调"""
        category = metadata.get("category", "未知分类")
        await self.workflow_strategy.on_streaming_content(
            content, 
            "planning", 
            "output",
            {
                "step": f"[{category}] 并行执行输出",
                "category": category,
                "phase": "parallel_execution"
            }
        )
