"""
流式响应解析工具类
用于解析自定义大模型返回的思考过程和完整回复
"""

import asyncio
from typing import AsyncGenerator, Callable, Any, Optional


class StreamingResponseParser:
    """流式响应解析器"""
    
    def __init__(self):
        # 分隔符定义
        self.thought_indicator = "\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n"
        self.answer_indicator = "\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n"
    
    async def parse_streaming_response(
        self,
        response_gen: AsyncGenerator[Any, None],
        on_thinking: Callable[[str, dict], None],
        on_content: Callable[[str, dict], None],
        thinking_metadata: Optional[dict] = None,
        content_metadata: Optional[dict] = None,
        category: Optional[str] = None
    ) -> str:
        """
        解析流式响应，区分思考过程和完整回复
        
        Args:
            response_gen: 流式响应生成器
            on_thinking: 思考过程回调函数
            on_content: 内容回调函数
            thinking_metadata: 思考过程的元数据
            content_metadata: 内容的元数据
            category: 工作流程分类标识
            
        Returns:
            str: 完整的响应内容
        """
        # 用于累积流式响应内容
        accumulated_content = ""
        current_mode = None  # 'thinking' 或 'content'
        full_response = ""
        
        # 默认元数据，包含分类信息
        if thinking_metadata is None:
            thinking_metadata = {"streaming": True, "step": "llm_thinking_stream"}
        if content_metadata is None:
            content_metadata = {"streaming": True, "step": "llm_content_stream"}
        
        # 添加分类标识到元数据
        if category:
            thinking_metadata["category"] = category
            content_metadata["category"] = category
        
        async for response in response_gen:
            # 处理流式响应 - 通过分隔符解析思考过程和内容
            if hasattr(response, 'delta') and response.delta:
                print(response.delta, end='', flush=True)
                accumulated_content += response.delta
                full_response += response.delta
                
                # 检查是否遇到思考过程分隔符
                if self.thought_indicator in accumulated_content:
                    if current_mode != 'thinking':
                        current_mode = 'thinking'
                        # 提取思考过程内容（分隔符之前的内容）
                        parts = accumulated_content.split(self.thought_indicator)
                        # 更新累积内容，移除已处理的部分
                        accumulated_content = parts[1] if len(parts) > 1 else ""
                
                # 检查是否遇到完整回复分隔符
                elif self.answer_indicator in accumulated_content:
                    if current_mode != 'content':
                        current_mode = 'content'
                        # 提取思考过程内容（分隔符之前的内容）
                        parts = accumulated_content.split(self.answer_indicator)
                        # 更新累积内容，移除已处理的部分
                        accumulated_content = parts[1] if len(parts) > 1 else ""
                
                # 如果当前在思考模式，发送思考内容
                elif current_mode == 'thinking':
                    await on_thinking(response.delta, thinking_metadata)
                
                # 如果当前在内容模式，发送回复内容
                elif current_mode == 'content':
                    await on_content(response.delta, content_metadata)
        
        return full_response
    
    def extract_final_content(self, full_response: str) -> str:
        """
        从完整响应中提取最终内容（完整回复部分）
        
        Args:
            full_response: 完整的响应内容
            
        Returns:
            str: 提取的最终内容
        """
        if self.answer_indicator in full_response:
            return full_response.split(self.answer_indicator)[-1]
        return full_response


    def extract_answer(self, full_response: str) -> str:
        """
        从完整响应中提取</think>标签之后的内容
        
        Args:
            full_response: 完整的响应内容
            
        Returns:
            str: 提取的最终内容
        """
        if "</think>" in full_response:
            return full_response.split("</think>")[-1].strip()
        return full_response

    def extract_thinking(self, full_response: str) -> str:
        """
        从完整响应中提取<think>与</think>标签之间的内容
        
        Args:
            full_response: 完整的响应内容
            
        Returns:
            str: 提取的思考过程内容
        """
        start_tag = "<think>"
        end_tag = "</think>"
        
        if start_tag in full_response and end_tag in full_response:
            start_index = full_response.find(start_tag) + len(start_tag)
            end_index = full_response.find(end_tag)
            return full_response[start_index:end_index].strip()
        return ""
    
    async def parse_parallel_streaming_response(
        self,
        category_responses: dict,
        on_thinking: Callable[[str, dict], None],
        on_content: Callable[[str, dict], None]
    ) -> dict:
        """
        解析并行工作流程的流式响应
        
        Args:
            category_responses: 分类响应字典 {category: response_gen}
            on_thinking: 思考过程回调函数
            on_content: 内容回调函数
            
        Returns:
            dict: 各分类的完整响应内容
        """
        results = {}
        
        # 为每个分类创建解析任务
        tasks = []
        for category, response_gen in category_responses.items():
            task = asyncio.create_task(
                self.parse_streaming_response(
                    response_gen,
                    on_thinking,
                    on_content,
                    thinking_metadata={"streaming": True, "step": "parallel_thinking", "category": category},
                    content_metadata={"streaming": True, "step": "parallel_content", "category": category},
                    category=category
                )
            )
            tasks.append((category, task))
        
        # 等待所有任务完成
        for category, task in tasks:
            try:
                results[category] = await task
            except Exception as e:
                results[category] = f"Error in category {category}: {str(e)}"
        
        return results
    
    def merge_parallel_results(self, category_results: dict) -> str:
        """
        合并并行工作流程的结果
        
        Args:
            category_results: 各分类的结果字典
            
        Returns:
            str: 合并后的结果
        """
        if not category_results:
            return "未获取到任何结果"
        
        merged_result = "## 并行工作流程执行结果\n\n"
        
        for category, result in category_results.items():
            merged_result += f"### {category} 分类结果\n"
            merged_result += f"{result}\n\n"
            merged_result += "---\n\n"
        
        return merged_result.strip()
