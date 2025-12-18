'''
This file defines a custom dashscope LLM which overrides the default completion method provided by llama_index.core.llms.CustomLLM.
So that we can use any LLM provided by dashscope as the query engine in a RAG pipeline.
Note: This file requires that you own a dashscope api key.
The default dashscope model used in qwen2-72B-Instruct.
'''

from typing import Any, Generator
import re

import llama_index.core.llms as llama_index_llms 
from llama_index.core.llms.callbacks import llm_completion_callback
from dashscope_agent import DashscopeAgent
from config import config

class customDashscopeLLM(llama_index_llms.CustomLLM):
    """
    Custom dashscope LLM class utilizing transformers to create a customed LLM that stored locally as llama_index query engine.
    """

    # declare attributes first to bypass the pydantic validation
    
    context_window: int = 8192 # default setting
    max_tokens: int = 2048 # default setting
    model_code: str = "rsv-wn9z1o54" # default setting
    api_key: Any = None # default setting
    tokernizer: Any = None
    temperature: float = 0.8 # default setting
    top_p: float = 0.8  # default setting


    def __init__(self,
                 context_window: int=8192,
                 max_tokens: int=8192,
                 model_code: str=config.DEFAULT_MODEL_NAME,
                 api_key=config.DASHSCOPE_API_KEY,
                 temperature: float=0.01,
                 top_p: float=0.01):
        """
        Initialize the customDashscopeLLM class.
        Args:
            context_window: The maximum context window size. Defaults to 3900.
            max_tokens: The maximum number of output tokens. Defaults to 512.
            model_name: The name of the model to use. Defaults to "Qwen3-32B".
        """

        super().__init__()
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.model_code = model_code
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p

    def input_gen(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate input for the model.
        Args:
            prompt: The prompt to generate input for.
            **kwargs: Additional keyword arguments.
        Returns:
            The generated input.
        """

        messages = [
            {"role": "system", "content": "You are Qwen3, a helpful assistant trained by Alibaba Group."},
            {"role": "user", "content": prompt},
        ]

        return messages

    def _extract_thinking_and_content(self, content: str) -> tuple[str, str]:
        """
        从content中提取思考过程和实际内容，兼容两种格式：
        1. reasoning_content字段 + content字段
        2. content字段中包含<think></think>标签
        
        参数:
        content: 原始内容字符串
        
        返回:
        tuple: (思考过程, 实际内容)
        """
        # 检查是否包含<think></think>标签
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, content, re.DOTALL)
        
        if think_match:
            # 提取思考过程
            thinking = think_match.group(1).strip()
            # 移除思考过程，保留实际内容
            actual_content = re.sub(think_pattern, '', content).strip()
            return thinking, actual_content
        
        # 如果没有<think></think>标签，返回空思考过程和原内容
        return "", content

    def answer_gen(self, prompt: str, streamed: str = False, tools: list = None, **kwargs: Any) -> dict | Generator:
        '''
        Generate answer. If streamed is True, return a generator. Otherwise return a string.
        '''

        agent = DashscopeAgent(model_code=self.model_code, api_key=self.api_key, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)

        # when answer generation is not in streaming mode
        if not streamed:
            response = agent.get_response(prompt, tools)
            # 处理非流式响应中的<think></think>标签
            if isinstance(response, dict) and "choices" in response:
                message = response["choices"][0]['message']
                reasoning_content = message.get('reasoning_content', "")
                content = message.get('content', "")
                
                # 如果没有reasoning_content但有content，尝试从content中提取思考过程
                if not reasoning_content and content:
                    reasoning_content, content = self._extract_thinking_and_content(content)
                    # 更新原始响应对象
                    message['reasoning_content'] = reasoning_content
                    message['content'] = content
            
            return response

        else:
            # 流式响应需要特殊处理<think></think>标签
            return self._process_stream_with_thinking(agent.get_stream_response(prompt, tools))

    def _process_stream_with_thinking(self, streamer: Generator) -> Generator:
        """
        处理流式响应中的<think></think>标签
        
        参数:
        streamer: 原始流式响应生成器
        
        返回:
        Generator: 处理后的流式响应生成器
        """
        buffer = ""
        in_thinking_mode = False
        
        for chunk in streamer:
            try:
                # 初始化delta变量以避免UnboundLocalError
                delta = {}
                
                # 检查是否有content字段
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                
                # 如果有content字段且不为null
                # if "content" in delta and delta["content"] is not None and delta["content"].strip() != "":
                if "content" in delta and delta["content"] is not None:
                    current_content = delta["content"]
                    buffer += current_content
                    
                    # 不在思考模式中
                    if not in_thinking_mode:
                        # 检查是否遇到<think>标签
                        if "<think>" in buffer:
                            think_start = buffer.find("<think>")
                            
                            # 如果<think>前面有内容，先作为普通content返回
                            if think_start > 0:
                                content_before_think = buffer[:think_start]
                                new_chunk = chunk.copy()
                                new_chunk["choices"][0]["delta"] = delta.copy()
                                new_chunk["choices"][0]["delta"]["content"] = content_before_think
                                yield new_chunk
                            
                            # 进入思考模式，移除已处理的内容
                            in_thinking_mode = True
                            buffer = buffer[think_start + 7:]  # 移除<think>
                        else:
                            # 没有<think>标签，直接返回原始chunk
                            yield chunk
                            buffer = ""  # 清空缓冲区
                    
                    # 在思考模式中
                    else:
                        # 检查是否遇到</think>标签
                        if "</think>" in buffer:
                            think_end = buffer.find("</think>")
                            
                            # 将</think>前的内容作为思考内容返回
                            thinking_part = buffer[:think_end]
                            if thinking_part:
                                new_chunk = chunk.copy()
                                new_chunk["choices"][0]["delta"] = delta.copy()
                                new_chunk["choices"][0]["delta"]["reasoning_content"] = thinking_part
                                new_chunk["choices"][0]["delta"]["content"] = None
                                yield new_chunk
                            
                            # 退出思考模式
                            in_thinking_mode = False
                            buffer = buffer[think_end + 8:]  # 移除</think>
                            
                            # 如果</think>后面还有内容，作为普通content返回
                            if buffer:
                                new_chunk = chunk.copy()
                                new_chunk["choices"][0]["delta"] = delta.copy()
                                new_chunk["choices"][0]["delta"]["content"] = buffer
                                new_chunk["choices"][0]["delta"]["reasoning_content"] = None
                                yield new_chunk
                                buffer = ""
                        else:
                            # 在思考模式中但还没遇到</think>，将当前内容作为思考内容返回
                            new_chunk = chunk.copy()
                            new_chunk["choices"][0]["delta"] = delta.copy()
                            new_chunk["choices"][0]["delta"]["reasoning_content"] = current_content
                            new_chunk["choices"][0]["delta"]["content"] = None
                            yield new_chunk
                            buffer = ""  # 清空缓冲区，因为已经返回了
                
                # 如果content为null但有reasoning_content字段，直接传递
                if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                    yield chunk
                
                # 如果content为null且reasoning_content也为null，跳过这个chunk（避免重复处理）
                if delta.get("content") is None and delta.get("reasoning_content") is None:
                    # 只传递非content/reasoning_content的字段变化（如finish_reason等）
                    if any(key not in ["content", "reasoning_content"] for key in delta.keys() if delta[key] is not None):
                        yield chunk
                    # 否则跳过这个chunk
                
                # 其他情况直接传递
                if "choices" in chunk and len(chunk["choices"]) > 0 and not ("content" in delta or "reasoning_content" in delta):
                    yield chunk
            except (KeyError, IndexError, TypeError) as e:
                # 处理可能的键错误、索引错误或类型错误
                # 记录错误但继续处理下一个chunk
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"处理流式响应chunk时出现异常: {e}, chunk: {chunk}")
                # 如果chunk结构异常，尝试直接传递
                if chunk:
                    yield chunk
                continue

    @property
    def metadata(self) -> llama_index_llms.LLMMetadata:
        """
        Get the metadata for the model.
        Returns:
            The metadata for the model.
        """
        return llama_index_llms.LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_code,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> llama_index_llms.CompletionResponse:
        """Get LLM completion response."""
        outputs = self.answer_gen(prompt, streamed=False)
        
        # 获取消息内容（思考过程已经在answer_gen中处理）
        message = outputs["choices"][0]['message']
        reasoning_content = message.get('reasoning_content', "")
        content = message.get('content', "")

        thought_indicator = "\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n"
        answer_indicator = "\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n"
        final_outputs = thought_indicator + reasoning_content + answer_indicator + content

        return llama_index_llms.CompletionResponse(text=final_outputs)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> llama_index_llms.CompletionResponseGen:
        """Get LLM completion response."""
        streamer = self.answer_gen(prompt, streamed=True)

        response = ""
        is_thinking = False # 是否进入思考阶段
        is_answering = False  # 是否进入回复阶段
        thought_indicator = "\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n"
        answer_indicator = "\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n"
        tokens = ""
        
        if not is_thinking:
            response += thought_indicator
            is_thinking = True
            yield llama_index_llms.CompletionResponse(text=response, delta=thought_indicator)

        for chunk in streamer:
            try:
                tokens = ""  # 重置tokens变量
                
                # 检查chunk结构是否正确
                if not isinstance(chunk, dict) or "choices" not in chunk:
                    continue
                    
                if not chunk["choices"] or len(chunk["choices"]) == 0:
                    continue
                    
                choice = chunk["choices"][0]
                if "delta" not in choice:
                    continue
                    
                delta = choice["delta"]
                
                # 优先检查reasoning_content字段
                if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                    tokens = delta["reasoning_content"]

                # 检查content字段
                # elif "content" in delta and delta["content"] is not None and delta["content"].strip() != "":
                elif "content" in delta and delta["content"] is not None:
                    if not is_answering:
                        response += answer_indicator
                        is_answering = True
                        yield llama_index_llms.CompletionResponse(text=response, delta=answer_indicator)
                    tokens = delta["content"]

                # 检查是否需要结束
                if choice.get("finish_reason") == "stop":
                    break
                
                # 只有当tokens不为空时才返回
                if tokens and len(tokens) > 0:
                    response += tokens
                    yield llama_index_llms.CompletionResponse(text=response, delta=tokens)
                    
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                # 处理可能的各种异常
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"处理流式响应chunk时出现异常: {e}, chunk类型: {type(chunk)}, chunk内容: {chunk}")
                # 跳过有问题的chunk，继续处理下一个
                continue

if __name__ == "__main__":
    llm = customDashscopeLLM(max_tokens=8192, model_code=config.DEFAULT_MODEL_NAME, api_key=config.DASHSCOPE_API_KEY)
    response = llm.stream_complete("curl命令在windows 下如何使用")
    for chunk in response:
        print(chunk.delta)