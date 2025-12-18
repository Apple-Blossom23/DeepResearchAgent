import requests
import json
from typing import Generator, Dict, List, Optional, Union

from config import config


class Agent:
    """
    Agent基本类，作为对话代理的抽象基类。
    """
    def __init__(self):
        """
        初始化代理的基本设置。
        """
        pass

    def get_response(self, messages):
        """
        获取代理对给定提示的响应。

        参数:
        messages (list): 用于触发代理响应的用户消息。

        返回:
        object: 代理生成的响应。
        """
        pass


class DashscopeAgent(Agent):
    """
    使用Dashscope API的对话代理类。
    注：此版本为适配op2.0.1提供openai compatible 接口的版本

    参数:
    model_code (str): 要使用的Dashscope模型名称。
    temperature (float): 生成文本的随机性程度。较高的温度会导致更创造性但可能不准确的输出。
    top_p (float): 用于采样的概率阈值。较高的top_p值会导致更多样化的输出。
    max_tokens (int): 生成文本的最大token数量。
    """
    def __init__(self, model_code: str, api_key: Optional[str] = None,
                 temperature: float = 0.01, top_p: float = 0.01, max_tokens: int = 8192):
        """
        初始化Dashscope代理，包括获取API密钥和设置模型参数。
        """
        super().__init__()
        self.base_url = config.DASHSCOPE_BASE_URL
        self.model_code = model_code
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_request_data(self, messages: Union[str, List[Dict]], 
                           stream: bool = False, tools: Optional[List] = None) -> Dict:
        """构建请求数据"""
        # 处理消息格式
        if isinstance(messages, str):
            formatted_messages = [{"role": "user", "content": messages}]
        else:
            formatted_messages = messages

        data = {
            "model": self.model_code,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": stream
        }

        # 只有当tools存在且不为空时才添加到请求数据中
        if tools:
            data["tools"] = tools

        return data

    def _handle_stream_response(self, response) -> Generator[Dict, None, None]:
        """处理流式响应"""
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if line.startswith('data: '):
                _line = line[len('data: '):]
                if _line == "[DONE]":
                    break
                try:
                    yield json.loads(_line)
                except json.JSONDecodeError:
                    continue  # 跳过无法解析的行
            else:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # 跳过无法解析的行

    def _make_request(self, data: Dict, stream: bool = False):
        """发送API请求"""
        try:
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=data,
                stream=stream
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {str(e)}")

    def get_response(self, messages: Union[str, List[Dict]], tools: Optional[List] = None) -> Dict:
        """
        与Dashscope API通信，根据给定的消息获取模型的响应。非流式返回内容。

        参数:
        messages: 要发送给模型的消息，可以是字符串或消息列表。
        tools: 可选，工具列表，如果不提供或为空列表则不包含在请求中。

        返回:
        dict: 从Dashscope API接收的响应。
        """
        data = self._build_request_data(messages, stream=False, tools=tools)
        response = self._make_request(data, stream=False)
        
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise Exception(f"响应解析失败: {str(e)}")

    def get_stream_response(self, messages: Union[str, List[Dict]], tools: Optional[List] = None) -> Generator[Dict, None, None]:
        """
        与Dashscope API通信，根据给定的消息获取模型的响应。流式返回内容。

        参数:
        messages: 要发送给模型的消息，可以是字符串或消息列表。
        tools: 可选，工具列表，如果不提供或为空列表则不包含在请求中。

        返回:
        Generator: 从Dashscope API接收的流式响应生成器。
        """
        data = self._build_request_data(messages, stream=True, tools=tools)
        response = self._make_request(data, stream=True)
        return self._handle_stream_response(response)


if __name__ == "__main__":
    agent = DashscopeAgent(model_code=config.DEFAULT_MODEL_NAME, api_key=config.DASHSCOPE_API_KEY)
    messages = "curl命令在windows 下如何使用"
    responses = agent.get_stream_response(messages=messages)
    for chunk in responses:
        print(json.dumps(chunk, ensure_ascii=False, indent=2))
