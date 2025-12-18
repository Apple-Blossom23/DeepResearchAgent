from typing import Any, List
from llama_index.core.embeddings import BaseEmbedding
import requests

class OnlineQwen3Embedding(BaseEmbedding):
    online_url:str = None
    def __init__(
        self,
        online_url = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.online_url = online_url

    def _get_query_embedding(self, query: str) -> List[float]:
        messages = {
            "texts": [query]
        }
        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Authorization': 'Bearer 1pbwBnFq82SYLxKd39TbHHJEONMF455m',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'User-Agent': 'PostmanRuntime-ApipostRuntime/1.1.0'
        }
        embeddings = requests.post(url=self.online_url, headers=headers, json=messages)
        # 根据新API的响应格式调整返回值解析
        response_data = embeddings.json()
        if 'success' in response_data and response_data['success'] and 'data' in response_data:
            # 新API格式: {"success": true, "data": [[embedding_vector]]}
            return response_data['data'][0]
        elif 'data' in response_data and len(response_data['data']) > 0:
            # 标准格式: {"data": [{"embedding": [vector]}]}
            if isinstance(response_data['data'][0], dict) and 'embedding' in response_data['data'][0]:
                return response_data['data'][0]['embedding']
            else:
                return response_data['data'][0]
        elif isinstance(response_data, list) and len(response_data) > 0:
            return response_data[0]
        else:
            raise ValueError(f"Unexpected response format: {response_data}")

    def _get_text_embedding(self, text: str) -> List[float]:
        messages = {
            "texts": [text]
        }
        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Authorization': 'Bearer 1pbwBnFq82SYLxKd39TbHHJEONMF455m',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'User-Agent': 'PostmanRuntime-ApipostRuntime/1.1.0'
        }
        embeddings = requests.post(url=self.online_url, headers=headers, json=messages)
        # 根据新API的响应格式调整返回值解析
        response_data = embeddings.json()
        if 'success' in response_data and response_data['success'] and 'data' in response_data:
            # 新API格式: {"success": true, "data": [[embedding_vector]]}
            return response_data['data'][0]
        elif 'data' in response_data and len(response_data['data']) > 0:
            # 标准格式: {"data": [{"embedding": [vector]}]}
            if isinstance(response_data['data'][0], dict) and 'embedding' in response_data['data'][0]:
                return response_data['data'][0]['embedding']
            else:
                return response_data['data'][0]
        elif isinstance(response_data, list) and len(response_data) > 0:
            return response_data[0]
        else:
            raise ValueError(f"Unexpected response format: {response_data}")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        messages = {
            "texts": texts
        }
        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Authorization': 'Bearer 1pbwBnFq82SYLxKd39TbHHJEONMF455m',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'User-Agent': 'PostmanRuntime-ApipostRuntime/1.1.0'
        }
        embeddings = requests.post(url=self.online_url, headers=headers, json=messages)
        # 根据新API的响应格式调整返回值解析
        response_data = embeddings.json()
        if 'success' in response_data and response_data['success'] and 'data' in response_data:
            # 新API格式: {"success": true, "data": [[embedding_vector1], [embedding_vector2], ...]}
            return response_data['data']
        elif 'data' in response_data:
            # 标准格式: {"data": [{"embedding": [vector1]}, {"embedding": [vector2]}, ...]}
            if len(response_data['data']) > 0 and isinstance(response_data['data'][0], dict) and 'embedding' in response_data['data'][0]:
                return [element['embedding'] for element in response_data['data']]
            else:
                return response_data['data']
        elif isinstance(response_data, list):
            return response_data
        else:
            raise ValueError(f"Unexpected response format: {response_data}")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)