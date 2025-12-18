import json
import asyncio
import os
from typing import Any, Dict, List

from fastmcp import Client
from custom_dashscope_llm import customDashscopeLLM
from config import config
from log_config import get_mcp_logger

# 初始化日志
logger = get_mcp_logger()

class MCPClient:
    def __init__(self):
        # 初始化LLM
        self.llm = customDashscopeLLM(
            model_code=config.DEFAULT_MODEL_NAME,
            api_key=config.DASHSCOPE_API_KEY,
            temperature=0.01,
            top_p=0.01,
            max_tokens=8192
        )
        
        # fast mcp streamable-http
        self.config = {
            "mcpServers": {
                "server": {
                    "url": os.getenv("MCP_HTTP_BASE_URL", f"http://{config.MCP_CLIENT_HOST}:{config.MCP_CLIENT_PORT}/mcp"),
                    "transport": "streamable-http"
                }
            }
        }
        # # nacos mcp server
        # self.config = {
        #     "mcpServers":
        #         {
        #             "nacos-mcp-router":
        #                 {
        #                     "command": "uvx",
        #                     "args":
        #                         [
        #                             "nacos-mcp-router@latest"
        #                         ],
        #                     "env":
        #                         {
        #                             "NACOS_ADDR": "http://192.168.226.144:8848",
        #                             "NACOS_USERNAME": "nacos",
        #                             "NACOS_PASSWORD": "nacos"
        #                         }
        #                 }
        #         }
        # }
        
        # 初始化FastMCP客户端
        self.client = Client(self.config)
        self.session = None
        self._tool_schema_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect_to_server(self):
        """连接到MCP服务器 - 增强连接复用"""
        async with self._lock:
            try:
                if self.session and self.client:
                    # 尝试验证现有连接是否活跃
                    try:
                        await self.client.list_tools()
                        logger.info("✅ 复用现有MCP连接")
                        return
                    except Exception:
                        logger.warning("⚠️ 现有连接已失效，正在重新连接...")
                        self.session = None

                await self.client.__aenter__()
                logger.info("✅ MCP服务器连接成功")
                self.session = self
            except Exception as e:
                logger.error(f"❌ 无法连接到MCP服务器: {e}")
                self.session = None
                raise

    async def ensure_connected(self):
        """确保已连接到服务器 - 增加状态检查"""
        if not self.session:
            await self.connect_to_server()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        tools = await self.client.list_tools()
        # 规范化为统一格式
        normalized = []
        for t in tools:
            name = getattr(t, "name", None) or t.get("name")
            description = getattr(t, "description", None) or t.get("description", "")
            input_schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or t.get("input_schema") or t.get("inputSchema")
            normalized.append({
                "name": name,
                "description": description,
                "input_schema": input_schema,
            })
        return normalized

    async def _get_allowed_keys(self, tool_name: str) -> List[str]:
        if tool_name in self._tool_schema_cache:
            schema = self._tool_schema_cache[tool_name]
        else:
            tools = await self.list_tools()
            schema = None
            for t in tools:
                if t.get("name") == tool_name:
                    schema = t.get("input_schema") or {}
                    break
            self._tool_schema_cache[tool_name] = schema or {}
        props = {}
        if isinstance(schema, dict):
            props = schema.get("properties", {}) or {}
        allowed = list(props.keys())
        extras = {"doc_chunks", "category"}
        return list(set(allowed).union(extras))

    async def _prune_arguments(self, tool_name: str, arguments: dict) -> dict:
        try:
            allowed = await self._get_allowed_keys(tool_name)
            return {k: v for k, v in (arguments or {}).items() if k in allowed}
        except Exception:
            return arguments or {}

    async def call_tool(self, tool_name: str, arguments: dict):
        """调用工具 - 增加重试机制"""
        max_retries = 3
        base_timeout = 60.0  # 默认300秒超时

        try:
            # 尝试从tools模块获取工具的超时时间配置
            from tools import get_tool_timeout
            base_timeout = get_tool_timeout(tool_name)
        except (ImportError, AttributeError):
            # 如果无法导入tools模块，使用默认超时时间
            pass

        for attempt in range(max_retries):
            try:
                # 每次重试前检查连接状态并确保连接有效
                if not self.session:
                    logger.warning("MCP会话丢失，尝试重新连接...")
                    await self.connect_to_server()
                
                pruned_args = await self._prune_arguments(tool_name, arguments)
                logger.info(f"正在调用工具 {tool_name}，第 {attempt + 1} 次尝试，超时时间: {base_timeout}秒...")
                result = await asyncio.wait_for(
                    self.client.call_tool(tool_name, pruned_args),
                    timeout=base_timeout
                )
                logger.info(f"工具 {tool_name} 调用成功")
                return {"status": "success", "result": str(result)}

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"工具调用超时，正在重试... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"工具调用失败，已重试 {max_retries} 次")
                    raise
            except Exception as e:
                logger.error(f"工具调用出错: {e}")
                msg = str(e)
                
                # 处理连接丢失错误，强制重连
                if "Client is not connected" in msg or "session" in msg.lower():
                    logger.warning("检测到连接错误，强制重置会话并重连...")
                    async with self._lock:
                        self.session = None
                        # 尝试清理旧连接
                        try:
                            await self.client.__aexit__(None, None, None)
                        except:
                            pass
                    # 下一次循环会触发 connect_to_server
                
                if "unexpected keyword argument" in msg or "ValidationError" in msg:
                    try:
                        arguments = await self._prune_arguments(tool_name, arguments)
                    except Exception:
                        pass
                        
                if attempt < max_retries - 1:
                    logger.info(f"正在重试... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    raise

    async def process_query(self, query: str) -> str:
        """处理用户查询"""
        system_prompt = """
        你是一个专业的助手，负责帮助用户选择合适的查询工具来获取相关信息。
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # 获取所有工具列表信息
        tools = await self.list_tools()
        # 生成function call的描述信息
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            }
        } for tool in tools]

        logger.debug(f"可用工具列表: {available_tools}")
        # 请求大模型，function call的描述信息通过tools参数传入
        response = self.llm.answer_gen(messages, streamed=False, tools=available_tools)
        logger.debug(f"LLM响应: {response}")
        
        # 处理返回的内容
        message = response["choices"][0]["message"]
        content = message.get("content", "")

        if response["choices"][0].get("finish_reason") == "tool_calls":
            # 如果需要使用工具，就解析工具
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                raise ValueError("No tool calls found in response")

            tool_call = tool_calls[0]
            tool_name = tool_call["function"]["name"]
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse tool arguments: {e}")

            # 执行工具
            logger.info(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")
            result = await self.call_tool(tool_name, tool_args)
            logger.info(f"工具调用结果: {result}")
            
            # 将大模型返回的调用工具数据和工具执行结果存入messages
            messages.append(message)
            # 处理工具调用结果，确保数据可序列化

            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
                "tool_call_id": tool_call["id"],
            })

            # 将结果再返回给大模型用于生成最终结果
            response = self.llm.answer_gen(messages, streamed=False)
            return response["choices"][0]["message"].get("content", "")

        return content

    async def chat_loop(self):
        """交互式聊天循环"""
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                logger.info("\n" + response)

            except Exception as e:
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """清理资源"""
        try:
            await self.client.__aexit__(None, None, None)
        except Exception:
            pass

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
