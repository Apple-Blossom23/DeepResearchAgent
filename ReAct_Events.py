from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event


class PrepEvent(Event):
    pass


class InputEvent(Event):
    input: list[ChatMessage]


class ConciergeEvent(Event):
    input: str # the original query of the user
    additional_input: list[str] # addtional information provided by the user in later progress

class PlanningEvent(Event):
    input: str
    additional_input: list[str] # the lastest user feedback about the current plan

class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]

class FilterEvent(Event):
    doc_chunks: list[str]

class FilterStartEvent(Event):
    """过滤开始事件"""
    total_chunks: int  # 总文档块数
    query: str  # 查询关键词
    category: str  # 工作流分类

class FilterProgressEvent(Event):
    """过滤进度事件"""
    batch_id: int  # 批次ID
    chunk_index: int  # 当前块索引
    chunk_content: str  # 文档块内容
    is_relevant: bool  # 是否相关
    thinking_process: str  # 思考过程
    category: str  # 工作流分类

class FilterCompleteEvent(Event):
    """过滤完成事件"""
    total_chunks: int  # 总文档块数
    relevant_count: int  # 相关文档块数
    filtered_count: int  # 过滤掉的文档块数
    relevant_chunks: list[str]  # 相关文档块列表
    category: str  # 工作流分类

class ConcludeEvent(Event):
    input: list[str]

class IntentRecognitionEvent(Event):
    input: str  # 用户输入
    
class EntityRecognitionEvent(Event):
    input: str  # 用户输入
    additional_input: list[str] = []  # 额外的用户输入信息
    
class EntityAnalysisEvent(Event):
    input: str  # 用户输入
    recognized_entities: list[dict]  # 识别出的实体信息
    selected_workflow_template: str  # 选择的工作流模板

class FunctionOutputEvent(Event):
    output: ToolOutput