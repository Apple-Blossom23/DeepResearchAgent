from llama_index.core.agent.react import ReActChatFormatter
from llama_index.core.agent.react.formatter import BaseAgentChatFormatter
from custom_react_system_prompt import (CONCLUSION_PROMPT_TEMPLATE, 
                                        CUSTOM_REACT_CHAT_SYSTEM_HEADER, 
                                        CUSTOM_CONTEXT_REACT_CHAT_SYSTEM_HEADER, 
                                        FILTER_PROMPT_TEMPLATE, 
                                        NER_EXTRACT_TEMPLATE,
                                        ENTITY_RECOGNITION_TEMPLATE,
                                        INTENT_RECOGNITION_TEMPLATE,
                                        PLANNING_JUDGE_TEMPLATE,
                                        PLANNING_TEMPLATE,
                                        PLAN_MODIFY_TEMPLATE,
                                        PLAN_UPDATE_TEMPLATE,
                                        )
from typing import List, Optional, Sequence
from llama_index.core.tools import BaseTool
from llama_index.core.agent.react.types import (
    BaseReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.bridge.pydantic import Field

from llama_index.core.types import ChatMessage, MessageRole

class EntityRecognitionFormatter(BaseAgentChatFormatter):
    """Entity recognition chat formatter."""
    
    entity_recognition_template: str = Field(
        default=ENTITY_RECOGNITION_TEMPLATE,
        description="Entity recognition prompt template"
    )
    
    def format(self, user_input: str) -> List[ChatMessage]:
        """格式化实体识别请求"""
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=self.entity_recognition_template.format(input=user_input),
            )
        ]

class IntentRecognitionFormatter(BaseAgentChatFormatter):
    """Intent recognition chat formatter."""
    
    intent_recognition_template: str = Field(
        default=INTENT_RECOGNITION_TEMPLATE,
        description="Intent recognition prompt template"
    )
    
    def format(self, user_input: str) -> List[ChatMessage]:
        """格式化意图识别请求"""
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=self.intent_recognition_template.format(input=user_input),
            )
        ]

class NERExtractChatFormatter(BaseAgentChatFormatter):
    """NERExtract chat formatter."""
    @classmethod
    def format(
        cls, 
        query: str,
    ) -> List[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.TOOL,
                content=NER_EXTRACT_TEMPLATE.format(query=query),
            ),
        ]


class ConclusionChatFormatter(BaseAgentChatFormatter):
    """Conclusion chat formatter."""

    @classmethod
    def format(
        cls, 
        query: str,
        doc_chunks: str,
    ) -> List[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.TOOL,
                content=CONCLUSION_PROMPT_TEMPLATE.format(
                    query=query,
                    doc_chunks=doc_chunks,
                ),
            )
        ]
    
class FilterChatFormatter(BaseAgentChatFormatter):
    
    @classmethod
    def format(
        cls, 
        query: str,
        doc_chunks: str,
    ) -> List[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.TOOL,
                content=FILTER_PROMPT_TEMPLATE.format(
                    query=query,
                    doc_chunks=doc_chunks,
                ),
            )
        ]
    
class PlanningJudgeFormatter(BaseAgentChatFormatter):
    """Planning judge chat formatter."""
    @classmethod
    def format(
        cls, 
        query: str,
    ) -> List[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=PLANNING_JUDGE_TEMPLATE.format(
                    query=query,
                )
            )
        ]
    
class PlanningFormatter(BaseAgentChatFormatter):
    """Planning chat formatter."""
    @classmethod
    def format(
        cls,
        query: str,
        tool_desc: str,
        plan_examples: str,
        workflow: str = "",
        metadata_context: str = "",
    ):
        return [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=PLANNING_TEMPLATE.format(
                    query=query,
                    tool_desc=tool_desc,
                    plan_examples=plan_examples,
                    workflow=workflow,
                    metadata_context=metadata_context,
                )
            )
        ]
    
class PlanModifyFormatter(BaseAgentChatFormatter):
    """
    Formatter for the plan modify prompt.
    """
    @classmethod
    def format(
        cls,
        current_plan: str,
        modify_suggestion: str,
    ):
        return [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=PLAN_MODIFY_TEMPLATE.format(
                    current_plan=current_plan,
                    modify_suggestion=modify_suggestion,
                )
            )
        ]

class PlanUpdateFormatter(BaseAgentChatFormatter):
    """
    Formatter for the plan update prompt.
    """
    
    context: str = ""  # not needed w/ default
    observation_role: MessageRole = Field(
        default=MessageRole.TOOL,
        description=(
            "Message role of tool outputs. If the LLM you use supports function/tool "
            "calling, you may set it to `MessageRole.TOOL` to avoid the tool outputs "
            "being misinterpreted as new user messages."
        ),
    )

    def format(
        self,
        current_plan: str,
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ):
        
        reasoning_history = []
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ChatMessage(
                    role=self.observation_role,
                    content=reasoning_step.get_content(),
                )
            else:
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=reasoning_step.get_content(),
                )
            reasoning_history.append(message)

        # only filter out chatmessage where role is assistant, other roles like system or user are useless
        assistant_messages = []
        for message in chat_history:
            if message.role == MessageRole.ASSISTANT:
                assistant_messages.append(message)

        return [
            ChatMessage(
                role=MessageRole.USER,
                content=PLAN_UPDATE_TEMPLATE.format(
                    current_plan=current_plan,
                )
            ),
            *reasoning_history,
            *assistant_messages,
        ]

class customReActChatFormatter(ReActChatFormatter):

    system_header: str = CUSTOM_REACT_CHAT_SYSTEM_HEADER  # default
    context: str = ""  # not needed w/ default
    observation_role: MessageRole = Field(
        default=MessageRole.TOOL,
        description=(
            "Message role of tool outputs. If the LLM you use supports function/tool "
            "calling, you may set it to `MessageRole.TOOL` to avoid the tool outputs "
            "being misinterpreted as new user messages."
        ),
    )
    
    def get_react_tool_descriptions(self, tools: Sequence[BaseTool]) -> List[str]:
        """Tool."""
        tool_descs = []
        for tool in tools:
            tool_desc = (
                f"> Tool Name: {tool.metadata.name}\n"
                f"Tool Description: {tool.metadata.description}\n"
                f"Tool Args: {tool.metadata.fn_schema_str}\n"
            )
            tool_descs.append(tool_desc)
        return tool_descs

    def format(
            self,
            tools: Sequence[BaseTool],
            chat_history: List[ChatMessage],
            current_reasoning: Optional[List[BaseReasoningStep]] = None,
            current_plan: str = ""
        ) -> List[ChatMessage]:
            """Format chat history into list of ChatMessage."""
            current_reasoning = current_reasoning or []

            format_args = {
                "tool_desc": "\n".join(self.get_react_tool_descriptions(tools)),
                "tool_names": ", ".join([tool.metadata.get_name() for tool in tools]),
                "current_plan": current_plan,
            }
            if self.context:
                format_args["context"] = self.context

            fmt_sys_header = self.system_header.format(**format_args)

            # format reasoning history as alternating user and assistant messages
            # where the assistant messages are thoughts and actions and the tool
            # messages are observations
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

    @classmethod
    def from_custom(cls,
        system_header: Optional[str] = None,
        context: Optional[str] = None,
        observation_role: MessageRole = MessageRole.USER,) -> 'customReActChatFormatter':

        if not system_header:
            system_header = (
                CUSTOM_REACT_CHAT_SYSTEM_HEADER
                if not context
                else CUSTOM_CONTEXT_REACT_CHAT_SYSTEM_HEADER
            )

        return customReActChatFormatter(
            system_header=system_header,
            context=context or "",
            observation_role=observation_role,
        )