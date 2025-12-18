from llama_index.core.agent.react.types import BaseReasoningStep

class MilestoneReasoningStep(BaseReasoningStep):
    """Milestone Reasoning step."""

    thought: str
    milestone: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Thought: {self.thought}\nMilestone: {self.milestone}\n"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False