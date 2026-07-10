from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """Mirrors the fields the old `Agent` class tracked on `self` (agent.py)."""

    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    max_turns: int | None
    context_length: int
    context_window: int
    context_warning: int  # 0/1/2 budget-notice stage (folding is not ported in Phase 1)
    agent_name: str
    final_outcome: str | None
