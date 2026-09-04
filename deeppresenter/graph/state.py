import operator
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
    # One entry per LLM call (engine.py's agent_node), in order — accumulated via list
    # concatenation (each node returns just its own new entry, not the whole list) so
    # every turn's timing/input/output survives to the end of the run for _save_history
    # to dump. See design_graph.py/research_graph.py's _save_llm_call_log.
    llm_call_log: Annotated[list[dict], operator.add]
