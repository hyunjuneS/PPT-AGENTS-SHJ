"""Generic LangGraph StateGraph builder mirroring the old ReAct loop in
deeppresenter/agents/agent.py (action() -> execute() -> loop until `finalize`
echoes its own outcome). Kept independent of the old engine; Research/Planner
keep running on deeppresenter/agents/ untouched during this pilot.
"""

import asyncio
import os
import time
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from deeppresenter.graph.state import GraphState
from deeppresenter.utils.constants import HALF_BUDGET_NOTICE_MSG, RETRY_TIMES, URGENT_BUDGET_NOTICE_MSG
from deeppresenter.utils.log import (
    logging_openai_exceptions,
    show_agent_done,
    show_agent_turn,
    show_tool_call,
    show_tool_result,
)


def _has_image(message: BaseMessage) -> bool:
    return isinstance(message.content, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in message.content
    )


def cap_images(messages: list[BaseMessage], max_images: int) -> list[BaseMessage]:
    """Port of Agent._cap_images (agent.py:182-203). Only applied to the view
    fed to the model on this call — the full, uncapped message list stays in
    graph state, exactly like the old engine only capped the argument passed
    into `self.llm.run(...)` while `self.chat_history` kept every image."""
    image_indices = [i for i, m in enumerate(messages) if _has_image(m)]
    to_strip = set(image_indices[:-max_images]) if len(image_indices) > max_images else set()
    if not to_strip:
        return messages

    result: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        if i in to_strip:
            text_blocks = [b for b in msg.content if isinstance(b, dict) and b.get("type") == "text"]
            text_blocks.append({"type": "text", "text": "(이전 슬라이드 이미지 — 컨텍스트 한도로 제거됨)"})
            result.append(msg.model_copy(update={"content": text_blocks}))
        else:
            result.append(msg)
    return result


def _append_notice(message: BaseMessage, notice_text: str) -> BaseMessage:
    """Port of the 'turns running out' injection (agent.py:213-225), which
    mutates the actual last history message (not just the model-facing view)."""
    if isinstance(message.content, list):
        new_content = [*message.content, {"type": "text", "text": notice_text}]
    else:
        new_content = f"{message.content}\n\n{notice_text}" if message.content else notice_text
    return message.model_copy(update={"content": new_content})


def _prepend_notice(message: BaseMessage, notice_block: dict) -> BaseMessage:
    """Port of `observations[0].content.insert(0, NOTICE)` (agent.py:318/322)."""
    if isinstance(message.content, list):
        new_content = [notice_block, *message.content]
    else:
        new_content = [notice_block, {"type": "text", "text": message.content or ""}]
    return message.model_copy(update={"content": new_content})


async def _invoke_with_retry(model_with_tools, messages, model_name: str, retry_times: int = RETRY_TIMES):
    """Port of the old engine's LLM.run() retry loop
    (deeppresenter/utils/config.py:85-113): retry on any exception, AND on a
    genuinely empty response (no content, no tool_calls) — the same failure
    mode this app has hit before ("Empty response from model" / "All 3
    retries failed") — with the same exponential backoff (2**attempt, capped
    at 30s) and per-attempt warning log the old engine had. Without this, an
    empty response would silently become a no-op turn that loops straight
    back to another LLM call with zero delay and no logging, instead of
    retrying with backoff and then failing loudly."""
    errors: list[str] = []
    for attempt in range(retry_times):
        try:
            response = await model_with_tools.ainvoke(messages)
            assert response.content or response.tool_calls, "Empty response from model"
            return response
        except Exception as e:
            errors.append(str(e))
            logging_openai_exceptions(model_name, e)
            if attempt < retry_times - 1:
                await asyncio.sleep(min(2 ** attempt, 30))
    raise ValueError(f"All {retry_times} retries failed:\n" + "\n".join(errors))


def is_finalize_confirmed(
    ai_message: AIMessage, tool_messages: list[ToolMessage]
) -> tuple[bool, str | None]:
    """Port of Agent.execute()'s termination check (agent.py:307-312): a
    `finalize` call only really ends the loop if the tool's own return value
    exactly echoes the `outcome` arg it was called with. task.py's finalize()
    returns a *different* string on validation failure, and that difference
    alone is what tells the old (and new) engine to keep looping — reproduced
    verbatim here rather than inventing a new completion flag."""
    for tc in ai_message.tool_calls or []:
        if tc["name"] == "finalize":
            outcome = tc["args"].get("outcome", "")
            for tm in tool_messages:
                if tm.tool_call_id == tc["id"] and tm.content == outcome:
                    return True, outcome
    return False, None


def build_graph(
    chat_model,
    tools: Sequence[StructuredTool],
    context_window: int,
) -> CompiledStateGraph:
    # handle_tool_errors=True: catch any exception raised inside a tool (not just
    # LangGraph's own ToolInvocationError) and turn it into ToolMessage content,
    # matching AgentEnv.tool_execute()'s blanket try/except (env.py:92-98) — the
    # old engine never lets a tool exception crash the outer loop.
    tool_node = ToolNode(list(tools), handle_tool_errors=True)
    model_with_tools = chat_model.bind_tools(list(tools))
    model_name = getattr(chat_model, "model_name", None) or getattr(chat_model, "model", "unknown")
    base_url = getattr(chat_model, "openai_api_base", None) or getattr(chat_model, "base_url", None)
    model_name = f"{model_name} @ {base_url}" if base_url else model_name
    start_time = time.time()

    async def agent_node(state: GraphState) -> dict:
        turn_count = state["turn_count"] + 1
        max_turns = state.get("max_turns")
        agent_name = state["agent_name"]

        if max_turns is not None and turn_count > max_turns:
            raise RuntimeError(f"{agent_name} exceeded max turns: {turn_count - 1}/{max_turns}")

        max_images = int(os.environ.get("VLM_MAX_IMAGES", "2"))
        messages = cap_images(state["messages"], max_images)

        updates: list[BaseMessage] = []
        if max_turns is not None and max_turns - turn_count < 2 and messages:
            warned_last = _append_notice(
                messages[-1],
                f"You have only {max_turns - turn_count} turn(s) left. "
                "Finish the remaining work and call `finalize` immediately.",
            )
            messages = [*messages[:-1], warned_last]
            updates.append(warned_last)

        show_agent_turn(agent_name, turn_count, max_turns)
        response = await _invoke_with_retry(model_with_tools, messages, model_name)

        for tc in response.tool_calls or []:
            show_tool_call(tc["name"], tc["args"])

        context_length = state["context_length"]
        usage = getattr(response, "usage_metadata", None)
        if usage:
            context_length = usage.get("total_tokens", context_length)

        return {
            "messages": [*updates, response],
            "turn_count": turn_count,
            "context_length": context_length,
        }

    def route_after_agent(state: GraphState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        # Mirrors the old engine: Agent.execute([]) returns [] immediately and
        # loop() just calls action() again — a turn with no tool calls does
        # not end the loop, it just tries another turn.
        return "agent"

    async def postprocess_node(state: GraphState) -> dict:
        messages = state["messages"]
        last_ai = next(m for m in reversed(messages) if isinstance(m, AIMessage))
        tool_call_ids = {tc["id"] for tc in (last_ai.tool_calls or [])}
        new_tool_msgs = [
            m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id in tool_call_ids
        ]

        for tm in new_tool_msgs:
            show_tool_result(str(tm.content))

        confirmed, outcome = is_finalize_confirmed(last_ai, new_tool_msgs)
        if confirmed:
            show_agent_done(state["agent_name"], state["turn_count"], time.time() - start_time)
            return {"final_outcome": outcome}

        context_length = state["context_length"]
        context_window_val = state["context_window"]
        context_warning = state["context_warning"]
        updates: list[BaseMessage] = []

        if new_tool_msgs:
            first = new_tool_msgs[0]
            if context_warning == 0 and context_length > context_window_val * 0.5:
                context_warning = 1
                updates.append(_prepend_notice(first, HALF_BUDGET_NOTICE_MSG))
            elif context_warning == 1 and context_length > context_window_val * 0.8:
                context_warning = 2
                updates.append(_prepend_notice(first, URGENT_BUDGET_NOTICE_MSG))

        # Context folding is not ported in Phase 1 (dormant in the old engine
        # too — context_folding defaults to False) — same hard stop as before.
        if context_length > context_window_val and context_warning != -1:
            raise RuntimeError(
                f"{state['agent_name']} exceeded context window: {context_length}/{context_window_val}"
            )

        return {"messages": updates, "context_warning": context_warning}

    def route_after_postprocess(state: GraphState) -> str:
        return END if state.get("final_outcome") is not None else "agent"

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("postprocess", postprocess_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "agent": "agent"})
    graph.add_edge("tools", "postprocess")
    graph.add_conditional_edges("postprocess", route_after_postprocess, {"agent": "agent", END: END})

    return graph.compile()
