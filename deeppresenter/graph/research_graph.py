"""Research-agent entrypoint for the new LangGraph-based engine (Phase 2).

Mirrors deeppresenter/agents/research.py's Research.loop() + the relevant
slices of Agent.__init__/action()/execute(), the same way design_graph.py
mirrors Design.loop() — as a single awaitable function instead of a stateful
class + async generator, since main-ui.py never streams intermediate items to
the HTTP client anyway.

Helper functions are intentionally duplicated from design_graph.py rather than
shared, so design_graph.py (already verified working) stays untouched.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deeppresenter.agents.env import AgentEnv
from deeppresenter.graph.engine import build_graph
from deeppresenter.graph.llm_adapter import to_chat_openai
from deeppresenter.graph.tools import build_tools_for_role
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.constants import CONTEXT_MODE_PROMPT, OFFLINE_PROMPT, PACKAGE_DIR
from deeppresenter.utils.log import show_agent_start
from deeppresenter.utils.typings import InputRequest, RoleConfig

_RECURSION_LIMIT = 500  # old engine had no hard turn cap for Research; generous headroom here

_LANG_INSTRUCTION = {
    "ko": "IMPORTANT: Write all output text content (slides, manuscripts) in Korean (한국어로 작성하세요).",
    "en": "IMPORTANT: Write all output text content (slides, manuscripts) in English.",
}


@dataclass
class ResearchGraphResult:
    manuscript_path: str
    messages_log: list[dict]
    turn_count: int
    cost: dict


def _load_role_config(config_file: str | Path | None) -> RoleConfig:
    role_config_file = Path(config_file) if config_file else PACKAGE_DIR / "roles" / "Research.yaml"
    if not role_config_file.exists():
        raise FileNotFoundError(f"Role config not found: {role_config_file}")
    with open(role_config_file, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return RoleConfig(**config_data)


def _build_system_prompt(
    role_config: RoleConfig,
    language: str,
    config: DeepPresenterConfig,
) -> str:
    system = role_config.system
    system += f"\n\n{_LANG_INSTRUCTION.get(language, _LANG_INSTRUCTION['en'])}"

    if config.offline_mode:
        system += OFFLINE_PROMPT
    if config.context_folding:
        system += CONTEXT_MODE_PROMPT

    return system


def _message_preview(content) -> str:
    text = content if isinstance(content, str) else str(content)
    return text[:200]


def _save_history(
    workspace: Path,
    agent_name: str,
    messages: list[BaseMessage],
    model: str,
    context_length: int,
    cost: dict,
    tool_names: list[str],
) -> None:
    """Port of Agent.save_history() (agent.py:401-429) so the same
    .history/{agent}-history.json / -config.json artifacts keep appearing —
    used as a baseline/parity check against the old engine."""
    hist_dir = workspace / ".history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    def _dump_message(m: BaseMessage) -> dict:
        return {
            "type": m.__class__.__name__,
            "content": m.content,
            "tool_calls": getattr(m, "tool_calls", None),
            "tool_call_id": getattr(m, "tool_call_id", None),
        }

    with open(hist_dir / f"{agent_name}-history.json", "w", encoding="utf-8") as f:
        json.dump([_dump_message(m) for m in messages], f, ensure_ascii=False, indent=2)

    with open(hist_dir / f"{agent_name}-config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": agent_name,
                "model": model,
                "context_length": context_length,
                "cost": cost,
                "tools": tool_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _save_llm_call_log(workspace: Path, agent_name: str, calls: list[dict]) -> None:
    """Writes .history/{agent}-llm-calls.json: one entry per LLM call this run made
    (engine.py's agent_node), each with its turn number, exact input messages, output
    message, and how long that single request took — plus the summed total across every
    call, so total LLM wall-clock time for the run is visible without adding it up by hand."""
    hist_dir = workspace / ".history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    total_elapsed = sum(c["elapsed_seconds"] for c in calls)
    with open(hist_dir / f"{agent_name}-llm-calls.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent": agent_name,
                "call_count": len(calls),
                "total_elapsed_seconds": round(total_elapsed, 3),
                "calls": calls,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


async def run_research_graph(
    config: DeepPresenterConfig,
    workspace: Path,
    req: InputRequest,
    outline_path: str | Path | None = None,
    config_file: str | Path | None = None,
    language: str = "en",
    langfuse_handler=None,
    session_id: str | None = None,
) -> ResearchGraphResult:
    workspace.mkdir(parents=True, exist_ok=True)

    role_config = _load_role_config(config_file)
    llm = config[role_config.use_model]
    chat_model = to_chat_openai(llm)

    async with AgentEnv(workspace) as env:
        tools = build_tools_for_role(
            role_config,
            env._tools_dict,
            env._server_tools,
            finalize_overrides={"agent_name": "Research"},
            llm=llm,
        )
        tool_names = [t.name for t in tools]

        system_text = _build_system_prompt(role_config, language, config)

        prompt_template = Template(role_config.instruction, undefined=StrictUndefined)
        instruction_text = prompt_template.render(
            prompt=req.deepresearch_prompt,
            outline_path=str(outline_path) if outline_path else None,
        )

        show_agent_start("Research", None)
        graph = build_graph(chat_model, tools, context_window=config.context_window)

        initial_state = {
            "messages": [
                SystemMessage(content=system_text),
                HumanMessage(content=instruction_text),
            ],
            "turn_count": 0,
            "max_turns": None,
            "context_length": 0,
            "context_window": config.context_window,
            "context_warning": -1 if config.context_folding else 0,
            "agent_name": "Research",
            "final_outcome": None,
            "llm_call_log": [],
        }

        run_config: dict = {"recursion_limit": _RECURSION_LIMIT}
        if langfuse_handler is not None:
            run_config["callbacks"] = [langfuse_handler]
            if session_id:
                run_config["metadata"] = {"langfuse_session_id": session_id}

        final_state = await graph.ainvoke(initial_state, config=run_config)

    manuscript_path = final_state.get("final_outcome")
    if not manuscript_path:
        raise RuntimeError("Research agent did not call finalize with a confirmed outcome.")

    messages_log: list[dict] = []
    prompt_tokens = completion_tokens = total_tokens = 0
    for m in final_state["messages"]:
        if isinstance(m, AIMessage):
            messages_log.append({"role": "assistant", "text": _message_preview(m.content)})
            usage = getattr(m, "usage_metadata", None)
            if usage:
                prompt_tokens += usage.get("input_tokens", 0) or 0
                completion_tokens += usage.get("output_tokens", 0) or 0
                total_tokens += usage.get("total_tokens", 0) or 0
        elif isinstance(m, ToolMessage):
            messages_log.append({"role": "tool", "text": _message_preview(m.content)})

    cost = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}

    _save_history(
        workspace, "Research", final_state["messages"], llm.model_name, total_tokens, cost, tool_names
    )
    _save_llm_call_log(workspace, "Research", final_state.get("llm_call_log", []))

    return ResearchGraphResult(
        manuscript_path=manuscript_path,
        messages_log=messages_log,
        turn_count=final_state["turn_count"],
        cost=cost,
    )
