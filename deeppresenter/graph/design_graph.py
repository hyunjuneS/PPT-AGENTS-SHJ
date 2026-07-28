"""Design-agent entrypoint for the new LangGraph-based engine (Phase 1 pilot).

Mirrors deeppresenter/agents/design.py's Design.loop() + the relevant slices of
Agent.__init__/action()/execute(), but as a single awaitable function instead
of a stateful class + async generator, since main-ui.py never streams
intermediate items to the HTTP client anyway (see main-ui.py's design
endpoints: it only accumulates a `turns` count and truncated previews).
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deeppresenter.agents.env import AgentEnv
from deeppresenter.graph.engine import build_graph
from deeppresenter.graph.llm_adapter import to_chat_openai
from deeppresenter.graph.tools import build_tools_for_role
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.constants import (
    AGENT_PROMPT,
    CONTEXT_MODE_PROMPT,
    MAX_TOOLCALL_PER_TURN,
    OFFLINE_PROMPT,
    PACKAGE_DIR,
)
from deeppresenter.utils.log import show_agent_start
from deeppresenter.utils.typings import InputRequest, RoleConfig

_HYNIX_TEMPLATE_DIR = str(PACKAGE_DIR / "roles" / "templates" / "hynix")
_HYNIX_LOGO_FILENAME = "ppt-main_logo.png"
_RECURSION_LIMIT = 500  # old engine had no hard turn cap for Design; generous headroom here

_LANG_INSTRUCTION = {
    "ko": "IMPORTANT: Write all output text content (slides, manuscripts) in Korean (한국어로 작성하세요).",
    "en": "IMPORTANT: Write all output text content (slides, manuscripts) in English.",
}


@dataclass
class DesignGraphResult:
    slides_dir: str
    messages_log: list[dict]
    turn_count: int
    cost: dict


def _load_role_config(config_file: str | Path | None) -> RoleConfig:
    role_config_file = Path(config_file) if config_file else PACKAGE_DIR / "roles" / "Design.yaml"
    if not role_config_file.exists():
        raise FileNotFoundError(f"Role config not found: {role_config_file}")
    with open(role_config_file, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return RoleConfig(**config_data)


def _build_system_prompt(
    role_config: RoleConfig,
    language: str,
    tool_names: list[str],
    workspace: Path,
    cutoff_len: int,
    config: DeepPresenterConfig,
) -> str:
    system = role_config.system
    system += f"\n\n{_LANG_INSTRUCTION.get(language, _LANG_INSTRUCTION['en'])}"

    if "execute_command" in tool_names:
        system += AGENT_PROMPT.format(
            workspace=str(workspace),
            cutoff_len=cutoff_len,
            time=datetime.now().strftime("%Y-%m-%d"),
            max_toolcall_per_turn=MAX_TOOLCALL_PER_TURN,
        )
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
    used as a baseline/parity check against the old engine during the pilot."""
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


async def run_design_graph(
    config: DeepPresenterConfig,
    workspace: Path,
    req: InputRequest,
    markdown_file: str,
    template_content: str = "",
    config_file: str | Path | None = None,
    language: str = "en",
    langfuse_handler=None,
    session_id: str | None = None,
) -> DesignGraphResult:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "slides").mkdir(exist_ok=True)

    role_config = _load_role_config(config_file)

    if "template_dir" in role_config.instruction:
        # cover-page.html(등 하이닉스 템플릿)이 배경 이미지 등에서 상대경로로 참조할 수 있도록,
        # 슬라이드가 실제로 저장되는 slides/ 안에도 로고를 넣어준다 — 원본 template_dir 기준
        # 상대경로는 slides/에 복사된 slide_01.html 입장에선 해석되지 않는다.
        logo_src = Path(_HYNIX_TEMPLATE_DIR) / _HYNIX_LOGO_FILENAME
        if logo_src.exists():
            shutil.copy(logo_src, workspace / "slides" / _HYNIX_LOGO_FILENAME)

    llm = config[role_config.use_model]
    chat_model = to_chat_openai(llm)

    async with AgentEnv(workspace) as env:
        tools = build_tools_for_role(
            role_config,
            env._tools_dict,
            env._server_tools,
            finalize_overrides={"agent_name": "Design"},
        )
        tool_names = [t.name for t in tools]

        system_text = _build_system_prompt(
            role_config, language, tool_names, workspace, env.cutoff_len, config
        )

        prompt_template = Template(role_config.instruction, undefined=StrictUndefined)
        instruction_text = prompt_template.render(
            markdown_file=markdown_file,
            prompt=req.designagent_prompt,
            template_content=template_content,
            template_dir=_HYNIX_TEMPLATE_DIR,
        )

        show_agent_start("Design", None)
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
            "agent_name": "Design",
            "final_outcome": None,
        }

        run_config: dict = {"recursion_limit": _RECURSION_LIMIT}
        if langfuse_handler is not None:
            run_config["callbacks"] = [langfuse_handler]
            if session_id:
                run_config["metadata"] = {"langfuse_session_id": session_id}

        final_state = await graph.ainvoke(initial_state, config=run_config)

    slides_dir = final_state.get("final_outcome")
    if not slides_dir:
        raise RuntimeError("Design agent did not call finalize with a confirmed outcome.")

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
        workspace, "Design", final_state["messages"], llm.model_name, total_tokens, cost, tool_names
    )

    return DesignGraphResult(
        slides_dir=slides_dir,
        messages_log=messages_log,
        turn_count=final_state["turn_count"],
        cost=cost,
    )
