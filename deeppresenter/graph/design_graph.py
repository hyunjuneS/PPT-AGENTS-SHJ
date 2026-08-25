"""Design-agent entrypoint for the new LangGraph-based engine (Phase 1 pilot).

Mirrors deeppresenter/agents/design.py's Design.loop() + the relevant slices of
Agent.__init__/action()/execute(), but as a single awaitable function instead
of a stateful class + async generator, since main-ui.py never streams
intermediate items to the HTTP client anyway (see main-ui.py's design
endpoints: it only accumulates a `turns` count and truncated previews).
"""

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deeppresenter.agents.env import AgentEnv
from deeppresenter.graph.engine import build_graph
from deeppresenter.graph.llm_adapter import to_chat_openai
from deeppresenter.graph.tools import build_tools_for_role
from deeppresenter.tools.task import split_pages
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.constants import (
    CONTEXT_MODE_PROMPT,
    DESIGN_PARALLEL_CHUNK_SIZE,
    DESIGN_PARALLEL_CONCURRENCY,
    HEAVY_REFLECT,
    OFFLINE_PROMPT,
    PACKAGE_DIR,
)
from deeppresenter.utils.log import show_agent_start, warning
from deeppresenter.utils.typings import InputRequest, RoleConfig

_HYNIX_TEMPLATE_DIR = str(PACKAGE_DIR / "roles" / "templates" / "hynix")
_HYNIX_LOGO_FILENAME = "ppt-main_logo.png"
_HYNIX_SECRET_LABEL_FILENAME = "ppt-secret-label.png"
_HYNIX_TEMPLATE_ASSETS = (_HYNIX_LOGO_FILENAME, _HYNIX_SECRET_LABEL_FILENAME)
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
        # 슬라이드가 실제로 저장되는 slides/ 안에도 로고/라벨 이미지를 넣어준다 — 원본 template_dir 기준
        # 상대경로는 slides/에 복사된 slide_01.html 입장에선 해석되지 않는다. main-ui.py의 MinIO
        # 업로드(htmls/combined_html)는 slides/ 안의 이미지 파일을 확장자로 스캔해서 통째로 올리므로,
        # 여기 복사되는 것만으로 그쪽도 자동으로 같이 업로드된다.
        for asset_name in _HYNIX_TEMPLATE_ASSETS:
            asset_src = Path(_HYNIX_TEMPLATE_DIR) / asset_name
            if asset_src.exists():
                shutil.copy(asset_src, workspace / "slides" / asset_name)

    llm = config[role_config.use_model]
    chat_model = to_chat_openai(llm)

    # HEAVY_REFLECT일 때만 별도 VLM 모델을 inspect_slide에 바인딩한다 — Design 에이전트 자신의
    # chat_model(llm, 위)은 이미지를 받지 않고, VLM이 돌려준 텍스트 겹침 리뷰만 전달받는다.
    vlm_llm = config.vlm_agent if HEAVY_REFLECT else None

    async with AgentEnv(workspace) as env:
        tools = build_tools_for_role(
            role_config,
            env._tools_dict,
            env._server_tools,
            finalize_overrides={"agent_name": "Design"},
            vlm_llm=vlm_llm,
        )
        tool_names = [t.name for t in tools]

        system_text = _build_system_prompt(role_config, language, config)

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
            "llm_call_log": [],
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
    _save_llm_call_log(workspace, "Design", final_state.get("llm_call_log", []))

    return DesignGraphResult(
        slides_dir=slides_dir,
        messages_log=messages_log,
        turn_count=final_state["turn_count"],
        cost=cost,
    )


# ── Parallel Design (template-based/hynix path only, see plan doc Area 2) ──────
#
# run_design_graph (above) stays untouched as the serial default/fallback. The
# functions below implement an alternative orchestration: instead of one
# build_graph().ainvoke() writing the whole deck slide-by-slide, a fixed slide
# manifest is computed here in plain Python (never left to the LLM to decide),
# and one independent build_graph().ainvoke() per slide-owning worker runs
# concurrently (asyncio.Semaphore-limited), all sharing the same
# workspace/slides directory and a global.css that a dedicated "Phase A" run
# writes before any worker starts (see DesignPlanPhase-hynix.yaml).
#
# Wired in only for main-ui.py's _run_design_hynix_stage (DESIGN_PARALLEL_MODE
# env flag) — the template-free path is untouched and always uses the serial
# run_design_graph above.

@dataclass
class _WorkerResult:
    tag: str
    slides_dir: str
    messages_log: list[dict] = field(default_factory=list)
    turn_count: int = 0
    cost: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    llm_call_log: list[dict] = field(default_factory=list)


async def _run_design_worker(
    config: DeepPresenterConfig,
    workspace: Path,
    req: InputRequest,
    role_config_file: str | Path,
    render_vars: dict,
    finalize_agent_name: str,
    worker_tag: str,
    language: str = "en",
    langfuse_handler=None,
    session_id: str | None = None,
) -> _WorkerResult:
    """Runs one independent build_graph().ainvoke() for a single Design worker
    role (Phase A, cover, a content chunk, or references) — the same
    AgentEnv/build_graph/graph.ainvoke() pattern run_design_graph uses for the
    whole deck, just scoped to this worker's own role YAML and render_vars.
    Every worker's tools operate on the shared workspace/slides directory, but
    each worker instance (chat_model, tools, graph, AgentEnv) is independent, so
    running several of these concurrently is safe — nothing here holds shared
    mutable state across workers."""
    role_config = _load_role_config(role_config_file)
    llm = config[role_config.use_model]
    chat_model = to_chat_openai(llm)
    vlm_llm = config.vlm_agent if HEAVY_REFLECT else None

    async with AgentEnv(workspace) as env:
        tools = build_tools_for_role(
            role_config,
            env._tools_dict,
            env._server_tools,
            finalize_overrides={"agent_name": finalize_agent_name},
            vlm_llm=vlm_llm,
        )
        tool_names = [t.name for t in tools]

        system_text = _build_system_prompt(role_config, language, config)
        prompt_template = Template(role_config.instruction, undefined=StrictUndefined)
        instruction_text = prompt_template.render(**render_vars)

        agent_label = f"Design-{worker_tag}"
        show_agent_start(agent_label, None)
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
            "agent_name": agent_label,
            "final_outcome": None,
            "llm_call_log": [],
        }

        run_config: dict = {"recursion_limit": _RECURSION_LIMIT}
        if langfuse_handler is not None:
            run_config["callbacks"] = [langfuse_handler]
            if session_id:
                run_config["metadata"] = {"langfuse_session_id": session_id}

        final_state = await graph.ainvoke(initial_state, config=run_config)

    slides_dir = final_state.get("final_outcome")
    if not slides_dir:
        raise RuntimeError(f"Design worker '{worker_tag}' did not call finalize with a confirmed outcome.")

    messages_log: list[dict] = []
    prompt_tokens = completion_tokens = total_tokens = 0
    for m in final_state["messages"]:
        if isinstance(m, AIMessage):
            messages_log.append({"role": "assistant", "text": _message_preview(m.content), "worker": worker_tag})
            usage = getattr(m, "usage_metadata", None)
            if usage:
                prompt_tokens += usage.get("input_tokens", 0) or 0
                completion_tokens += usage.get("output_tokens", 0) or 0
                total_tokens += usage.get("total_tokens", 0) or 0
        elif isinstance(m, ToolMessage):
            messages_log.append({"role": "tool", "text": _message_preview(m.content), "worker": worker_tag})

    cost = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}

    _save_history(
        workspace, agent_label, final_state["messages"], llm.model_name, total_tokens, cost, tool_names
    )
    _save_llm_call_log(workspace, agent_label, final_state.get("llm_call_log", []))

    return _WorkerResult(
        tag=worker_tag,
        slides_dir=slides_dir,
        messages_log=messages_log,
        turn_count=final_state["turn_count"],
        cost=cost,
        llm_call_log=final_state.get("llm_call_log", []),
    )


async def run_design_plan_phase(
    config: DeepPresenterConfig,
    workspace: Path,
    req: InputRequest,
    markdown_file: str,
    language: str = "en",
    langfuse_handler=None,
    session_id: str | None = None,
) -> _WorkerResult:
    """Phase A of parallel Design: decides the deck's shared slide-master style
    and saves slides/global.css. Run alone (awaited before any other worker
    starts) since every other worker's slides link to this file."""
    workspace = Path(workspace)
    slides_dir = workspace / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    render_vars = {
        "markdown_file": markdown_file,
        "prompt": req.designagent_prompt,
        "template_dir": _HYNIX_TEMPLATE_DIR,
        "slides_dir": str(slides_dir),
    }
    return await _run_design_worker(
        config, workspace, req,
        role_config_file=PACKAGE_DIR / "roles" / "DesignPlanPhase-hynix.yaml",
        render_vars=render_vars,
        finalize_agent_name="DesignPlan",
        worker_tag="plan",
        language=language,
        langfuse_handler=langfuse_handler,
        session_id=session_id,
    )


async def run_design_graph_parallel(
    config: DeepPresenterConfig,
    workspace: Path,
    req: InputRequest,
    markdown_file: str,
    language: str = "en",
    langfuse_handler=None,
    session_id: str | None = None,
) -> DesignGraphResult:
    """Parallel Design-agent orchestration for the template-based/hynix path
    only. Splits the manuscript into a fixed slide manifest — cover, N content
    chunks, references, end-page — computed here in plain Python, then runs one
    independent worker per manifest entry, concurrency-limited by
    asyncio.Semaphore(DESIGN_PARALLEL_CONCURRENCY). Any worker failure
    (exception, or a missing expected slide_NN.html once all workers finish)
    fails the whole request — no partial deck is ever returned, since
    _design_response (main-ui.py) assumes a complete slides directory."""
    workspace = Path(workspace)
    slides_dir = workspace / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    for asset_name in _HYNIX_TEMPLATE_ASSETS:
        asset_src = Path(_HYNIX_TEMPLATE_DIR) / asset_name
        if asset_src.exists():
            shutil.copy(asset_src, slides_dir / asset_name)

    manuscript_content = Path(markdown_file).read_text(encoding="utf-8")
    pages = split_pages(manuscript_content)
    page_count = len(pages)
    if page_count < 2:
        raise RuntimeError(
            "Design parallel mode needs at least 2 manuscript pages (cover + closing), "
            f"got {page_count}."
        )

    references_slide_no = page_count + 1
    end_slide_no = page_count + 2
    total_slides = end_slide_no

    show_agent_start("Design-parallel", None)

    plan_result = await run_design_plan_phase(
        config, workspace, req, markdown_file,
        language=language, langfuse_handler=langfuse_handler, session_id=session_id,
    )

    # Read once and embed directly into every downstream worker's instruction text
    # (rather than just telling them the file exists and letting them read_file it
    # themselves) — this guarantees every worker actually sees Phase A's color/font
    # choices before it starts generating, and skips an extra read_file round trip
    # per worker. Workers are separate conversations from Phase A, so they have no
    # other way to know what's actually inside global.css.
    global_css_content = (slides_dir / "global.css").read_text(encoding="utf-8")

    worker_specs: list[dict] = [
        {
            "role_config_file": PACKAGE_DIR / "roles" / "DesignCoverWorker-hynix.yaml",
            "render_vars": {
                "cover_content": pages[0],
                "prompt": req.designagent_prompt,
                "template_dir": _HYNIX_TEMPLATE_DIR,
                "slides_dir": str(slides_dir),
                "global_css_content": global_css_content,
            },
            "worker_tag": "cover",
        },
    ]

    content_start, content_end = 2, page_count  # closing page folds into the last chunk
    page = content_start
    while page <= content_end:
        chunk_end = min(page + DESIGN_PARALLEL_CHUNK_SIZE - 1, content_end)
        # Slice out just this worker's own pages (1-indexed pages -> 0-indexed list),
        # re-joined with the same "---" separator so the worker can still tell where
        # one assigned page ends and the next begins — same as passing the whole
        # manuscript would show, just without the pages it has no business reading.
        assigned_content = "\n\n---\n\n".join(pages[page - 1: chunk_end])
        worker_specs.append({
            "role_config_file": PACKAGE_DIR / "roles" / "DesignContentWorker-hynix.yaml",
            "render_vars": {
                "assigned_manuscript_content": assigned_content,
                "prompt": req.designagent_prompt,
                "template_dir": _HYNIX_TEMPLATE_DIR,
                "slides_dir": str(slides_dir),
                "start_page": page,
                "end_page": chunk_end,
                "global_css_content": global_css_content,
            },
            "worker_tag": f"chunk_{page}-{chunk_end}",
        })
        page = chunk_end + 1

    worker_specs.append({
        "role_config_file": PACKAGE_DIR / "roles" / "DesignReferencesWorker-hynix.yaml",
        "render_vars": {
            "prompt": req.designagent_prompt,
            "template_dir": _HYNIX_TEMPLATE_DIR,
            "slides_dir": str(slides_dir),
            "ref_slide_no": references_slide_no,
            "global_css_content": global_css_content,
        },
        "worker_tag": "references",
    })

    sem = asyncio.Semaphore(DESIGN_PARALLEL_CONCURRENCY)

    async def _run_guarded(spec: dict) -> _WorkerResult:
        async with sem:
            return await _run_design_worker(
                config, workspace, req,
                role_config_file=spec["role_config_file"],
                render_vars=spec["render_vars"],
                finalize_agent_name="Design",
                worker_tag=spec["worker_tag"],
                language=language,
                langfuse_handler=langfuse_handler,
                session_id=session_id,
            )

    results = await asyncio.gather(*(_run_guarded(s) for s in worker_specs), return_exceptions=True)

    end_page_src = Path(_HYNIX_TEMPLATE_DIR) / "end-page.html"
    if end_page_src.exists():
        shutil.copy(end_page_src, slides_dir / f"slide_{end_slide_no:02d}.html")
    else:
        warning(f"end-page.html not found in template dir {_HYNIX_TEMPLATE_DIR} — slide_{end_slide_no:02d}.html not created")

    errors = [r for r in results if isinstance(r, BaseException)]
    expected = {f"slide_{n:02d}.html" for n in range(1, total_slides + 1)}
    actual = {p.name for p in slides_dir.glob("slide_*.html")}
    missing = expected - actual
    if errors or missing:
        raise RuntimeError(
            "Design parallel run incomplete — "
            f"{len(errors)} worker(s) failed: {[str(e) for e in errors]}; "
            f"missing slides: {sorted(missing)}"
        )

    worker_results: list[_WorkerResult] = [plan_result, *results]  # type: ignore[misc]
    messages_log: list[dict] = []
    turn_count = 0
    cost = {"prompt": 0, "completion": 0, "total": 0}
    all_calls: list[dict] = []
    for wr in worker_results:
        messages_log.extend(wr.messages_log)
        turn_count += wr.turn_count
        cost["prompt"] += wr.cost["prompt"]
        cost["completion"] += wr.cost["completion"]
        cost["total"] += wr.cost["total"]
        all_calls.extend(wr.llm_call_log)

    _save_llm_call_log(workspace, "Design-summary", all_calls)

    return DesignGraphResult(
        slides_dir=str(slides_dir),
        messages_log=messages_log,
        turn_count=turn_count,
        cost=cost,
    )
