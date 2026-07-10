"""Parallel Design-agent entrypoint (design-hynix-template pilot only).

Splits the manuscript into its `---`-separated sections and runs one
independent LangGraph conversation per section concurrently, instead of the
single continuous conversation `design_graph.py` uses. `design_graph.py` is
left untouched — this module duplicates the small set of helpers it needs
(same principle already applied when research_graph.py was added: keep the
already-verified sequential path risk-free, copy rather than share).

Architecture (see the plan for the full rationale):
  Phase 0 (sequential) - one LangGraph run produces slides/global.css, the
                         cover slide, and the closing slide.
  Phase 1 (parallel)   - one independent LangGraph run per manuscript section,
                         gated by a semaphore, each producing exactly one
                         body slide that reads (but never rewrites)
                         slides/global.css.
  Phase 2 (no LLM)     - deterministic checks across the finished deck
                         (missing files, no slide has a chart, same template
                         used 3+ times in a row) and, only for slides that
                         violate something, a single regeneration re-run of
                         that slide's Phase-1 worker with an added instruction.
"""

import asyncio
import re
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deeppresenter.agents.env import AgentEnv
from deeppresenter.graph.design_graph import DesignGraphResult, _build_system_prompt, _save_history
from deeppresenter.graph.engine import build_graph
from deeppresenter.graph.llm_adapter import to_chat_openai
from deeppresenter.graph.tools import build_tools_for_role
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.constants import PACKAGE_DIR
from deeppresenter.utils.log import debug, show_agent_start
from deeppresenter.utils.typings import InputRequest, RoleConfig

_HYNIX_TEMPLATE_DIR = str(PACKAGE_DIR / "roles" / "templates" / "hynix")
_RECURSION_LIMIT = 500
_MAX_REGEN_ROUNDS = 2

_SECTION_SPLIT_RE = re.compile(r"(?m)^[ \t]*-{3,}[ \t]*$")
_TEMPLATE_MARKER_RE = re.compile(r"<!--\s*template-used:\s*(\S+)\s*-->")


def _split_manuscript_sections(text: str) -> list[str]:
    """Split a manuscript into its `---`-separated body sections.

    Raises ValueError if the delimiter never appears, since the parallel
    pipeline has no sequential fallback — the manuscript must already follow
    the per-slide `---` convention that Research.yaml produces."""
    pieces = [p.strip() for p in _SECTION_SPLIT_RE.split(text) if p.strip()]
    if len(pieces) < 2:
        raise ValueError(
            "Manuscript has no '---' section separators — cannot split it for "
            "parallel Design generation. Either add '---' between slide "
            "sections, or call this endpoint with parallel=false."
        )
    return pieces


def _load_role_config(filename: str) -> RoleConfig:
    role_config_file = PACKAGE_DIR / "roles" / filename
    if not role_config_file.exists():
        raise FileNotFoundError(f"Role config not found: {role_config_file}")
    with open(role_config_file, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return RoleConfig(**config_data)


def _summarize(final_state: dict) -> tuple[list[dict], dict]:
    """Same accounting as design_graph.run_design_graph's post-loop (lines
    200-213): per-message log preview + summed token usage for this one
    sub-run's messages."""
    messages_log: list[dict] = []
    prompt_tokens = completion_tokens = total_tokens = 0
    for m in final_state["messages"]:
        if isinstance(m, AIMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            messages_log.append({"role": "assistant", "text": text[:200]})
            usage = getattr(m, "usage_metadata", None)
            if usage:
                prompt_tokens += usage.get("input_tokens", 0) or 0
                completion_tokens += usage.get("output_tokens", 0) or 0
                total_tokens += usage.get("total_tokens", 0) or 0
        elif isinstance(m, ToolMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            messages_log.append({"role": "tool", "text": text[:200]})
    cost = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}
    return messages_log, cost


async def _run_role(
    config: DeepPresenterConfig,
    workspace: Path,
    role_config: RoleConfig,
    instruction_text: str,
    language: str,
    agent_label: str,
    langfuse_handler=None,
    session_id: str | None = None,
) -> dict:
    """Runs one independent LangGraph conversation for the given role/instruction
    and returns its final graph state. Shared by the Phase-0 prepass and every
    Phase-1 section worker — each call gets its own AgentEnv (safe for
    concurrent use, confirmed: AgentEnv keeps all tool state on the instance,
    no module-level/global mutable state)."""
    llm = config[role_config.use_model]
    chat_model = to_chat_openai(llm)

    async with AgentEnv(workspace) as env:
        tools = build_tools_for_role(
            role_config,
            env._tools_dict,
            env._server_tools,
            finalize_overrides={"agent_name": "Design"},
        )
        system_text = _build_system_prompt(
            role_config, language, [t.name for t in tools], workspace, env.cutoff_len, config
        )

        show_agent_start(agent_label, None)
        graph = build_graph(chat_model, tools, context_window=config.context_window)

        initial_state = {
            "messages": [SystemMessage(content=system_text), HumanMessage(content=instruction_text)],
            "turn_count": 0,
            "max_turns": None,
            "context_length": 0,
            "context_window": config.context_window,
            "context_warning": -1 if config.context_folding else 0,
            "agent_name": agent_label,
            "final_outcome": None,
        }

        run_config: dict = {"recursion_limit": _RECURSION_LIMIT}
        if langfuse_handler is not None:
            run_config["callbacks"] = [langfuse_handler]
            if session_id:
                run_config["metadata"] = {"langfuse_session_id": session_id}

        final_state = await graph.ainvoke(initial_state, config=run_config)

    if not final_state.get("final_outcome"):
        raise RuntimeError(f"{agent_label} did not call finalize with a confirmed outcome.")
    return final_state


async def _run_prepass(
    config: DeepPresenterConfig,
    workspace: Path,
    req: InputRequest,
    markdown_file: str,
    template_content: str,
    language: str,
    total_slides: int,
    langfuse_handler=None,
    session_id: str | None = None,
) -> dict:
    role_config = _load_role_config("design-hynix-prepass.yaml")
    prompt_template = Template(role_config.instruction, undefined=StrictUndefined)
    instruction_text = prompt_template.render(
        markdown_file=markdown_file,
        prompt=req.designagent_prompt,
        template_content=template_content,
        template_dir=_HYNIX_TEMPLATE_DIR,
        total_slides=total_slides,
        last_slide_filename=f"slide_{total_slides:02d}.html",
    )
    return await _run_role(
        config, workspace, role_config, instruction_text, language,
        "Design-prepass", langfuse_handler, session_id,
    )


async def _run_section_worker(
    config: DeepPresenterConfig,
    workspace: Path,
    section_text: str,
    slide_number: int,
    language: str,
    extra_instruction: str = "",
    langfuse_handler=None,
    session_id: str | None = None,
) -> dict:
    role_config = _load_role_config("design-hynix-section.yaml")
    prompt_template = Template(role_config.instruction, undefined=StrictUndefined)
    instruction_text = prompt_template.render(
        template_dir=_HYNIX_TEMPLATE_DIR,
        global_css_path=str(workspace / "slides" / "global.css"),
        slide_filename=f"slide_{slide_number:02d}.html",
        section_text=section_text,
        extra_instruction=extra_instruction,
    )
    return await _run_role(
        config, workspace, role_config, instruction_text, language,
        f"Design-section-{slide_number:02d}", langfuse_handler, session_id,
    )


def _check_deck_constraints(slides_dir: Path, total_slides: int) -> dict[int, str]:
    """Deterministic, LLM-free checks across the finished deck. Returns
    {slide_number: reason} for every body slide (2..total_slides-1; the cover
    and closing slides are Phase-0's fixed responsibility and not re-checked
    here) that needs to be regenerated."""
    violations: dict[int, str] = {}
    body_range = range(2, total_slides)  # slide_02..slide_{total_slides-1}

    templates_by_slide: dict[int, str] = {}
    has_chart = False
    for n in body_range:
        path = slides_dir / f"slide_{n:02d}.html"
        if not path.exists():
            violations[n] = (
                "This slide is missing entirely — create it from scratch following "
                "the assigned manuscript content and the shared slides/global.css style."
            )
            continue
        html = path.read_text(encoding="utf-8", errors="replace")
        if "data-chart-type" in html:
            has_chart = True
        m = _TEMPLATE_MARKER_RE.search(html)
        if m:
            templates_by_slide[n] = m.group(1)

    # Same-template-3-times-in-a-row check, in slide order.
    run_template, run_length = None, 0
    for n in body_range:
        if n in violations:
            run_template, run_length = None, 0
            continue
        tmpl = templates_by_slide.get(n)
        if tmpl is not None and tmpl == run_template:
            run_length += 1
        else:
            run_template, run_length = tmpl, 1
        if run_length >= 3:
            violations[n] = (
                f"You reused the '{run_template}' template for 3 or more slides in a "
                "row (including this one). Pick a different template that still fits "
                "this content — do not use the same template more than 2 consecutive times."
            )
            run_template, run_length = None, 0  # restart the run after fixing this one

    if not has_chart and body_range:
        # Heuristic: the section with the most digits is the most likely
        # candidate to hold chartable numeric data.
        candidates = [n for n in body_range if n not in violations]
        if candidates:
            def _digit_count(n: int) -> int:
                p = slides_dir / f"slide_{n:02d}.html"
                text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                return sum(c.isdigit() for c in text)

            target = max(candidates, key=_digit_count)
            violations[target] = (
                "The deck currently has no slide with a `data-chart-type` element. "
                "Redesign this slide using `template4.html` (or otherwise add a "
                "`data-chart-type` element) so its numeric data is rendered as a chart."
            )

    return violations


async def run_design_graph_parallel(
    config: DeepPresenterConfig,
    workspace: Path,
    req: InputRequest,
    markdown_file: str,
    template_content: str = "",
    language: str = "en",
    langfuse_handler=None,
    session_id: str | None = None,
    max_workers: int = 4,
) -> DesignGraphResult:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "slides").mkdir(exist_ok=True)

    manuscript_text = Path(markdown_file).read_text(encoding="utf-8")
    sections = _split_manuscript_sections(manuscript_text)
    total_slides = len(sections) + 2  # + cover + closing
    slides_dir = workspace / "slides"

    all_messages_log: list[dict] = []
    total_cost = {"prompt": 0, "completion": 0, "total": 0}
    total_turns = 0
    all_raw_messages: list[BaseMessage] = []
    llm_model_name = config[_load_role_config("design-hynix-section.yaml").use_model].model_name

    def _accumulate(final_state: dict) -> None:
        nonlocal total_turns
        log, cost = _summarize(final_state)
        all_messages_log.extend(log)
        for k in total_cost:
            total_cost[k] += cost[k]
        total_turns += final_state["turn_count"]
        all_raw_messages.extend(final_state["messages"])

    prepass_state = await _run_prepass(
        config, workspace, req, markdown_file, template_content, language,
        total_slides, langfuse_handler, session_id,
    )
    _accumulate(prepass_state)

    for fixed_path in (slides_dir / "slide_01.html", slides_dir / f"slide_{total_slides:02d}.html"):
        if not fixed_path.exists():
            raise RuntimeError(f"Design prepass finished but {fixed_path.name} was not created.")

    sem = asyncio.Semaphore(max_workers)

    async def _guarded_worker(slide_number: int, section_text: str, extra_instruction: str = ""):
        async with sem:
            return await _run_section_worker(
                config, workspace, section_text, slide_number, language,
                extra_instruction, langfuse_handler, session_id,
            )

    body_slide_numbers = list(range(2, total_slides))  # len == len(sections)
    results = await asyncio.gather(
        *[_guarded_worker(n, text) for n, text in zip(body_slide_numbers, sections)],
        return_exceptions=True,
    )

    section_states: dict[int, dict] = {}
    for n, result in zip(body_slide_numbers, results):
        if isinstance(result, BaseException):
            debug(f"[DesignParallel] section slide_{n:02d} failed: {result}")
        else:
            section_states[n] = result

    for round_num in range(_MAX_REGEN_ROUNDS):
        violations = _check_deck_constraints(slides_dir, total_slides)
        if not violations:
            break
        debug(f"[DesignParallel] regen round {round_num + 1}: {violations}")
        section_index = {n: sections[n - 2] for n in body_slide_numbers}
        regen_results = await asyncio.gather(
            *[_guarded_worker(n, section_index[n], reason) for n, reason in violations.items()],
            return_exceptions=True,
        )
        for n, result in zip(violations.keys(), regen_results):
            if isinstance(result, BaseException):
                debug(f"[DesignParallel] regen of slide_{n:02d} failed: {result}")
            else:
                section_states[n] = result
    else:
        remaining = _check_deck_constraints(slides_dir, total_slides)
        if remaining:
            debug(f"[DesignParallel] giving up after {_MAX_REGEN_ROUNDS} regen rounds: {remaining}")

    for n in body_slide_numbers:
        if n in section_states:
            _accumulate(section_states[n])

    _save_history(workspace, "Design", all_raw_messages, llm_model_name, total_cost["total"], total_cost, [])

    return DesignGraphResult(
        slides_dir=str(slides_dir),
        messages_log=all_messages_log,
        turn_count=total_turns,
        cost=total_cost,
    )
