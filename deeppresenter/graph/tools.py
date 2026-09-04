"""Wraps deeppresenter/tools/task.py's plain functions + hand-written OpenAI
function-calling specs as LangChain StructuredTools, without rewriting either
the tool implementations or their schemas. Toolset filtering (which tools a
role YAML allows) is shared with the old engine via deeppresenter.utils.toolset.
"""

import inspect
from typing import Any, Callable, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from deeppresenter.tools.task import ALL_TOOLS, InspectContentLimiter
from deeppresenter.utils.constants import HEAVY_CONTENT_REVIEW
from deeppresenter.utils.toolset import resolve_toolset
from deeppresenter.utils.typings import RoleConfig

def _bind_kwargs(func: Callable, **bound_kwargs: Any) -> Callable:
    """functools.partial isn't introspectable via typing.get_type_hints, which
    LangGraph's ToolNode uses internally — so bind extra kwargs (e.g.
    agent_name="Design" for finalize) with a plain wrapper function instead."""
    if inspect.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs, **bound_kwargs)
        return async_wrapper

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs, **bound_kwargs)
    return wrapper


_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _build_args_model(spec: dict, func: Callable) -> type[BaseModel]:
    """Build a pydantic args-schema from task.py's own hand-written JSON schema
    (spec["function"]["parameters"]) — not re-derived from type hints/docstrings,
    so the model-facing schema stays byte-for-byte equivalent to what the old
    engine already sends. Optional fields fall back to the wrapped function's
    real Python default (not None) so an omitted arg behaves identically."""
    sig = inspect.signature(func)
    params = spec["function"]["parameters"]
    properties: dict[str, dict] = params.get("properties", {})
    required = set(params.get("required", []))

    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        if "enum" in prop:
            py_type: Any = Literal[tuple(prop["enum"])]
        else:
            py_type = _JSON_TYPE_MAP.get(prop.get("type"), str)
        description = prop.get("description", "")

        if name in required:
            fields[name] = (py_type, Field(..., description=description))
        else:
            default = sig.parameters[name].default if name in sig.parameters else None
            if default is inspect.Parameter.empty:
                default = None
            fields[name] = (py_type | None, Field(default=default, description=description))

    model_name = "".join(w.capitalize() for w in spec["function"]["name"].split("_")) + "Args"
    return create_model(model_name, **fields)


def build_tools_for_role(
    role_config: RoleConfig,
    tools_dict: dict[str, dict],
    server_tools: dict[str, list[str]],
    finalize_overrides: dict[str, Any] | None = None,
    llm: Any | None = None,
    vlm_llm: Any | None = None,
    expected_pages: int | None = None,
) -> list[StructuredTool]:
    """finalize_overrides lets the caller bind e.g. agent_name="Design" onto the
    finalize tool at construction time (mirrors the old engine's post-hoc
    args["agent_name"] = self.name injection in Agent.execute(), agent.py:275-276,
    without mutating call arguments after the fact).

    llm, if given, is bound onto inspect_content the same way — so its content
    review runs on the exact same model/base_url/api_key the calling role's own
    chat_model uses, not a separately-configured one. inspect_content also always
    gets a fresh InspectContentLimiter bound here (one per build_tools_for_role
    call, i.e. per agent run) so it can't be called an unbounded number of times.
    inspect_content itself is only included in the built toolset at all when
    HEAVY_CONTENT_REVIEW is enabled — otherwise it's left out entirely, same as if
    the role's own YAML excluded it, so the model never sees it as an option.

    vlm_llm, if given, is bound onto inspect_slide's HEAVY_REFLECT overlap check —
    a separate, dedicated vision model requested independently of the role's own
    chat_model, so the agent that writes the HTML never itself needs to be
    vision-capable.

    expected_pages, if given, is bound onto inspect_manuscript the same way, so it
    can report an explicit page-count mismatch instead of leaving the LLM to judge
    the target count itself."""
    specs = resolve_toolset(role_config.toolset, tools_dict, server_tools)

    structured_tools: list[StructuredTool] = []
    for spec in specs:
        name = spec["function"]["name"]
        if name == "inspect_content" and not HEAVY_CONTENT_REVIEW:
            continue
        raw_func = ALL_TOOLS[name][1]
        func: Callable = raw_func
        if name == "finalize" and finalize_overrides:
            func = _bind_kwargs(raw_func, **finalize_overrides)
        elif name == "inspect_content":
            overrides: dict[str, Any] = {"limiter": InspectContentLimiter()}
            if llm is not None:
                overrides["llm"] = llm
            func = _bind_kwargs(raw_func, **overrides)
        elif name == "inspect_slide" and vlm_llm is not None:
            func = _bind_kwargs(raw_func, vlm_llm=vlm_llm)
        elif name == "inspect_manuscript" and expected_pages is not None:
            func = _bind_kwargs(raw_func, expected_pages=expected_pages)

        args_model = _build_args_model(spec, raw_func)
        kwargs: dict[str, Any] = dict(
            name=name,
            description=spec["function"]["description"],
            args_schema=args_model,
        )
        if inspect.iscoroutinefunction(raw_func):
            structured_tools.append(StructuredTool.from_function(coroutine=func, **kwargs))
        else:
            structured_tools.append(StructuredTool.from_function(func=func, **kwargs))

    return structured_tools
