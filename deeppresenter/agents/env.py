import asyncio
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deeppresenter.utils.constants import TOOL_CUTOFF_LEN
from deeppresenter.utils.log import debug, info, warning
from deeppresenter.utils.typings import ChatMessage, Role


class AgentEnv:
    """
    Simplified AgentEnv — registers local Python functions as tools
    (no Docker / MCP server required).
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.cutoff_len = TOOL_CUTOFF_LEN
        self._tools_dict: dict[str, dict] = {}       # name → OpenAI spec
        self._tool_funcs: dict[str, Callable] = {}   # name → callable
        self._server_tools: dict[str, list[str]] = {}  # server → [tool names]

    def register_tool(
        self,
        spec: dict,
        func: Callable,
        server: str = "task",
    ):
        name = spec["function"]["name"]
        self._tools_dict[name] = spec
        self._tool_funcs[name] = func
        self._server_tools.setdefault(server, []).append(name)
        debug(f"AgentEnv: registered tool '{name}' on server '{server}'")

    def register_all_tools(self):
        """Register all default tools from deeppresenter.tools.task."""
        from deeppresenter.tools.task import ALL_TOOLS
        for name, (spec, func) in ALL_TOOLS.items():
            self.register_tool(spec, func, server="task")

    async def tool_execute(self, tool_call) -> ChatMessage:
        name = tool_call.function.name
        raw_args = tool_call.function.arguments

        try:
            args: dict[str, Any] = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            return ChatMessage(
                role=Role.TOOL,
                content=f"Failed to parse arguments: {e}",
                tool_call_id=tool_call.id,
                is_error=True,
            )

        func = self._tool_funcs.get(name)
        if func is None:
            return ChatMessage(
                role=Role.TOOL,
                content=f"Unknown tool: {name}",
                tool_call_id=tool_call.id,
                is_error=True,
            )

        info(f"Executing tool '{name}' with args: {list(args.keys())}")
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**args))

            # tool이 content block list를 반환하면 그대로 보존 (이미지 포함 가능)
            if isinstance(result, list):
                return ChatMessage(
                    role=Role.TOOL,
                    content=result,
                    tool_call_id=tool_call.id,
                )

            result_str = str(result)
            if len(result_str) > self.cutoff_len:
                result_str = result_str[:self.cutoff_len] + "\n... (truncated)"

            return ChatMessage(
                role=Role.TOOL,
                content=result_str,
                tool_call_id=tool_call.id,
            )
        except Exception as e:
            warning(f"Tool '{name}' raised: {e}")
            return ChatMessage(
                role=Role.TOOL,
                content=str(e),
                tool_call_id=tool_call.id,
                is_error=True,
            )

    async def __aenter__(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.register_all_tools()
        return self

    async def __aexit__(self, *_):
        pass
