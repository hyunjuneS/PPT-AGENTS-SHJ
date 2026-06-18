import asyncio
import json
import os
import time
import uuid
from abc import abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from jinja2 import Template
from jinja2 import StrictUndefined
from pydantic import BaseModel

from deeppresenter.agents.env import AgentEnv
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.constants import (
    AGENT_PROMPT,
    CONTEXT_MODE_PROMPT,
    CONTINUE_MSG,
    HALF_BUDGET_NOTICE_MSG,
    HIST_LOST_MSG,
    LAST_ITER_MSG,
    MAX_LOGGING_LENGTH,
    MAX_TOOLCALL_PER_TURN,
    MEMORY_COMPACT_MSG,
    OFFLINE_PROMPT,
    PACKAGE_DIR,
    TOOL_CUTOFF_LEN,
    URGENT_BUDGET_NOTICE_MSG,
)
from deeppresenter.utils.log import (
    debug, info, timer,
    show_agent_start, show_agent_turn, show_tool_call, show_tool_result, show_agent_done,
)
from deeppresenter.utils.typings import (
    ChatMessage,
    Cost,
    InputRequest,
    Role,
    RoleConfig,
)


class Agent:
    def __init__(
        self,
        config: DeepPresenterConfig,
        agent_env: AgentEnv,
        workspace: Path,
        language: Literal["ko", "en"] = "en",
        config_file: str | Path | None = None,
        keep_reasoning: bool = True,
        max_turns: int | None = None,
    ):
        self.name = self.__class__.__name__
        self.cost = Cost()
        self.context_length = 0
        self.context_warning = 0
        self.workspace = workspace
        self.agent_env = agent_env
        self.language = language
        self.keep_reasoning = keep_reasoning
        self.context_window = config.context_window
        self.max_context_turns = config.max_context_folds
        self.max_turns = max_turns
        self.turn_count = 0
        self.research_iter = 0
        self._start_time = time.time()

        role_config_file = (
            Path(config_file)
            if config_file
            else PACKAGE_DIR / "roles" / f"{self.name}.yaml"
        )
        if not role_config_file.exists():
            raise FileNotFoundError(f"Role config not found: {role_config_file}")

        workspace.mkdir(parents=True, exist_ok=True)

        with open(role_config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.role_config = RoleConfig(**config_data)
        self.llm = config[self.role_config.use_model]
        self.model = self.llm.model_name

        if config.context_folding:
            self.context_warning = -1

        self._setup_toolset()

        if language not in self.role_config.system:
            language = "en"

        self.system = self.role_config.system[language]
        self.prompt: Template = Template(self.role_config.instruction, undefined=StrictUndefined)

        if any(t["function"]["name"] == "execute_command" for t in self.tools):
            self.system += AGENT_PROMPT.format(
                workspace=str(self.workspace),
                cutoff_len=self.agent_env.cutoff_len,
                time=datetime.now().strftime("%Y-%m-%d"),
                max_toolcall_per_turn=MAX_TOOLCALL_PER_TURN,
            )

        if config.offline_mode:
            self.system += OFFLINE_PROMPT

        if config.context_folding:
            self.system += CONTEXT_MODE_PROMPT

        self.chat_history: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system)
        ]

        available = [t["function"]["name"] for t in self.tools]
        debug(f"{self.name} Agent: {len(self.tools)} tools: {', '.join(available)}")
        show_agent_start(self.name, self.max_turns)

    def _setup_toolset(self):
        toolset = self.role_config.toolset

        if toolset.include_tool_servers == "all":
            servers = list(self.agent_env._server_tools.keys())
        else:
            servers = toolset.include_tool_servers

        self.tools: list[dict] = []
        for server in servers:
            if server in toolset.exclude_tool_servers:
                continue
            for tool_name in self.agent_env._server_tools.get(server, []):
                if tool_name not in toolset.exclude_tools:
                    self.tools.append(self.agent_env._tools_dict[tool_name])

        for tool_name in toolset.include_tools:
            spec = self.agent_env._tools_dict.get(tool_name)
            if spec and spec not in self.tools:
                self.tools.append(spec)

    async def chat(
        self,
        message: ChatMessage,
        response_format: type[BaseModel] | None = None,
        **chat_kwargs,
    ) -> ChatMessage:
        if len(self.chat_history) == 1:
            self.chat_history.append(
                ChatMessage(role=Role.USER, content=self.prompt.render(**chat_kwargs))
            )
            self.log_message(self.chat_history[-1])

        self.chat_history.append(message)
        self.log_message(message)

        with timer(f"{self.name} LLM chat"):
            response = await self.llm.run(
                messages=self.chat_history,
                response_format=response_format,
            )

        if response.usage:
            self.cost += response.usage
            self.context_length = response.usage.total_tokens or 0

        msg = response.choices[0].message
        assistant_msg = ChatMessage(
            role=Role.ASSISTANT,
            content=msg.content,
            cost=response.usage,
            reasoning=getattr(msg, "reasoning", None) if self.keep_reasoning else None,
        )
        self.chat_history.append(assistant_msg)
        self.log_message(assistant_msg)
        return assistant_msg

    def _cap_images(self, messages: list) -> list:
        """이미지가 누적되어 모델 한도를 초과하지 않도록 오래된 이미지를 텍스트로 교체."""
        max_images = int(os.environ.get("VLM_MAX_IMAGES", "2"))
        image_indices = [i for i, m in enumerate(messages) if m.has_image]
        to_strip = set(image_indices[:-max_images]) if len(image_indices) > max_images else set()
        if not to_strip:
            return messages
        result = []
        for i, msg in enumerate(messages):
            if i in to_strip:
                text_blocks = [b for b in (msg.content or []) if b.get("type") == "text"]
                text_blocks.append({"type": "text", "text": "(이전 슬라이드 이미지 — 컨텍스트 한도로 제거됨)"})
                stripped = ChatMessage(
                    role=msg.role,
                    content=text_blocks,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=msg.tool_calls,
                )
                result.append(stripped)
            else:
                result.append(msg)
        return result

    async def action(self, **chat_kwargs) -> ChatMessage:
        self.turn_count += 1

        if self.max_turns is not None and self.turn_count > self.max_turns:
            raise RuntimeError(
                f"{self.name} exceeded max turns: {self.turn_count - 1}/{self.max_turns}"
            )

        # WSL: inject "turns running out" warning so the agent wraps up in time
        if (
            self.max_turns is not None
            and self.max_turns - self.turn_count < 2
            and self.chat_history
        ):
            self.chat_history[-1].content.append({
                "type": "text",
                "text": (
                    f"You have only {self.max_turns - self.turn_count} turn(s) left. "
                    "Finish the remaining work and call `finalize` immediately."
                ),
            })

        if len(self.chat_history) == 1:
            self.chat_history.append(
                ChatMessage(role=Role.USER, content=self.prompt.render(**chat_kwargs))
            )
            self.log_message(self.chat_history[-1])

        show_agent_turn(self.name, self.turn_count, self.max_turns)
        with timer(f"{self.name} LLM action (turn {self.turn_count})"):
            response = await self.llm.run(
                messages=self._cap_images(self.chat_history),
                tools=self.tools,
            )

        if response.usage:
            self.cost += response.usage
            self.context_length = response.usage.total_tokens or 0

        msg = response.choices[0].message
        assistant_msg = ChatMessage(
            role=Role.ASSISTANT,
            content=msg.content,
            cost=response.usage,
            tool_calls=msg.tool_calls,
            reasoning=getattr(msg, "reasoning", None) if self.keep_reasoning else None,
        )
        self.chat_history.append(assistant_msg)
        self.log_message(assistant_msg)
        return assistant_msg

    @abstractmethod
    def loop(
        self, req: InputRequest, *args, **kwargs
    ) -> AsyncGenerator[str | ChatMessage, None]: ...

    async def execute(self, tool_calls) -> str | list[ChatMessage]:
        if not tool_calls:
            return []

        finish_id = None
        outcome = None
        coros = []

        for t in tool_calls:
            args_str = t.function.arguments or ""
            try:
                args = json.loads(args_str) if args_str.strip() else {}
                assert isinstance(args, dict)

                if t.function.name == "finalize":
                    args["agent_name"] = self.name
                    finish_id = t.id
                    outcome = args.get("outcome", "")
                    t.function.arguments = json.dumps(args, ensure_ascii=False)

            except (json.JSONDecodeError, AssertionError) as e:
                obs = ChatMessage(
                    role=Role.TOOL,
                    content=str(e),
                    tool_call_id=t.id,
                    is_error=True,
                )
                self.chat_history.append(obs)
                self.log_message(obs)
                continue

            try:
                display_args = json.loads(t.function.arguments or "{}")
            except Exception:
                display_args = {}
            show_tool_call(t.function.name, display_args)
            coros.append(self.agent_env.tool_execute(t))

        observations: list[ChatMessage] = await asyncio.gather(*coros)

        # Show tool results
        for obs in observations:
            show_tool_result(obs.text, obs.is_error)

        self.chat_history.extend(observations)

        if finish_id is not None:
            for obs in observations:
                if obs.tool_call_id == finish_id and obs.text == outcome:
                    elapsed = time.time() - self._start_time
                    show_agent_done(self.name, self.turn_count, elapsed)
                    return obs.text

        # Context budget warnings
        if self.context_warning == 0 and self.context_length > self.context_window * 0.5:
            self.context_warning = 1
            if observations:
                observations[0].content.insert(0, HALF_BUDGET_NOTICE_MSG)
        elif self.context_warning == 1 and self.context_length > self.context_window * 0.8:
            self.context_warning = 2
            if observations:
                observations[0].content.insert(0, URGENT_BUDGET_NOTICE_MSG)

        # tool results already shown via show_tool_result above; debug log only
        for obs in observations:
            debug(f"[{self.name}|tool] {obs.text[:200]}")

        if self.context_length > self.context_window:
            if self.context_warning == -1:
                await self.compact_history()
            else:
                raise RuntimeError(
                    f"{self.name} exceeded context window: {self.context_length}/{self.context_window}"
                )

        return observations

    def log_message(self, msg: ChatMessage):
        text = msg.text
        if len(text) > MAX_LOGGING_LENGTH:
            text = text[:MAX_LOGGING_LENGTH] + "..."
        debug(f"[{self.name}|{msg.role}] {text}")

    async def compact_history(self, keep_head: int = 10, keep_tail: int = 4):
        if keep_head + keep_tail > len(self.chat_history):
            return
        if self.research_iter >= self.max_context_turns:
            return

        self.save_history(message_only=True)
        self.research_iter += 1

        head, tail = self._split_history(keep_head, keep_tail)

        summary_ask = ChatMessage(
            role=Role.USER,
            content=MEMORY_COMPACT_MSG.format(language=self.language),
        )

        response = await self.llm.run(self.chat_history + [summary_ask], tools=self.tools)
        agent_message = response.choices[0].message

        summary_msg = ChatMessage(
            id=f"context_fold_{uuid.uuid4().hex[:8]}",
            role=Role.ASSISTANT,
            content=agent_message.content,
            tool_calls=agent_message.tool_calls,
            reasoning=getattr(agent_message, "reasoning", None) if self.keep_reasoning else None,
        )

        tasks = [self.agent_env.tool_execute(tc) for tc in (summary_msg.tool_calls or [])]
        observations = await asyncio.gather(*tasks)

        if observations:
            observations[-1].content.append(CONTINUE_MSG)
            if self.research_iter >= self.max_context_turns:
                observations[-1].content.append(LAST_ITER_MSG)

        self.chat_history = head + tail + [summary_ask, summary_msg, *observations]

    def _split_history(self, keep_head: int, keep_tail: int):
        head = []
        for msg in self.chat_history:
            if len(head) < keep_head or msg.role == Role.TOOL:
                head.append(msg)
            else:
                break
        if head:
            head[-1].content.append(HIST_LOST_MSG)

        tail = self.chat_history[-keep_tail:]
        for i, m in enumerate(tail):
            if m.role == Role.ASSISTANT and m not in head:
                tail = tail[i:]
                break
        else:
            tail = []

        return head, tail

    def save_history(self, hist_dir: Path | None = None, message_only: bool = False):
        hist_dir = hist_dir or self.workspace / ".history"
        hist_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"-{self.research_iter:02d}" if self.research_iter >= 0 else ""
        history_file = hist_dir / f"{self.name}{suffix}-history.json"

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in self.chat_history], f, ensure_ascii=False, indent=2)

        if message_only:
            return

        config_file = hist_dir / f"{self.name}-config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "model": self.model,
                    "context_length": self.context_length,
                    "cost": self.cost.model_dump(),
                    "tools": [t["function"]["name"] for t in self.tools],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        debug(f"{self.name} done | turns:{self.turn_count} cost:{self.cost} ctx:{self.context_length}")
