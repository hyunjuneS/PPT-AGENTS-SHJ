import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from agents.llms import AsyncLLM

logger = logging.getLogger(__name__)

ROLES_DIR = Path(__file__).parent.parent / "roles"


def get_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to extract from ```json ... ``` block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if match:
        response = match.group(1).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from response: %s\nResponse: %s", e, response[:200])
        raise


@dataclass
class Turn:
    id: int
    prompt: str
    response: str
    message: list
    retry: int = -1

    def __eq__(self, other):
        return self is other


class Agent:
    """Agent defined by a YAML role config and backed by an AsyncLLM."""

    def __init__(
        self,
        name: str,
        llm_mapping: dict[str, AsyncLLM],
        config: dict | None = None,
    ):
        self.name = name

        if config is None:
            role_path = ROLES_DIR / f"{name}.yaml"
            with open(role_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

        assert isinstance(config, dict), "Agent config must be a dict"
        self.config = config

        self.llm: AsyncLLM = llm_mapping[config["use_model"]]
        self.system_message: str = config["system_prompt"]
        self.prompt_args: set[str] = set(config["jinja_args"])
        self.return_json: bool = config.get("return_json", False)

        env = Environment(undefined=StrictUndefined)
        self.template = env.from_string(config["template"])

        self._history: list[Turn] = []

    @property
    def next_turn_id(self) -> int:
        if not self._history:
            return 0
        return max(t.id for t in self._history) + 1

    async def __call__(
        self,
        recent: int = 0,
        **jinja_args,
    ) -> tuple[int, str | dict]:
        assert self.prompt_args == set(jinja_args.keys()), (
            f"Expected args {self.prompt_args}, got {set(jinja_args.keys())}"
        )

        prompt = self.template.render(**jinja_args)
        history_msgs = self._build_history_msgs(recent)

        logger.debug("[Agent:%s] turn=%d calling LLM", self.name, self.next_turn_id)

        response, message = await self.llm(
            prompt,
            system_message=self.system_message,
            history=history_msgs,
            return_message=True,
        )

        turn = Turn(
            id=self.next_turn_id,
            prompt=prompt,
            response=response,
            message=message,
        )
        self._history.append(turn)

        result = get_json_from_response(response) if self.return_json else response

        logger.debug("[Agent:%s] turn=%d done", self.name, turn.id)
        return turn.id, result

    def _build_history_msgs(self, recent: int) -> list:
        history = self._history[-recent:] if recent > 0 else []
        msgs = []
        for t in sorted(history, key=lambda x: x.id):
            msgs.extend(t.message)
        return msgs

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, llm={self.llm})"
