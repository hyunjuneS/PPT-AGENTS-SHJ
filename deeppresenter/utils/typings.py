import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: Role
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: None | str | list[dict]
    reasoning: None | str = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_error: bool = False
    cost: CompletionUsage | None = None
    from_tool: Function | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    extra_info: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _):
        if not isinstance(self.content, list):
            content = []
            if self.content is not None and str(self.content).strip():
                content.append({"type": "text", "text": str(self.content).strip()})
            self.content = content
        else:
            for block in self.content:
                if block.get("type") == "text":
                    block["text"] = block["text"].strip()

    @property
    def text(self) -> str:
        texts = []
        for block in (self.content or []):
            if block.get("type") == "text":
                texts.append(block["text"])
            elif block.get("type") == "image_url":
                texts.append("<image>")
        for tc in (self.tool_calls or []):
            texts.append(tc.function.model_dump_json())
        return texts[0] if len(texts) == 1 else str(texts) if texts else ""

    @property
    def has_image(self) -> bool:
        return any(b.get("type") == "image_url" for b in (self.content or []))

    def to_api_dict(self) -> dict:
        """Convert to OpenAI API message format."""
        d: dict[str, Any] = {"role": self.role.value}

        if self.role == Role.TOOL:
            # tool result: content must be a string
            d["tool_call_id"] = self.tool_call_id
            d["content"] = self.text
            return d

        # content
        if not self.content:
            d["content"] = None
        elif len(self.content) == 1 and self.content[0].get("type") == "text":
            d["content"] = self.content[0]["text"]
        else:
            d["content"] = self.content

        if self.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]

        return d


class ToolSet(BaseModel):
    include_tool_servers: list[str] | str = "all"
    exclude_tool_servers: list[str] = []
    include_tools: list[str] = []
    exclude_tools: list[str] = []


class RoleConfig(BaseModel):
    system: dict[str, str]
    instruction: str
    use_model: str
    toolset: ToolSet = Field(default_factory=ToolSet)


class Cost(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def __add__(self, other: CompletionUsage) -> "Cost":
        self.prompt += other.prompt_tokens or 0
        self.completion += other.completion_tokens or 0
        self.total += other.total_tokens or 0
        return self

    def __repr__(self) -> str:
        return f"{self.prompt/1000:.1f}K prompt + {self.completion/1000:.1f}K completion tokens"


class InputRequest(BaseModel):
    instruction: str
    attachments: list[str] = []
    num_pages: str | None = None
    language: str = "en"

    def copy_to_workspace(self, workspace: Path):
        import shutil
        if not self.attachments:
            return
        (workspace / "attachments").mkdir(parents=True, exist_ok=True)
        new_attachments: list[str] = []
        for att in self.attachments:
            src = Path(att).expanduser().resolve()
            assert src.exists(), f"Attachment {att} does not exist"
            dst = workspace / "attachments" / src.name
            if not dst.exists():
                shutil.copy(src, dst)
            new_attachments.append(str(dst))
        self.attachments = new_attachments

    @property
    def deepresearch_prompt(self) -> str:
        parts = [self.instruction]
        if self.num_pages:
            parts.append("Number of pages: " + self.num_pages)
        if self.attachments:
            parts.append("Attachments: " + ", ".join(self.attachments))
        return "\n".join(parts)

    @property
    def designagent_prompt(self) -> str:
        return self.instruction
