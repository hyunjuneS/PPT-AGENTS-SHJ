"""Local tool implementations for the DeepPresenter agents."""

import asyncio
import subprocess
from pathlib import Path

from deeppresenter.utils.constants import TOOL_CUTOFF_LEN
from deeppresenter.utils.log import debug, warning


# ── finalize ──────────────────────────────────────────────────────────────────

def finalize(outcome: str, agent_name: str = "") -> str:
    """
    When all tasks are finished, call this to finalize the loop.
    outcome: path to the final output file or directory.
    """
    path = Path(outcome)
    assert path.exists(), f"Outcome path does not exist: {outcome}"

    if agent_name == "Planner":
        assert path.suffix == ".json", f"Planner outcome must be a .json file, got {path.suffix}"
    elif agent_name == "Research":
        assert path.suffix == ".md", f"Research outcome must be a .md file, got {path.suffix}"

    debug(f"Agent {agent_name} finalized outcome: {outcome}")
    return outcome


FINALIZE_SPEC = {
    "type": "function",
    "function": {
        "name": "finalize",
        "description": "When all tasks are finished, call this function to finalize the loop.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "description": "The path to the final outcome file or directory.",
                }
            },
            "required": ["outcome"],
        },
    },
}


# ── read_file ─────────────────────────────────────────────────────────────────

def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
    """
    Read a text file. Use offset/limit for large files.
    offset: starting line number (0-based).
    limit: max lines to return.
    """
    p = Path(path)
    assert p.exists(), f"File not found: {path}"
    lines = p.read_text(encoding="utf-8").splitlines()
    chunk = lines[offset: offset + limit]
    result = "\n".join(chunk)
    if len(result) > TOOL_CUTOFF_LEN:
        result = result[:TOOL_CUTOFF_LEN] + f"\n... (truncated, use offset={offset+limit} to continue)"
    return result


READ_FILE_SPEC = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read contents of a local text file. Use offset and limit for large files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file."},
                "offset": {"type": "integer", "description": "Starting line number (0-based). Default 0."},
                "limit": {"type": "integer", "description": "Max lines to return. Default 200."},
            },
            "required": ["path"],
        },
    },
}


# ── write_file ────────────────────────────────────────────────────────────────

def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written {len(content)} chars to {path}"


WRITE_FILE_SPEC = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write text content to a file. Creates parent directories if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to write to."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
        },
    },
}


# ── execute_command ───────────────────────────────────────────────────────────

async def execute_command(command: str, timeout: int = 30) -> str:
    """Run a shell command and return its output."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > TOOL_CUTOFF_LEN:
            output = output[:TOOL_CUTOFF_LEN] + "\n... (truncated)"
        return output or "(no output)"
    except asyncio.TimeoutError:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


EXECUTE_COMMAND_SPEC = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Execute a shell command and return its stdout/stderr output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "timeout": {"type": "integer", "description": "Timeout in seconds. Default 30."},
            },
            "required": ["command"],
        },
    },
}


# ── inspect_manuscript ────────────────────────────────────────────────────────

def inspect_manuscript(path: str) -> str:
    """
    Basic validation of a markdown manuscript.
    Checks that it has at least one --- separator and is non-empty.
    """
    p = Path(path)
    assert p.exists() and p.suffix == ".md", f"Not a valid .md file: {path}"
    content = p.read_text(encoding="utf-8")
    assert content.strip(), "Manuscript is empty"
    pages = [s.strip() for s in content.split("---") if s.strip()]
    return f"Manuscript looks good: {len(pages)} page(s), {len(content)} chars."


INSPECT_MANUSCRIPT_SPEC = {
    "type": "function",
    "function": {
        "name": "inspect_manuscript",
        "description": "Validate a Markdown manuscript file. Returns page count and size.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the .md manuscript file."},
            },
            "required": ["path"],
        },
    },
}


# ── registry ──────────────────────────────────────────────────────────────────

ALL_TOOLS: dict[str, tuple[dict, object]] = {
    "finalize":           (FINALIZE_SPEC,          finalize),
    "read_file":          (READ_FILE_SPEC,          read_file),
    "write_file":         (WRITE_FILE_SPEC,         write_file),
    "execute_command":    (EXECUTE_COMMAND_SPEC,    execute_command),
    "inspect_manuscript": (INSPECT_MANUSCRIPT_SPEC, inspect_manuscript),
}
