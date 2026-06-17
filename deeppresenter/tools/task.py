"""Local tool implementations for the DeepPresenter agents."""

import asyncio
import base64
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from deeppresenter.utils.constants import HEAVY_REFLECT, TOOL_CUTOFF_LEN
from deeppresenter.utils.log import debug, warning

_SCREENSHOT_JS = Path(__file__).resolve().parents[1] / "html2pptx" / "screenshot.js"
_DEFAULT_CHROMIUM = Path(
    "/mnt/c/Users/X0160146/Desktop/26/playwright/chromium-1223/chrome-linux64/chrome"
)


def _get_chromium_executable() -> str | None:
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    if _DEFAULT_CHROMIUM.exists():
        return str(_DEFAULT_CHROMIUM)
    return None


async def _screenshot_slide(html_file: str, aspect_ratio: str = "16:9") -> bytes | None:
    """HTML 슬라이드를 Playwright로 렌더링하여 JPEG bytes를 반환. 실패 시 None."""
    if not _SCREENSHOT_JS.exists():
        warning("screenshot.js not found — visual inspect disabled")
        return None

    SIZES = {
        "16:9": (1280, 720), "4:3": (960, 720),
        "A1": (2244, 3178), "A2": (1587, 2244), "A3": (1122, 1587), "A4": (794, 1123),
    }
    w, h = SIZES.get(aspect_ratio, (1280, 720))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        output = f.name

    try:
        env = os.environ.copy()
        chromium_exe = _get_chromium_executable()
        if chromium_exe:
            env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = chromium_exe

        proc = await asyncio.create_subprocess_exec(
            "node", str(_SCREENSHOT_JS),
            "--html", str(Path(html_file).resolve()),
            "--output", output,
            "--width", str(w), "--height", str(h),
            cwd=str(_SCREENSHOT_JS.parent),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode == 0 and Path(output).exists():
            return Path(output).read_bytes()
        warning(f"screenshot.js failed: {stdout.decode(errors='replace')}")
    except Exception as e:
        warning(f"_screenshot_slide error: {e}")
    finally:
        try:
            os.unlink(output)
        except Exception:
            pass
    return None


# ── finalize ──────────────────────────────────────────────────────────────────

def _rewrite_image_links(path: Path) -> None:
    """Research 결과물의 이미지 경로를 절대 경로로 재작성하고
    alt 텍스트에 이미지 비율을 주입 (WSL task.py 완전 동일)."""
    md_dir = path.parent
    content = path.read_text(encoding="utf-8")

    def _replace(match: re.Match) -> str:
        alt_text = match.group(1)
        target = match.group(2).strip()
        if not target:
            return match.group(0)
        parts = re.match(r"([^\s]+)(.*)", target)
        if not parts:
            return match.group(0)
        local_path = parts.group(1).strip("\"'")
        rest = parts.group(2)
        p = Path(local_path)
        if not p.is_absolute() and (md_dir / local_path).exists():
            p = md_dir / local_path
        if not p.exists():
            return match.group(0)

        # 이미지 크기로 비율 계산 → alt에 주입 (Design 에이전트 레이아웃 힌트)
        updated_alt = alt_text
        try:
            from PIL import Image as _Image
            with _Image.open(p) as img:
                width, height = img.size
            if width > 0 and height > 0 and not re.search(r"\b\d+:\d+\b", updated_alt):
                factor = math.gcd(width, height)
                ratio = f"{width // factor}:{height // factor}"
                updated_alt = f"{updated_alt}, {ratio}" if updated_alt else ratio
        except Exception as e:
            warning(f"Failed to get image size for {p}: {e}")

        new_path = p.resolve().as_posix()
        return f"![{updated_alt}]({new_path}{rest})"

    try:
        rewritten = re.sub(r"!\[(.*?)\]\((.*?)\)", _replace, content)
        shutil.copyfile(path, md_dir / ("." + path.name))  # 원본 백업
        path.write_text(rewritten, encoding="utf-8")
    except Exception as e:
        warning(f"Failed to rewrite image links: {e}")


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
        _rewrite_image_links(path)

    elif agent_name == "Design":
        html_files = list(path.glob("*.html"))
        if not html_files:
            return "Outcome path should be a directory containing HTML files"
        if not all(f.stem.startswith("slide_") for f in html_files):
            return "All HTML files should be named slide_NN.html"

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

def read_file(path: str, offset: int = 0, length: int = 200) -> str:
    """
    Read a text file. Use offset/length for large files.
    offset: starting line number (0-based).
    length: max lines to return.
    """
    p = Path(path)
    assert p.exists(), f"File not found: {path}"
    lines = p.read_text(encoding="utf-8").splitlines()
    chunk = lines[offset: offset + length]
    result = "\n".join(chunk)
    if len(result) > TOOL_CUTOFF_LEN:
        result = result[:TOOL_CUTOFF_LEN] + f"\n... (truncated, use offset={offset+length} to continue)"
    return result


READ_FILE_SPEC = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read contents of a local text file. Use offset and length for large files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file."},
                "offset": {"type": "integer", "description": "Starting line number (0-based). Default 0."},
                "length": {"type": "integer", "description": "Max lines to return. Default 200."},
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


# ── edit_file ─────────────────────────────────────────────────────────────────

def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    expected_replacements: int = 1,
) -> str:
    """
    Replace occurrences of old_string with new_string in a file.
    Raises if match count does not equal expected_replacements.
    """
    p = Path(file_path)
    assert p.exists(), f"File not found: {file_path}"
    content = p.read_text(encoding="utf-8")
    count = content.count(old_string)
    assert count > 0, f"old_string not found in {file_path}"
    assert count == expected_replacements, (
        f"old_string matches {count} location(s) in {file_path}, "
        f"expected {expected_replacements} — make it more specific or adjust expected_replacements"
    )
    new_content = content.replace(old_string, new_string, expected_replacements)
    p.write_text(new_content, encoding="utf-8")
    delta = len(new_string) - len(old_string)
    return (
        f"Edited {file_path}: replaced {expected_replacements} occurrence(s) "
        f"(size delta {delta:+d} chars)"
    )


EDIT_FILE_SPEC = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file with new content. "
            "old_string must match exactly expected_replacements times. "
            "Use read_file first to locate the unique context. "
            "Prefer this over write_file when making targeted edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to substitute in place of old_string.",
                },
                "expected_replacements": {
                    "type": "integer",
                    "description": "Number of occurrences to replace. Default 1.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
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


# ── inspect_slide ─────────────────────────────────────────────────────────────

async def inspect_slide(
    html_file: str,
    aspect_ratio: str = "16:9",
) -> str | list:
    """
    Validate an HTML slide file.
    Structural checks first (text-based). If HEAVY_REFLECT is enabled,
    also renders the slide and returns an image for visual VLM review.
    """
    path = Path(html_file)
    assert path.exists() and path.suffix == ".html", \
        f"Not a valid HTML file: {html_file}"

    content = path.read_text(encoding="utf-8")
    assert content.strip(), "HTML file is empty"
    assert "<body" in content.lower(), "HTML file is missing <body> tag"

    issues = []

    SIZES = {
        "16:9": ("1280", "720"),
        "4:3":  ("960",  "720"),
        "A1":   ("2244", "3178"),
        "A2":   ("1587", "2244"),
        "A3":   ("1122", "1587"),
        "A4":   ("794",  "1123"),
    }
    if aspect_ratio in SIZES:
        w, h = SIZES[aspect_ratio]
        if w not in content or h not in content:
            issues.append(f"Body may not have the correct fixed size ({w}x{h}px) for {aspect_ratio}.")

    if "url(" in content and "http" in content:
        issues.append("External image URL detected — images should be local paths.")

    bare_text_issues = _check_bare_text(content)
    issues.extend(bare_text_issues)

    # 구조적 문제가 있으면 먼저 수정하도록 텍스트로 반환 (스크린샷 불필요)
    if issues:
        return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)

    # 구조 검사 통과 — heavy_reflect 모드면 렌더링 이미지 반환
    if HEAVY_REFLECT:
        img_bytes = await _screenshot_slide(html_file, aspect_ratio)
        if img_bytes:
            vlm_dir = path.parent / "vlm_input"
            vlm_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            save_path = vlm_dir / f"{path.stem}_{ts}.jpg"
            save_path.write_bytes(img_bytes)
            debug(f"VLM input image saved: {save_path}")

            b64 = base64.b64encode(img_bytes).decode()
            return [
                {
                    "type": "text",
                    "text": (
                        "Slide structure is valid. "
                        "Review the rendered image below for visual quality "
                        "(layout balance, font readability, overflow, spacing, aesthetics). "
                        "If improvements are needed, rewrite the HTML and call inspect_slide again."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ]

    return f"Slide is valid. ({len(content)} chars, aspect_ratio={aspect_ratio})"


def _check_bare_text(html: str) -> list[str]:
    """
    Block 요소(<div> 등) 안에 텍스트가 <p>/<h1~6>/<li>/<span> 없이
    직접 존재하는 경우를 탐지한다.
    html2pptx.js는 이런 텍스트를 PPTX에 포함하지 않으므로 반드시 수정이 필요하다.
    """
    from html.parser import HTMLParser

    BLOCK_TAGS  = {"div", "section", "header", "footer", "article", "aside", "main", "nav"}
    INLINE_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "span",
                   "a", "strong", "em", "b", "i", "small", "mark", "code"}

    class _Checker(HTMLParser):
        def __init__(self):
            super().__init__()
            self._stack: list[str] = []
            self.found: list[str] = []

        def handle_starttag(self, tag, attrs):
            self._stack.append(tag.lower())

        def handle_endtag(self, tag):
            t = tag.lower()
            for i in range(len(self._stack) - 1, -1, -1):
                if self._stack[i] == t:
                    self._stack.pop(i)
                    break

        def handle_data(self, data):
            text = data.strip()
            if not text:
                return
            if not self._stack:
                return
            parent = self._stack[-1]
            if parent in BLOCK_TAGS:
                preview = text[:30] + ("…" if len(text) > 30 else "")
                self.found.append(
                    f'<{parent.upper()}> contains unwrapped text "{preview}" — '
                    f'wrap it in <p> or <span> so it appears in PowerPoint.'
                )

    checker = _Checker()
    try:
        checker.feed(html)
    except Exception:
        pass
    return checker.found


INSPECT_SLIDE_SPEC = {
    "type": "function",
    "function": {
        "name": "inspect_slide",
        "description": (
            "Validate an HTML slide file after generation. "
            "Checks structure, fixed body size, and common issues. "
            "Call this immediately after writing each slide HTML file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "html_file": {
                    "type": "string",
                    "description": "Absolute path to the .html slide file.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["16:9", "4:3", "A1", "A2", "A3", "A4"],
                    "description": "Slide aspect ratio. Default: 16:9",
                },
            },
            "required": ["html_file"],
        },
    },
}


# ── create_directory ──────────────────────────────────────────────────────────

def create_directory(path: str) -> str:
    """Create a directory (and any missing parents)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Directory created: {path}"


CREATE_DIRECTORY_SPEC = {
    "type": "function",
    "function": {
        "name": "create_directory",
        "description": "Create a directory and any missing parent directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the directory to create."},
            },
            "required": ["path"],
        },
    },
}


# ── list_directory ─────────────────────────────────────────────────────────────

def list_directory(path: str, depth: int = 1) -> str:
    """List the contents of a directory up to the given depth."""
    p = Path(path)
    assert p.exists() and p.is_dir(), f"Directory not found: {path}"

    lines = []

    def _walk(current: Path, current_depth: int) -> None:
        if current_depth > depth:
            return
        for child in sorted(current.iterdir()):
            indent = "  " * (current_depth - 1)
            marker = "/" if child.is_dir() else ""
            lines.append(f"{indent}{child.name}{marker}")
            if child.is_dir() and current_depth < depth:
                _walk(child, current_depth + 1)

    _walk(p, 1)
    result = "\n".join(lines)
    if len(result) > TOOL_CUTOFF_LEN:
        result = result[:TOOL_CUTOFF_LEN] + "\n... (truncated)"
    return result or "(empty)"


LIST_DIRECTORY_SPEC = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": "List the contents of a directory. Use depth to control recursion.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the directory."},
                "depth": {"type": "integer", "description": "Recursion depth. Default 1 (immediate children only)."},
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
    "edit_file":          (EDIT_FILE_SPEC,          edit_file),
    "create_directory":   (CREATE_DIRECTORY_SPEC,   create_directory),
    "list_directory":     (LIST_DIRECTORY_SPEC,     list_directory),
    "execute_command":    (EXECUTE_COMMAND_SPEC,    execute_command),
    "inspect_manuscript": (INSPECT_MANUSCRIPT_SPEC, inspect_manuscript),
    "inspect_slide":      (INSPECT_SLIDE_SPEC,      inspect_slide),
}
