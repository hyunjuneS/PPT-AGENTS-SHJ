"""Local tool implementations for the DeepPresenter agents."""

import asyncio
import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from deeppresenter.utils.constants import (
    HEAVY_REFLECT,
    INSPECT_CONTENT_MAX_CALLS,
    READ_FILE_CUTOFF_LEN,
    TOOL_CUTOFF_LEN,
)
from deeppresenter.utils.log import debug, warning

_SCREENSHOT_JS = Path(__file__).resolve().parents[1] / "html2pptx" / "screenshot.js"
# _DEFAULT_CHROMIUM = Path(
#     "/mnt/c/Users/X0160146/Desktop/26/playwright/chromium-1223/chrome-linux64/chrome"
# )


def _get_chromium_executable() -> str | None:
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    # if _DEFAULT_CHROMIUM.exists():
    #     return str(_DEFAULT_CHROMIUM)
    return None


async def screenshot_slide(
    html_file: str, aspect_ratio: str = "16:9", image_format: str = "jpeg", _retry: bool = True
) -> tuple[bytes | None, dict | None]:
    """HTML 슬라이드를 Playwright로 렌더링.
    (이미지 bytes, body 치수 dict{width,height,scrollWidth,scrollHeight}) 반환.
    실패 시 (None, None). image_format은 "jpeg"(기본, VLM 검토용) 또는 "png"
    (main-ui.py의 슬라이드 PNG 갤러리 업로드용) — screenshot.js가 출력 파일 확장자로 판단한다.

    inspect_slide(VLM 겹침 검토)와 main-ui.py의 PNG 스크린샷 업로드가 공유하는 렌더링 로직 —
    두 곳 모두 여기 하나만 거쳐가므로 차트 placeholder 시각화, 한글 폰트 폴백 등 렌더링 방식이
    항상 동일하게 유지된다.

    동시 요청으로 여러 Chromium이 한꺼번에 뜨는 순간의 자원 경합 때문에 launch가
    간헐적으로 죽는 경우가 있어, 실패 시 한 번만 재시도한다.
    """
    if not _SCREENSHOT_JS.exists():
        warning("screenshot.js not found — visual inspect disabled")
        return None, None

    SIZES = {
        "16:9": (1280, 720), "4:3": (960, 720),
        "A1": (2244, 3178), "A2": (1587, 2244), "A3": (1122, 1587), "A4": (794, 1123),
    }
    w, h = SIZES.get(aspect_ratio, (1280, 720))

    suffix = ".png" if image_format == "png" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
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
        stdout_text = stdout.decode(errors="replace")

        dims = None
        for line in stdout_text.splitlines():
            if line.startswith("DIMS:"):
                try:
                    dims = json.loads(line[len("DIMS:"):])
                except json.JSONDecodeError:
                    pass
                break

        if proc.returncode == 0 and Path(output).exists():
            return Path(output).read_bytes(), dims
        warning(f"screenshot.js failed: {stdout_text}")
    except Exception as e:
        warning(f"screenshot_slide error: {e}")
    finally:
        try:
            os.unlink(output)
        except Exception:
            pass

    if _retry:
        debug("Retrying screenshot_slide once after failure")
        return await screenshot_slide(html_file, aspect_ratio, image_format, _retry=False)

    return None, None


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

    elif agent_name == "DesignPlan":
        # Phase A of parallel Design (design_graph.py's run_design_plan_phase) only
        # produces the shared global.css slide-master style — no slide_*.html yet,
        # so it can't use the "Design" branch's html-file check above.
        if not (path / "global.css").exists():
            return "Outcome path should be the slides/ directory containing global.css"

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

def read_file(path: str, offset: int = 0, length: int = 500) -> str:
    """
    Read a text file. Use offset/length for large files.
    offset: starting line number (0-based).
    length: max lines to return.

    Truncation never skips content: if the requested window exceeds
    READ_FILE_CUTOFF_LEN characters, the response is cut back to the last full
    line that still fits, and the "continue" hint advances offset by exactly
    that many lines actually returned — not by the full requested `length` —
    so following the hint literally can never jump over unread content.
    """
    p = Path(path)
    assert p.exists(), f"File not found: {path}"
    lines = p.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    if offset >= total:
        return f"(EOF — file has {total} lines total, offset {offset} is past the end)"
    chunk = lines[offset: offset + length]

    result = "\n".join(chunk)
    if len(result) > READ_FILE_CUTOFF_LEN:
        kept_lines: list[str] = []
        kept_len = 0
        for line in chunk:
            added = len(line) + (1 if kept_lines else 0)  # +1 for the joining "\n"
            if kept_len + added > READ_FILE_CUTOFF_LEN:
                break
            kept_lines.append(line)
            kept_len += added
        lines_returned = max(len(kept_lines), 1)  # always advance by at least one line
        next_offset = offset + lines_returned
        result = (
            "\n".join(chunk[:lines_returned])
            + f"\n... (truncated, {lines_returned}/{len(chunk)} lines shown — "
            f"use offset={next_offset} to continue, do not compute offset+length yourself)"
        )
    return result


READ_FILE_SPEC = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read contents of a local text file. Use offset and length for large files. "
            f"Responses are capped at {READ_FILE_CUTOFF_LEN} characters — if the result says "
            "'truncated', always continue from the exact offset it gives you, never offset+length."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file."},
                "offset": {"type": "integer", "description": "Starting line number (0-based). Default 0."},
                "length": {"type": "integer", "description": "Max lines to return. Default 500."},
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

_PAGE_SEPARATOR_RE = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)


def split_pages(content: str) -> list[str]:
    """Splits a manuscript into pages on lines that are a standalone `---` (or
    longer-dash) separator. Unlike a plain `content.split("---")`, this does not
    miscount a markdown table separator row (`|---|---|`) or any other inline use
    of the substring, since those never occupy an entire line by themselves.
    Shared with design_graph.py's parallel-mode slide manifest calculation, so
    both always agree on how many pages a manuscript has."""
    pages = [s.strip() for s in _PAGE_SEPARATOR_RE.split(content)]
    return [pg for pg in pages if pg]


def inspect_manuscript(path: str, expected_pages: int | None = None) -> str:
    """
    Basic validation of a markdown manuscript.
    Checks that it has at least one --- separator and is non-empty.
    expected_pages, if bound (see build_tools_for_role), reports an explicit
    mismatch instead of leaving the LLM to judge the page count itself.
    """
    p = Path(path)
    assert p.exists() and p.suffix == ".md", f"Not a valid .md file: {path}"
    content = p.read_text(encoding="utf-8")
    assert content.strip(), "Manuscript is empty"
    pages = split_pages(content)

    if expected_pages is not None and len(pages) != expected_pages:
        delta = expected_pages - len(pages)
        action = f"add {delta} page(s)" if delta > 0 else f"remove {-delta} page(s)"
        return f"Page count MISMATCH: expected {expected_pages}, found {len(pages)} — {action}."

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


# ── inspect_content ───────────────────────────────────────────────────────────

# Fallback only — used if this tool is ever invoked without an `llm` bound to it
# (e.g. the old engine, deeppresenter/agents/, which doesn't support bound tool
# kwargs). The graph engine always binds the role's own resolved LLM instead
# (see build_tools_for_role in deeppresenter/graph/tools.py), so this path uses
# the exact same model/base_url/api_key the research agent itself is using.
_FALLBACK_BASE_URL = "http://workplace-litellm.aipp02.skhynix.com/v1"

_MAX_REVIEW_CHARS = 60_000  # cap manuscript+sources sent for this review call only

_CONTENT_REVIEW_SYSTEM_PROMPT = """You are a meticulous editor reviewing a presentation manuscript
(Markdown, pages separated by `---`) for content-quality problems that a page-count/size
check cannot catch. Check for exactly three things:
1. Balance: is content spread reasonably evenly across pages, or are some pages nearly
   empty while others are overloaded? The first page (cover) and last page (closing) are
   expected to be short by design — never flag them for this.
2. Duplication: does any page repeat a fact or point already made on another page?
3. Omission: is there any key fact, data point, or claim in the source documents (if
   provided) that does not appear anywhere in the manuscript?
Respond with STRICT JSON ONLY — no markdown fences, no commentary — matching exactly this
shape: {"issues": ["<specific issue, naming the page number(s) or source file involved>"]}.
If there are no issues, respond with {"issues": []}."""

_CONTENT_REVIEW_USER_TEMPLATE = """<manuscript>
{manuscript}
</manuscript>

{sources_block}

Respond with STRICT JSON ONLY, exactly in this shape: {{"issues": ["..."]}} (empty list if none)."""


def _default_llm():
    from deeppresenter.utils.config import LLM
    return LLM(
        model=os.environ.get("MODEL_BIG", "claude-opus-4-5"),
        base_url=os.environ.get("OPENAI_BASE_URL") or _FALLBACK_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


# ── inspect_slide's VLM overlap review ──────────────────────────────────────

# Fallback only — used if inspect_slide is ever invoked without a `vlm_llm` bound
# to it. The graph engine always binds config.vlm_agent (VLM_MODEL_NAME/VLM_MODEL_URL)
# instead (see build_tools_for_role in deeppresenter/graph/tools.py).
def _default_vlm_llm():
    from deeppresenter.utils.config import LLM
    return LLM(
        model=os.environ.get("VLM_MODEL_NAME", ""),
        base_url=os.environ.get("VLM_MODEL_URL") or os.environ.get("OPENAI_BASE_URL") or _FALLBACK_BASE_URL,
        api_key=os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
    )


_VLM_OVERLAP_SYSTEM_PROMPT = (
    "You are a meticulous visual QA reviewer for presentation slides. You will be shown a "
    "rendered screenshot of one slide. Your ONLY job is to detect visual overlap between "
    "elements — do not evaluate aesthetics, layout balance, font choices, spacing, or "
    "overflow (overflow is already verified separately by a deterministic size check, not by "
    "you). Overlap means any text, image, shape, or chart placeholder box (dashed border, "
    "labeled '[CHART: <type>]') that visually overlaps another text, image, or shape."
)

_VLM_OVERLAP_USER_PROMPT = (
    "Review the attached slide screenshot for element overlap only. "
    "If any element overlaps another, describe exactly which elements overlap and roughly "
    "where on the slide. If no element overlaps another, respond with exactly: No overlap detected."
)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, original length {len(text)} characters)"


class InspectContentLimiter:
    """Caps how many times inspect_content will actually run its LLM review within
    a single agent run — each call is a real LLM request, and nothing in the ReAct
    loop otherwise stops the agent from calling it forever on a "keep checking until
    it's clean" instruction. One instance is created per agent run (see
    build_tools_for_role in deeppresenter/graph/tools.py) and bound onto the tool,
    so the count is per-run, not global/shared across concurrent requests."""

    def __init__(self, max_calls: int = INSPECT_CONTENT_MAX_CALLS):
        self.max_calls = max_calls
        self.count = 0


async def inspect_content(
    path: str,
    sources_dir: str | None = None,
    llm=None,
    limiter: InspectContentLimiter | None = None,
) -> str:
    """
    LLM-based content-quality review of a Markdown manuscript that inspect_manuscript
    (page count/size only) does not catch: whether content is evenly distributed across
    pages, whether any pages duplicate each other, and whether any source document's
    content is missing from the manuscript.
    """
    if limiter is not None:
        if limiter.count >= limiter.max_calls:
            return (
                f"inspect_content has already been called {limiter.count} time(s) — "
                f"the maximum ({limiter.max_calls}) for this run. Do not call it again. "
                "Use your best judgment on any remaining issues, then call finalize."
            )
        limiter.count += 1

    p = Path(path)
    assert p.exists() and p.suffix == ".md", f"Not a valid .md file: {path}"
    content = p.read_text(encoding="utf-8")
    assert content.strip(), "Manuscript is empty"

    src_dir = Path(sources_dir) if sources_dir else p.parent / "sources"
    sources_block = "(no source documents available to check omission against)"
    if src_dir.is_dir():
        src_files = sorted(src_dir.glob("*.md"))
        if src_files:
            parts = [
                f'<source name="{f.name}">\n{f.read_text(encoding="utf-8")}\n</source>'
                for f in src_files
            ]
            sources_block = "<source_documents>\n" + "\n\n".join(parts) + "\n</source_documents>"

    from deeppresenter.utils.config import get_json_from_response

    messages = [
        {"role": "system", "content": _CONTENT_REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _CONTENT_REVIEW_USER_TEMPLATE.format(
                manuscript=_truncate(content, _MAX_REVIEW_CHARS),
                sources_block=_truncate(sources_block, _MAX_REVIEW_CHARS),
            ),
        },
    ]

    response = await (llm or _default_llm()).run(messages=messages)
    text = response.choices[0].message.content or ""
    parsed = get_json_from_response(text)
    issues = parsed.get("issues") if isinstance(parsed, dict) else None
    issues = issues or []

    if issues:
        return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)
    return "Content looks good: no balance, duplication, or omission issues found."


INSPECT_CONTENT_SPEC = {
    "type": "function",
    "function": {
        "name": "inspect_content",
        "description": (
            "Check manuscript content quality beyond page count/size: whether content is "
            "evenly distributed across pages, whether any pages duplicate each other, and "
            "whether any source document's key content is missing from the manuscript. "
            "Call this after inspect_manuscript, before finalize. "
            f"Can be called at most {INSPECT_CONTENT_MAX_CALLS} times per run — budget your "
            "manuscript revisions accordingly, and finalize with your best judgment once "
            "the limit is reached even if minor issues remain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the .md manuscript file."},
                "sources_dir": {
                    "type": "string",
                    "description": (
                        "Absolute path to the directory of source .md files to check coverage against. "
                        "Defaults to a 'sources' folder next to the manuscript."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}


# ── inspect_slide ─────────────────────────────────────────────────────────────

async def inspect_slide(
    html_file: str,
    aspect_ratio: str = "16:9",
    vlm_llm=None,
) -> str:
    """
    Validate an HTML slide file.
    Structural checks first (text-based), then a deterministic content-overflow
    check via a real browser render (scrollWidth/scrollHeight vs the fixed body
    size — this catches overflow even though `overflow:hidden` clips it invisibly
    in any screenshot). If HEAVY_REFLECT is enabled, the rendered image is also
    sent as a standalone request to a dedicated VLM (vlm_llm, or VLM_MODEL_NAME by
    default) to check for element overlap — the calling agent's own model never
    sees the image itself, only the VLM's text verdict.
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

    # 구조적 문제가 있으면 먼저 수정하도록 텍스트로 반환 (렌더링 불필요)
    if issues:
        return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)

    # 구조 검사 통과 — 실제 브라우저 렌더링으로 overflow 여부를 결정적으로 확인.
    # overflow:hidden은 넘친 콘텐츠를 스크린샷에서 안 보이게 가리므로,
    # scrollWidth/scrollHeight로 직접 측정해야만 잡아낼 수 있다.
    img_bytes, dims = await screenshot_slide(html_file, aspect_ratio)

    # HEAVY_REFLECT 모드면 이번 inspect_slide 호출마다(overflow로 반려되더라도) 렌더링
    # 결과를 저장한다 — edit_file → inspect_slide 반복 이력을 slide_02_01, slide_02_02...
    # 순서로 남겨서 무엇이 왜 반려됐는지 나중에 확인할 수 있게 한다.
    if HEAVY_REFLECT and img_bytes:
        vlm_dir = path.parent / "vlm_input"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        seq = len(list(vlm_dir.glob(f"{path.stem}_*.jpg"))) + 1
        save_path = vlm_dir / f"{path.stem}_{seq:02d}.jpg"
        save_path.write_bytes(img_bytes)
        debug(f"VLM input image saved: {save_path}")

    if dims:
        width_overflow = max(0, dims.get("scrollWidth", 0) - dims.get("width", 0) - 1)
        height_overflow = max(0, dims.get("scrollHeight", 0) - dims.get("height", 0) - 1)
        if width_overflow > 0 or height_overflow > 0:
            directions = []
            if width_overflow > 0:
                directions.append(f"{width_overflow:.0f}px horizontally")
            if height_overflow > 0:
                directions.append(f"{height_overflow:.0f}px vertically")
            return (
                "Issues found:\n"
                f"- Content overflows the slide body by {' and '.join(directions)}. "
                "This is clipped by `overflow:hidden` so it looks fine in a screenshot, "
                "but the content is actually cut off / lost in the exported PPTX. "
                "Reduce font size, shorten text, or resize/reposition elements so everything "
                "fits within the fixed body bounds, then call inspect_slide again."
            )

    # overflow 없음 — heavy_reflect 모드면 방금 찍은 렌더링 이미지를 별도의 VLM 요청으로 보내
    # 겹침 여부만 검토받는다. 이 결과(텍스트)만 Design 에이전트에게 돌아가며, 이미지 자체는
    # Design 에이전트의 대화에 절대 실리지 않는다 — VLM 요청과 Design 에이전트의 요청은 완전히
    # 분리되어 있으므로, Design 에이전트가 어떤 모델을 쓰든(vision 지원 여부와 무관) 항상 동작한다.
    if HEAVY_REFLECT and img_bytes:
        llm = vlm_llm or _default_vlm_llm()
        b64 = base64.b64encode(img_bytes).decode()
        messages = [
            {"role": "system", "content": _VLM_OVERLAP_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VLM_OVERLAP_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ]
        response = await llm.run(messages=messages)
        review = (response.choices[0].message.content or "").strip()

        if "no overlap" in review.lower():
            return f"Slide is valid. ({len(content)} chars, aspect_ratio={aspect_ratio}) VLM overlap review: {review}"

        return (
            "Issues found (VLM overlap review):\n"
            f"- {review}\n"
            "Fix this only by resizing, repositioning, or reflowing the overlapping elements — "
            "never remove, shorten, or simplify text content to resolve it (always preserve the "
            "manuscript's content density). Then call inspect_slide again."
        )

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
    "inspect_content":    (INSPECT_CONTENT_SPEC,    inspect_content),
    "inspect_slide":      (INSPECT_SLIDE_SPEC,      inspect_slide),
}
