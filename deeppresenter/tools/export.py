"""HTML slides → PPTX via Node.js (Playwright screenshot + PptxGenJS)."""

import asyncio
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "html2pptx"
_CLI_JS = _SCRIPT_DIR / "html2pptx_cli.js"

# 기본 Chromium 경로 — 환경변수 PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH 로 덮어쓸 수 있음
_DEFAULT_CHROMIUM = Path(
    "/mnt/c/Users/X0160146/Desktop/26/playwright/chromium-1223/chrome-linux64/chrome"
)


def _get_chromium_executable() -> str | None:
    """Return Chromium executable path (env var > default path)."""
    # 1. 환경변수 우선
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. 기본 경로
    if _DEFAULT_CHROMIUM.exists():
        return str(_DEFAULT_CHROMIUM)

    return None


async def html_slides_to_pptx(
    slides_dir: str,
    output_path: str,
    aspect_ratio: str = "16:9",
) -> str:
    """
    Convert slide_*.html files in slides_dir to a PPTX file using Node.js.
    Returns the output path.
    """
    if not _CLI_JS.exists():
        raise FileNotFoundError(f"html2pptx_cli.js not found at {_CLI_JS}")

    slides_path = Path(slides_dir)
    html_files = sorted(slides_path.glob("slide_*.html"))
    if not html_files:
        raise ValueError(f"No slide_*.html files found in {slides_dir}")

    env = os.environ.copy()
    chromium_exe = _get_chromium_executable()
    if chromium_exe:
        env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = chromium_exe

    cmd = [
        "node", str(_CLI_JS),
        "--html_dir", str(slides_path.resolve()),
        "--output",   str(Path(output_path).resolve()),
        "--layout",   aspect_ratio,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_SCRIPT_DIR),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("html2pptx Node.js process timed out (5min)")

    log = stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"html2pptx failed (exit {proc.returncode}):\n{log}")

    return output_path
