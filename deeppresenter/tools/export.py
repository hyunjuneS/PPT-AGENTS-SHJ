"""HTML slides → PPTX via Node.js (Playwright screenshot + PptxGenJS)."""

import asyncio
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "html2pptx"
_CLI_JS = _SCRIPT_DIR / "html2pptx_cli.js"
_CHROMIUM_ZIP = Path(__file__).resolve().parents[3] / "bin" / "chromium-linux64.zip"
_CHROMIUM_DIR = Path(__file__).resolve().parents[3] / "bin" / "chromium"


def _get_chromium_executable() -> str | None:
    """Return path to local Chromium binary, extracting zip on first call."""
    if not _CHROMIUM_ZIP.exists():
        return None

    if not _CHROMIUM_DIR.exists():
        _extract_chromium()

    return _find_binary(_CHROMIUM_DIR)


def _extract_chromium() -> None:
    import stat
    import zipfile

    _CHROMIUM_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_CHROMIUM_ZIP, "r") as zf:
        zf.extractall(_CHROMIUM_DIR)

    for f in _CHROMIUM_DIR.rglob("*"):
        if f.is_file() and not f.suffix:
            f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _find_binary(base: Path) -> str | None:
    import os as _os
    for name in ("chrome", "chromium", "chromium-browser"):
        for candidate in base.rglob(name):
            if candidate.is_file() and _os.access(candidate, _os.X_OK):
                return str(candidate)
    return None


def _check_node_modules() -> None:
    node_modules = _SCRIPT_DIR / "node_modules"
    if not node_modules.exists():
        raise RuntimeError(
            f"Node.js dependencies not installed. Run:\n"
            f"  cd {_SCRIPT_DIR} && npm install\n"
            f"Or for offline: bash {_SCRIPT_DIR}/install_offline.sh"
        )


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

    _check_node_modules()

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
