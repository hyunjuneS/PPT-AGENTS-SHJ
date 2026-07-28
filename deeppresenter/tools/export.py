"""HTML slides → PPTX via Node.js (Playwright screenshot + PptxGenJS)."""

import asyncio
import html as html_escape
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "html2pptx"
_CLI_JS = _SCRIPT_DIR / "html2pptx_cli.js"

# _DEFAULT_CHROMIUM = Path(
#     "/mnt/c/Users/X0160146/Desktop/26/playwright/chromium-1223/chrome-linux64/chrome"
# )


def _get_chromium_executable() -> str | None:
    """Return Chromium executable path from env var, or None to use Playwright's installed Chromium."""
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    # if _DEFAULT_CHROMIUM.exists():
    #     return str(_DEFAULT_CHROMIUM)
    return None


async def html_slides_to_pptx(
    slides_dir: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    soft: bool = True,
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
    if soft:
        cmd.append("--soft")

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


def combine_html_slides(
    slides_dir: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
) -> str:
    """slides_dir의 slide_*.html N개(+ global.css)를 세로로 스크롤 가능한 하나의
    self-contained HTML로 합쳐 output_path에 저장하고 그 경로를 반환한다.

    각 슬라이드는 원본 문서 전체(글로벌 css를 <style>로 인라인해 넣은 버전)를
    <iframe srcdoc="...">로 통째로 embed한다 — 슬라이드마다 독립된 문서로 렌더링되므로
    슬라이드 간 CSS 선택자/id 충돌 없이 원본 그대로의 모습을 유지한다.
    """
    slides_path = Path(slides_dir)
    slide_files = sorted(slides_path.glob("slide_*.html"))
    if not slide_files:
        raise ValueError(f"No slide_*.html files found in {slides_dir}")

    css_path = slides_path / "global.css"
    global_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    # 브라우저 기본 UA 스타일은 <body>에 margin(보통 8px)을 넣는데, 슬라이드 자체 CSS가 이걸
    # 0으로 재설정하지 않은 경우 실제 렌더링 크기가 iframe의 고정 뷰포트(width x height)보다
    # 살짝 커져서 그 iframe 안에서만 스크롤이 생긴다. 원본 slide_NN.html은 건드리지 않고
    # 합쳐진 문서에서만 명시적으로 0으로 강제해 이 여유분을 없앤다.
    reset_css = "html,body{margin:0 !important;padding:0 !important;}"
    style_tag = f"<style>{reset_css}{global_css}</style>"

    frames = []
    for slide_file in slide_files:
        content = slide_file.read_text(encoding="utf-8")
        if "</head>" in content:
            content = content.replace("</head>", f"{style_tag}</head>", 1)
        else:
            content = style_tag + content
        escaped = html_escape.escape(content, quote=True)
        frames.append(
            f'<iframe class="combined-slide" scrolling="no" srcdoc="{escaped}" '
            f'style="width:{width}px;height:{height}px;border:0;display:block;overflow:hidden;'
            f'margin:0 auto 24px auto;box-shadow:0 2px 10px rgba(0,0,0,.15);"></iframe>'
        )

    combined_html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>Combined Slides</title>\n"
        "<style>html,body{margin:0;padding:0;background:#e5e5e5;}</style>\n"
        "</head>\n<body>\n" + "\n".join(frames) + "\n</body>\n</html>\n"
    )
    Path(output_path).write_text(combined_html, encoding="utf-8")
    return output_path
