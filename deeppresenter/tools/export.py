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
    """slides_dir의 slide_*.html N개를 세로로 스크롤 가능한 하나의 HTML로 합쳐
    output_path에 저장하고 그 경로를 반환한다.

    각 슬라이드는 원본 문서를 그대로 <iframe srcdoc="...">로 통째로 embed한다 — 슬라이드마다
    독립된 문서로 렌더링되므로 슬라이드 간 CSS 선택자/id 충돌이 없고, 슬라이드 자신의
    <link rel="stylesheet" href="global.css">(있다면)도 combined.html과 같은 폴더를 기준으로
    상대경로가 정상적으로 풀려서 그대로 적용된다. global.css를 이 함수가 별도로 다시 주입하지
    않는다 — 예전엔 안전하게 하려고 매 슬라이드에 global.css 전체를 <style>로 재주입했었는데,
    그러면 슬라이드 자신의 스타일보다 문서상 나중에 들어가버려서(예: global.css의
    `.slide{background:#fff}` 같은 축약 속성이 슬라이드 자체의 `.slide{background-image:...}`를
    뒤엎어버림) 오히려 슬라이드 고유 스타일을 깨뜨리는 버그가 됐다.

    로컬 이미지 참조(배경 이미지 url(), <img src>)도 원본 그대로 상대경로를 유지한다 —
    srcdoc 안의 상대경로는 이 함수가 만든 output_path(=combined.html)의 위치를 기준으로
    풀리므로, combined.html과 그 이미지 파일들이 항상 같은 폴더에 함께 있어야 렌더링된다
    (호출부가 이미지들도 combined.html과 같이 묶어서 배포/업로드해야 함).
    """
    slides_path = Path(slides_dir)
    slide_files = sorted(slides_path.glob("slide_*.html"))
    if not slide_files:
        raise ValueError(f"No slide_*.html files found in {slides_dir}")

    # 브라우저 기본 UA 스타일은 <body>에 margin(보통 8px)을 넣는데, 슬라이드 자체 CSS가 이걸
    # 0으로 재설정하지 않은 경우 실제 렌더링 크기가 iframe의 고정 뷰포트(width x height)보다
    # 살짝 커져서 그 iframe 안에서만 스크롤이 생긴다. 원본 slide_NN.html은 건드리지 않고
    # 합쳐진 문서에서만 명시적으로 0으로 강제해 이 여유분을 없앤다. !important라서 슬라이드
    # 자신의 스타일과 순서/충돌 걱정 없이 항상 이긴다.
    style_tag = "<style>html,body{margin:0 !important;padding:0 !important;}</style>"

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
