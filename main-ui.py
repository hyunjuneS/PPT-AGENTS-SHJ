import logging
import os
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from agents.llms import AsyncLLM

# .env 파일을 os.environ 에 주입. reload worker 재import 시에도 동일하게 적용된다.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPT Agent API", version="0.2.0")

# ---------------------------------------------------------------------------
# LLM — 환경변수에서 읽음 (.env 또는 시스템 환경변수)
# ---------------------------------------------------------------------------
_llm = AsyncLLM(
    model=os.environ.get("MODEL_NAME", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
)

# Design 에이전트 전용 모델 — DESIGN_MODEL_NAME이 없으면 기본 모델 사용
_design_llm = AsyncLLM(
    model=os.environ.get("DESIGN_MODEL_NAME") or os.environ.get("MODEL_NAME", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
)

# PPT_LANGUAGE env — 설정 시 모든 엔드포인트의 language form param을 덮어씀
# 값: "en" (영어) 또는 "ko" (한국어). 미설정 시 각 요청의 form param 사용.
_LANGUAGE_OVERRIDE: str | None = os.environ.get("PPT_LANGUAGE") or None

logger.info("LLM configured: research=%s  design=%s  language=%s",
            _llm, _design_llm, _LANGUAGE_OVERRIDE or "(from request)")


def _make_deep_config():
    """DeepPresenterConfig을 현재 _llm 설정으로 생성."""
    from deeppresenter.utils.config import DeepPresenterConfig, LLM

    def _to_deep_llm(llm: AsyncLLM) -> LLM:
        return LLM(model=llm.model, base_url=llm.base_url, api_key=llm.api_key)

    return DeepPresenterConfig(
        research_agent=_to_deep_llm(_llm),
        design_agent=_to_deep_llm(_design_llm),
        long_context_model=_to_deep_llm(_llm),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _llm.model}


@app.post("/research")
async def research(
    file: UploadFile = File(...),
    instruction: str = Form(...),
    language: str = Form(default="en"),
):
    """[DeepPresenter] .md 파일 + instruction → Research 에이전트로 슬라이드 원고 생성."""
    from deeppresenter.agents.env import AgentEnv
    from deeppresenter.agents.research import Research
    from deeppresenter.utils.constants import WORKSPACE_BASE
    from deeppresenter.utils.typings import InputRequest

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        md_content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    effective_language = _LANGUAGE_OVERRIDE or language

    # 세션별 workspace 생성
    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    # 업로드 파일 저장
    attachment_path = workspace / file.filename
    attachment_path.write_bytes(raw)

    req = InputRequest(
        instruction=instruction,
        attachments=[str(attachment_path)],
        language=effective_language,
    )

    logger.info("[Research] session=%s lang=%s instruction=%r",
                session_id, effective_language, instruction[:80])

    config = _make_deep_config()
    manuscript_path = None
    messages_log = []

    try:
        async with AgentEnv(workspace) as env:
            agent = Research(config=config, agent_env=env, workspace=workspace, language=effective_language)
            async for item in agent.loop(req):
                if isinstance(item, str):
                    manuscript_path = item
                    break
                else:
                    messages_log.append({"role": item.role, "text": item.text[:200]})
            agent.save_history()
    except Exception as e:
        logger.error("[Research] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Research agent failed: {e}")

    if manuscript_path is None:
        raise HTTPException(status_code=500, detail="Research agent did not produce a manuscript.")

    return FileResponse(
        path=manuscript_path,
        media_type="text/markdown",
        filename=Path(manuscript_path).name,
        headers={"X-Session-Id": session_id, "X-Turns": str(len(messages_log))},
    )


@app.post("/export")
async def export_pptx(
    slides_dir: str = Form(...),
    aspect_ratio: str = Form(default="16:9"),
    filename: str = Form(default="slides.pptx"),
    soft: bool = Form(default=True),
):
    """HTML 슬라이드 폴더(slides_dir) → PPTX 파일 변환 후 다운로드.
    soft=True(기본): 검증 경고는 로그로만 출력하고 PPTX 생성 계속.
    soft=False: 검증 오류 발생 시 변환 중단.
    """
    from deeppresenter.tools.export import html_slides_to_pptx

    slides_path = Path(slides_dir)
    if not slides_path.exists() or not slides_path.is_dir():
        raise HTTPException(status_code=400, detail=f"slides_dir not found: {slides_dir}")

    html_files = sorted(slides_path.glob("slide_*.html"))
    if not html_files:
        raise HTTPException(status_code=400, detail="No slide_*.html files found in slides_dir.")

    pptx_path = slides_path / filename
    logger.info("[Export] %d slides → %s (soft=%s)", len(html_files), pptx_path, soft)

    try:
        await html_slides_to_pptx(
            slides_dir=str(slides_path),
            output_path=str(pptx_path),
            aspect_ratio=aspect_ratio,
            soft=soft,
        )
    except Exception as e:
        logger.error("[Export] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    return FileResponse(
        path=str(pptx_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,
    )


@app.post("/design")
async def design(
    file: UploadFile = File(...),
    instruction: str = Form(default="Create a professional presentation."),
    language: str = Form(default="en"),
):
    """[DeepPresenter] 슬라이드 원고 .md → Design 에이전트 → HTML 슬라이드 생성."""
    from deeppresenter.agents.design import Design
    from deeppresenter.agents.env import AgentEnv
    from deeppresenter.utils.constants import WORKSPACE_BASE
    from deeppresenter.utils.typings import InputRequest

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        md_content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    effective_language = _LANGUAGE_OVERRIDE or language

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    manuscript_path = workspace / file.filename
    manuscript_path.write_bytes(raw)

    req = InputRequest(instruction=instruction, language=effective_language)

    template_content = ""
    tmpl_path = os.environ.get("DESIGN_TEMPLATE_FILE")
    if tmpl_path and Path(tmpl_path).exists():
        template_content = Path(tmpl_path).read_text(encoding="utf-8")

    config_file = os.environ.get("DESIGN_CONFIG_FILE") or None

    logger.info("[Design] session=%s lang=%s file=%s config=%s template=%s",
                session_id, effective_language, file.filename,
                Path(config_file).name if config_file else "Design.yaml",
                bool(template_content))

    config = _make_deep_config()
    slides_dir = None
    messages_log = []

    try:
        async with AgentEnv(workspace) as env:
            agent = Design(config=config, agent_env=env, workspace=workspace, language=effective_language,
                           config_file=config_file)
            async for item in agent.loop(req, markdown_file=str(manuscript_path), template_content=template_content):
                if isinstance(item, str):
                    slides_dir = item
                    break
                else:
                    messages_log.append({"role": item.role, "text": item.text[:200]})
            agent.save_history()
    except Exception as e:
        logger.error("[Design] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Design agent failed: {e}")

    if slides_dir is None:
        raise HTTPException(status_code=500, detail="Design agent did not produce a slides directory.")

    html_files = sorted(Path(slides_dir).glob("slide_*.html"))
    return JSONResponse(content={
        "session_id": session_id,
        "slides_dir": slides_dir,
        "slide_count": len(html_files),
        "slides": [str(f) for f in html_files],
        "turns": len(messages_log),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # 필수 환경변수 검증
    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is required (set in .env or environment)", file=sys.stderr)
        sys.exit(1)

    heavy_reflect = os.environ.get("DEEPPRESENTER_HEAVY_REFLECT", "").lower() in ("1", "true", "yes")
    if heavy_reflect and not os.environ.get("DESIGN_MODEL_NAME"):
        print("error: DESIGN_MODEL_NAME is required when DEEPPRESENTER_HEAVY_REFLECT is set", file=sys.stderr)
        sys.exit(1)

    # 경로 검증
    for env_key in ("DESIGN_CONFIG_FILE", "DESIGN_TEMPLATE_FILE"):
        val = os.environ.get(env_key)
        if val and not Path(val).exists():
            print(f"error: {env_key} not found: {val}", file=sys.stderr)
            sys.exit(1)

    host      = os.environ.get("HOST", "0.0.0.0")
    port      = int(os.environ.get("PORT", "5000"))
    reload    = os.environ.get("RELOAD", "true").lower() not in ("0", "false", "no")
    log_level = os.environ.get("LOG_LEVEL", "info")

    logger.info("LLM  : model=%s  vlm=%s  url=%s",
                os.environ.get("MODEL_NAME", "claude-opus-4-5"),
                os.environ.get("DESIGN_MODEL_NAME", "(none)"),
                os.environ.get("OPENAI_BASE_URL", "(none)"))
    logger.info("Server: host=%s port=%d reload=%s log_level=%s", host, port, reload, log_level)

    uvicorn.run(
        "main-ui:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
