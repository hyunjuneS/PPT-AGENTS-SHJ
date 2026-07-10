import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from agents.llms import AsyncLLM
from deeppresenter.utils.config import LLM

# .env 파일을 os.environ 에 주입. reload worker 재import 시에도 동일하게 적용된다.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPT Agent API", version="0.2.0")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """모든 API 응답에 처리 소요시간(초)을 X-Process-Time 헤더로 추가."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    logger.info("[Timing] %s %s took %.3fs", request.method, request.url.path, elapsed)
    return response

# ---------------------------------------------------------------------------
# LLM — 환경변수에서 읽음 (.env 또는 시스템 환경변수)
# ---------------------------------------------------------------------------
_llm = AsyncLLM(
    model=os.environ.get("MODEL_NAME", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
)

# Design 에이전트 전용 모델 (VLM) — DESIGN_MODEL_NAME이 없으면 기본 모델 사용.
# API 키는 VLM_API_KEY가 있으면 사용하고, 없으면 OPENAI_API_KEY로 폴백한다.
_design_llm = AsyncLLM(
    model=os.environ.get("DESIGN_MODEL_NAME") or os.environ.get("MODEL_NAME", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
    api_key=os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
    timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
)

# PPT_LANGUAGE env — 출력 언어 고정. 값: "en" (기본) 또는 "ko".
_LANGUAGE: str = os.environ.get("PPT_LANGUAGE", "en")

logger.info("LLM configured: research=%s  design=%s  language=%s",
            _llm, _design_llm, _LANGUAGE)


def _make_deep_config(research_llm=None, design_llm=None):
    """DeepPresenterConfig을 생성. research_llm/design_llm을 넘기면 해당 티어로
    선택된 LLM을 쓰고, 안 넘기면 기존처럼 정적 글로벌(_llm/_design_llm)에서 만든다."""
    from deeppresenter.utils.config import DeepPresenterConfig, LLM

    def _to_deep_llm(llm: AsyncLLM) -> LLM:
        return LLM(model=llm.model, base_url=llm.base_url, api_key=llm.api_key)

    r = research_llm or _to_deep_llm(_llm)
    d = design_llm or _to_deep_llm(_design_llm)
    return DeepPresenterConfig(
        research_agent=r,
        design_agent=d,
        long_context_model=r,
    )


# ---------------------------------------------------------------------------
# 모델 티어 (big/middle/small) — API 요청의 model_size 파라미터로 선택.
# .env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL을 서버 시작 시점에 한 번만 읽어서
# LLM 인스턴스로 만들어둔다 (요청마다 다시 만들지 않음).
# ---------------------------------------------------------------------------
_MODEL_TIER_ENV = {"big": "MODEL_BIG", "middle": "MODEL_MIDDLE", "small": "MODEL_SMALL"}


def _build_tier_llm(model_size: str) -> LLM | None:
    model_name = os.environ.get(_MODEL_TIER_ENV[model_size])
    if not model_name and model_size == "big":
        # MODEL_BIG 미설정 시 기존 동작(DESIGN_MODEL_NAME → MODEL_NAME)으로 폴백
        model_name = os.environ.get("DESIGN_MODEL_NAME") or os.environ.get("MODEL_NAME", "claude-opus-4-5")
    if not model_name:
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if model_size == "big":
        api_key = os.environ.get("OPENAI_API_KEY_BIG") or api_key

    return LLM(model=model_name, base_url=os.environ.get("OPENAI_BASE_URL") or None, api_key=api_key)


_TIER_LLMS: dict[str, LLM | None] = {size: _build_tier_llm(size) for size in _MODEL_TIER_ENV}

logger.info(
    "Model tiers configured: big=%s middle=%s small=%s",
    _TIER_LLMS["big"].model if _TIER_LLMS["big"] else None,
    _TIER_LLMS["middle"].model if _TIER_LLMS["middle"] else None,
    _TIER_LLMS["small"].model if _TIER_LLMS["small"] else None,
)


def _parse_provider_request_body(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider_request_body JSON: {e}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="provider_request_body must be a JSON object")
    return parsed


def _resolve_tiered_llm(model_size: str, provider_request_body: str | None) -> LLM:
    """시작 시점에 만들어둔 티어별 LLM(_TIER_LLMS)에 provider_request_body를 병합해 반환."""
    base = _TIER_LLMS[model_size]
    if base is None:
        raise HTTPException(
            status_code=400,
            detail=f"{_MODEL_TIER_ENV[model_size]} is not configured in .env",
        )
    params = _parse_provider_request_body(provider_request_body)
    if not params:
        return base
    return base.model_copy(update={"sampling_parameters": {**base.sampling_parameters, **params}})


async def _design_response(result, session_id: str, export: bool, export_filename: str):
    """Shared response-building for the two Design endpoints.

    When export=True, converts the generated slides to PPTX in the same
    request (same replica) instead of requiring a separate /export call —
    with multiple replicas behind a load balancer and no shared storage,
    that follow-up call can land on a different replica than the one that
    generated slides_dir and fail with "slides_dir not found" even though
    the files genuinely exist, just on another replica's local disk.
    Returns the PPTX directly; a FileResponse can't also carry a JSON body,
    so slide metadata goes in headers instead (same pattern /research
    already uses for its FileResponse).
    """
    from deeppresenter.tools.export import html_slides_to_pptx

    slides_dir = result.slides_dir
    html_files = sorted(Path(slides_dir).glob("slide_*.html"))

    if not export:
        return JSONResponse(content={
            "session_id": session_id,
            "slides_dir": slides_dir,
            "slide_count": len(html_files),
            "slides": [str(f) for f in html_files],
            "turns": len(result.messages_log),
        })

    pptx_path = Path(slides_dir) / export_filename
    try:
        await html_slides_to_pptx(
            slides_dir=slides_dir,
            output_path=str(pptx_path),
            aspect_ratio="16:9",
            soft=True,
        )
    except Exception as e:
        logger.error("[Design] export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    return FileResponse(
        path=str(pptx_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=export_filename,
        headers={
            "X-Session-Id": session_id,
            "X-Slides-Dir": slides_dir,
            "X-Slide-Count": str(len(html_files)),
            "X-Turns": str(len(result.messages_log)),
        },
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
    num_pages: int = Form(default=10, description="총 슬라이드 수 (표지 + 마지막 장 포함, auto=false일 때만 사용)"),
    auto: bool = Form(default=True, description="기본값 true. 문서 내용을 분석해 자동으로 슬라이드 수를 결정. false로 지정하면 num_pages 값을 그대로 사용"),
    model_size: Literal["big", "middle", "small"] = Form(default="big", description="사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    provider_request_body: str | None = Form(default=None, description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """.md 파일 + instruction → Research 에이전트(LangGraph 엔진)로 슬라이드 원고 생성."""
    from deeppresenter.agents.page_planner import decide_num_pages
    from deeppresenter.graph.callbacks import get_langfuse_handler
    from deeppresenter.graph.research_graph import run_research_graph
    from deeppresenter.utils.constants import WORKSPACE_BASE
    from deeppresenter.utils.typings import InputRequest

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        md_content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    tiered_llm = _resolve_tiered_llm(model_size, provider_request_body)
    config = _make_deep_config(research_llm=tiered_llm)

    if auto:
        resolved_num_pages = await decide_num_pages(config.long_context_model, md_content, instruction)
    else:
        resolved_num_pages = num_pages

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
        num_pages=str(resolved_num_pages),
        language=_LANGUAGE,
    )

    logger.info(
        "[Research] session=%s lang=%s auto=%s num_pages_input=%d resolved_num_pages=%d instruction=%r",
        session_id, _LANGUAGE, auto, num_pages, resolved_num_pages, instruction[:80],
    )

    try:
        result = await run_research_graph(
            config=config,
            workspace=workspace,
            req=req,
            language=_LANGUAGE,
            langfuse_handler=get_langfuse_handler(session_id),
            session_id=session_id,
        )
    except Exception as e:
        logger.error("[Research] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Research agent failed: {e}")

    manuscript_path = result.manuscript_path
    messages_log = result.messages_log

    return FileResponse(
        path=manuscript_path,
        media_type="text/markdown",
        filename=Path(manuscript_path).name,
        headers={
            "X-Session-Id": session_id,
            "X-Turns": str(len(messages_log)),
            "X-Num-Pages": str(resolved_num_pages),
        },
    )


@app.post("/export")
async def export_pptx(
    slides_dir: str = Form(...),
    filename: str = Form(default="slides.pptx"),
    soft: bool = Form(default=True),
):
    """HTML 슬라이드 폴더(slides_dir) → PPTX 파일 변환 후 다운로드 (16:9 고정).
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
            aspect_ratio="16:9",
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


@app.post("/design-hynix-template")
async def design_hynix_template(
    file: UploadFile = File(...),
    instruction: str = Form(default="Create a professional presentation."),
    export: bool = Form(default=True, description="true(기본)면 슬라이드 생성 직후 같은 요청 안에서 PPTX로 변환해 바로 반환 (레플리카가 여러 대일 때 별도 /export 호출이 다른 레플리카로 라우팅되는 문제 회피)"),
    export_filename: str = Form(default="slides.pptx"),
    model_size: Literal["big", "middle", "small"] = Form(default="big", description="사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    provider_request_body: str | None = Form(default=None, description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """슬라이드 원고 .md → Design 에이전트(LangGraph 엔진) → HTML 슬라이드 생성."""
    from deeppresenter.graph.callbacks import get_langfuse_handler
    from deeppresenter.graph.design_graph import run_design_graph
    from deeppresenter.utils.constants import WORKSPACE_BASE
    from deeppresenter.utils.typings import InputRequest

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        md_content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    manuscript_path = workspace / file.filename
    manuscript_path.write_bytes(raw)

    req = InputRequest(instruction=instruction, language=_LANGUAGE)

    template_content = ""
    tmpl_path = os.environ.get("DESIGN_TEMPLATE_FILE")
    if tmpl_path and Path(tmpl_path).exists():
        template_content = Path(tmpl_path).read_text(encoding="utf-8")

    config_file = os.environ.get("DESIGN_CONFIG_FILE") or None

    logger.info("[DesignHynixTemplate] session=%s lang=%s file=%s config=%s template=%s",
                session_id, _LANGUAGE, file.filename,
                Path(config_file).name if config_file else "Design.yaml",
                bool(template_content))

    tiered_llm = _resolve_tiered_llm(model_size, provider_request_body)
    config = _make_deep_config(design_llm=tiered_llm)

    try:
        result = await run_design_graph(
            config=config,
            workspace=workspace,
            req=req,
            markdown_file=str(manuscript_path),
            template_content=template_content,
            config_file=config_file,
            language=_LANGUAGE,
            langfuse_handler=get_langfuse_handler(session_id),
            session_id=session_id,
        )
    except Exception as e:
        logger.error("[DesignHynixTemplate] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Design agent failed: {e}")

    return await _design_response(result, session_id, export, export_filename)


@app.post("/design-free-template")
async def design_free_template(
    file: UploadFile = File(...),
    instruction: str = Form(default="Create a professional presentation."),
    export: bool = Form(default=True, description="true(기본)면 슬라이드 생성 직후 같은 요청 안에서 PPTX로 변환해 바로 반환 (레플리카가 여러 대일 때 별도 /export 호출이 다른 레플리카로 라우팅되는 문제 회피)"),
    export_filename: str = Form(default="slides.pptx"),
    model_size: Literal["big", "middle", "small"] = Form(default="big", description="사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    provider_request_body: str | None = Form(default=None, description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """슬라이드 원고 .md → Design 에이전트(LangGraph 엔진) → HTML 슬라이드 생성.
    템플릿 디렉토리 없이 Design 에이전트가 자유롭게 레이아웃을 설계한다.
    DESIGN_CONFIG_FILE env를 무시하고 항상 DesignFreeTemplate.yaml을 사용한다.
    """
    from deeppresenter.graph.callbacks import get_langfuse_handler
    from deeppresenter.graph.design_graph import run_design_graph
    from deeppresenter.utils.constants import PACKAGE_DIR, WORKSPACE_BASE
    from deeppresenter.utils.typings import InputRequest

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        md_content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    manuscript_path = workspace / file.filename
    manuscript_path.write_bytes(raw)

    req = InputRequest(instruction=instruction, language=_LANGUAGE)

    config_file = PACKAGE_DIR / "roles" / "DesignFreeTemplate.yaml"

    logger.info("[DesignFreeTemplate] session=%s lang=%s file=%s config=%s",
                session_id, _LANGUAGE, file.filename, config_file.name)

    tiered_llm = _resolve_tiered_llm(model_size, provider_request_body)
    config = _make_deep_config(design_llm=tiered_llm)

    try:
        result = await run_design_graph(
            config=config,
            workspace=workspace,
            req=req,
            markdown_file=str(manuscript_path),
            config_file=config_file,
            language=_LANGUAGE,
            langfuse_handler=get_langfuse_handler(session_id),
            session_id=session_id,
        )
    except Exception as e:
        logger.error("[DesignFreeTemplate] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Design agent failed: {e}")

    return await _design_response(result, session_id, export, export_filename)


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
