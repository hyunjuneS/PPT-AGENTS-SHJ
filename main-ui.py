import json
import logging
import os
import re
import secrets
import string
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from agents.llms import AsyncLLM
from deeppresenter.utils.config import LLM

# .env 파일을 os.environ 에 주입. reload worker 재import 시에도 동일하게 적용된다.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# openai SDK가 내부적으로 쓰는 httpx의 요청/응답 로그를 켜둔다.
# "openai._base_client: Retrying request..." 메시지만으로는 재시도 원인(429 rate limit인지,
# 5xx인지, 커넥션 문제인지)을 알 수 없는데, httpx를 INFO로 올려두면 그 바로 옆에
# 'HTTP Request: POST .../chat/completions "HTTP/1.1 429 Too Many Requests"' 식으로
# 실제 상태 코드가 같이 찍혀서 원인을 바로 확인할 수 있다.
logging.getLogger("httpx").setLevel(logging.INFO)

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

# OPENAI_BASE_URL 미설정 시 폴백하는 기본 엔드포인트. API 요청의 base_url 파라미터로도 덮어쓸 수 있다.
_DEFAULT_BASE_URL = "http://workplace-litellm.aipp02.skhynix.com/v1"

# Research/Design 단계에 별도 instruction을 받지 않는 엔드포인트에서 사용하는 고정 지시문.
_RESEARCH_DEFAULT_INSTRUCTION = "Create presentation content based on the attached document."
_DESIGN_DEFAULT_INSTRUCTION = "Create a professional presentation."

_llm = AsyncLLM(
    model=os.environ.get("MODEL_BIG", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
)

# Design 에이전트 전용 모델 (VLM) — DESIGN_MODEL_NAME이 없으면 기본 모델 사용.
# API 키는 VLM_API_KEY가 있으면 사용하고, 없으면 OPENAI_API_KEY로 폴백한다.
_design_llm = AsyncLLM(
    model=os.environ.get("DESIGN_MODEL_NAME") or os.environ.get("MODEL_BIG", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL,
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
        # MODEL_BIG 미설정 시 DESIGN_MODEL_NAME → 기본값으로 폴백
        model_name = os.environ.get("DESIGN_MODEL_NAME") or "claude-opus-4-5"
    if not model_name:
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if model_size == "big":
        api_key = os.environ.get("OPENAI_API_KEY_BIG") or api_key

    return LLM(model=model_name, base_url=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, api_key=api_key)


_TIER_LLMS: dict[str, LLM | None] = {size: _build_tier_llm(size) for size in _MODEL_TIER_ENV}

logger.info(
    "Model tiers configured: big=%s middle=%s small=%s",
    _TIER_LLMS["big"].model if _TIER_LLMS["big"] else None,
    _TIER_LLMS["middle"].model if _TIER_LLMS["middle"] else None,
    _TIER_LLMS["small"].model if _TIER_LLMS["small"] else None,
)


def _random_emp_no() -> str:
    """/export 요청에 emp_no가 없을 때 사용하는 임시 사번: test_{영숫자 5글자}."""
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(5))
    return f"test_{suffix}"


def _cover_info_block(presenter_name: str, emp_no: str, team_name: str) -> str:
    """Design 에이전트 instruction에 덧붙여 첫 페이지(커버)의 팀명/이름(사번)/날짜 자리를 채우는 지시문.
    날짜는 요청을 받은 시점의 날짜를 'YY.MM.DD' 형식(예: 26.07.20)으로 사용한다."""
    date_str = datetime.now().strftime("%y.%m.%d")
    return (
        "Cover slide (slide_01) already has placeholder fields for team name, "
        "name(employee no), and date. Replace ONLY those fields with the values below — "
        "keep every other element, structure, and style unchanged:\n"
        f"- Team name: {team_name}\n"
        f"- Name(Employee No): {presenter_name}({emp_no})\n"
        f"- Date: {date_str}"
    )


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")


def _split_ids(ids: list[str]) -> list[str]:
    """list[str]=Form(...)로 여러 필드를 제대로 보낸 경우와, Swagger UI 등 일부 클라이언트가
    'a,b' 처럼 한 필드에 콤마로 이어붙여 보내는 경우를 모두 지원하도록 각 원소를 콤마로 추가 분리."""
    result = []
    for raw in ids:
        result.extend(part.strip() for part in raw.split(",") if part.strip())
    return result


def _write_sources_as_markdown(workspace: Path, ids: list[str], sources: dict[str, dict]) -> list[Path]:
    """DB에서 조회한 id별 title/raw_text를 'sources/{id}.md' 로 저장하고 경로 목록을 반환.
    파일 맨 앞에 식별 헤더(id, title)를 삽입해, Research 에이전트가 절대경로가 아니라
    본문 내부 라벨만으로도 정확한 출처를 인용할 수 있게 한다."""
    for source_id in ids:
        if not _SAFE_ID_RE.match(source_id):
            raise HTTPException(status_code=400, detail=f"Invalid id (unsafe filename): {source_id}")

    sources_dir = workspace / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for source_id in ids:
        title = sources[source_id]["title"] or "(untitled)"
        header = f"<!-- SOURCE id={source_id} title={title} -->\n# {title}\n\n"
        path = sources_dir / f"{source_id}.md"
        path.write_text(header + sources[source_id]["raw_text"], encoding="utf-8")
        paths.append(path)
    return paths


def _parse_additional_request(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid additional_request JSON: {e}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="additional_request must be a JSON object")
    return parsed


def _resolve_tiered_llm(model_size: str, additional_request: str | None, base_url: str | None = None) -> LLM:
    """시작 시점에 만들어둔 티어별 LLM(_TIER_LLMS)에 additional_request/base_url을 병합해 반환."""
    base = _TIER_LLMS[model_size]
    if base is None:
        raise HTTPException(
            status_code=400,
            detail=f"{_MODEL_TIER_ENV[model_size]} is not configured in .env",
        )
    params = _parse_additional_request(additional_request)
    updates = {}
    if params:
        updates["sampling_parameters"] = {**base.sampling_parameters, **params}
    if base_url:
        updates["base_url"] = base_url
    if not updates:
        return base
    return base.model_copy(update=updates)


async def _design_response(result, session_id: str, export_filename: str, emp_no: str):
    """Shared response-building for the Design endpoints.

    Converts the generated slides to PPTX in the same request (same replica)
    instead of requiring a separate /export call — with multiple replicas
    behind a load balancer and no shared storage, that follow-up call can
    land on a different replica than the one that generated slides_dir and
    fail with "slides_dir not found" even though the files genuinely exist,
    just on another replica's local disk.

    Uploads three artifacts to MinIO, all under "{emp_no}/{export_filename stem}/":
    - the PPTX at ".../ppt/{export_filename stem}.pptx"
    - every slide_*.html + global.css + any local image (e.g. the hynix cover logo) individually
      at ".../htmls/..."
    - the scrollable combined HTML + global.css + those same local images at ".../combined_html/...".
      Each slide inside combined.html still references global.css via its own <link>, so global.css
      must ship alongside it too, or every slide renders unstyled.

    Before any of that, injects a small JS/SVG chart-rendering script into any slide_*.html that
    has a data-chart-type element, so the chart is actually visible when viewing the html/combined
    html directly in a browser (html2pptx.js only ever turned it into a native PPTX chart — the
    div itself was otherwise empty in plain HTML).
    """
    from deeppresenter.tools.export import combine_html_slides, html_slides_to_pptx, inject_chart_rendering
    from deeppresenter.tools.storage import upload_combined_html, upload_html_files, upload_pptx

    slides_dir = result.slides_dir
    inject_chart_rendering(slides_dir)
    html_files = sorted(Path(slides_dir).glob("slide_*.html"))

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

    try:
        object_name = upload_pptx(str(pptx_path), emp_no, export_filename)
    except Exception as e:
        logger.error("[Design] MinIO upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"MinIO upload failed: {e}")

    css_path = Path(slides_dir) / "global.css"
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
    image_files = sorted(
        p for p in Path(slides_dir).iterdir() if p.is_file() and p.suffix.lower() in image_extensions
    )
    html_bundle_files = [str(p) for p in html_files] + [str(p) for p in image_files]
    if css_path.exists():
        html_bundle_files.append(str(css_path))
    try:
        htmls_object_names = upload_html_files(html_bundle_files, emp_no, export_filename)
    except Exception as e:
        logger.error("[Design] MinIO htmls upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"MinIO htmls upload failed: {e}")

    combined_path = Path(slides_dir) / "combined.html"
    try:
        combine_html_slides(slides_dir, str(combined_path))
        combined_bundle_files = [str(combined_path)] + [str(p) for p in image_files]
        if css_path.exists():
            combined_bundle_files.append(str(css_path))
        combined_object_names = upload_combined_html(combined_bundle_files, emp_no, export_filename)
    except Exception as e:
        logger.error("[Design] MinIO combined html upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"MinIO combined html upload failed: {e}")

    return FileResponse(
        path=str(pptx_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=export_filename,
        headers={
            "X-Session-Id": session_id,
            "X-Slides-Dir": slides_dir,
            "X-Slide-Count": str(len(html_files)),
            "X-Turns": str(len(result.messages_log)),
            "X-Minio-Object": object_name,
            "X-Minio-Htmls-Count": str(len(htmls_object_names)),
            "X-Minio-Combined-Html-Count": str(len(combined_object_names)),
        },
    )


# ---------------------------------------------------------------------------
# Stage runners — Research/Design 각 단계의 실제 실행 로직. 단독 엔드포인트(/research,
# /design-hynix-template, /design-free-template)와 Research+Design을 이어 붙인 결합
# 엔드포인트(/template-based-ppt-generation, /template-free-ppt-generation)가 공유한다.
# ---------------------------------------------------------------------------

async def _run_research_stage_from_paths(config, workspace, session_id: str, md_paths: list[Path],
                                          num_pages: int, auto_page: bool,
                                          instruction: str = _RESEARCH_DEFAULT_INSTRUCTION,
                                          config_file: str | Path | None = None):
    """이미 workspace에 저장된 .md 파일 목록 → 슬라이드 원고(ResearchGraphResult) 생성.
    /research, /template-*-ppt-generation(-db) 이 모두 이 헬퍼를 거쳐간다.
    instruction/config_file을 넘기지 않으면 기존 기본 동작(Research.yaml, 고정 지시문) 그대로다."""
    from deeppresenter.agents.page_planner import decide_num_pages
    from deeppresenter.graph.callbacks import get_langfuse_handler
    from deeppresenter.graph.research_graph import run_research_graph
    from deeppresenter.utils.typings import InputRequest

    if auto_page:
        combined_content = "\n\n".join(p.read_text(encoding="utf-8") for p in md_paths)
        resolved_num_pages = await decide_num_pages(config.long_context_model, combined_content, _RESEARCH_DEFAULT_INSTRUCTION)
    else:
        resolved_num_pages = num_pages

    req = InputRequest(
        instruction=instruction,
        attachments=[str(p) for p in md_paths],
        num_pages=str(resolved_num_pages),
        language=_LANGUAGE,
    )

    logger.info(
        "[Research] session=%s lang=%s auto_page=%s num_pages_input=%d resolved_num_pages=%d attachments=%d",
        session_id, _LANGUAGE, auto_page, num_pages, resolved_num_pages, len(md_paths),
    )

    try:
        result = await run_research_graph(
            config=config,
            workspace=workspace,
            req=req,
            language=_LANGUAGE,
            langfuse_handler=get_langfuse_handler(session_id),
            session_id=session_id,
            config_file=config_file,
        )
    except Exception as e:
        logger.error("[Research] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Research agent failed: {e}")

    return result, resolved_num_pages


async def _run_research_stage(config, workspace, session_id: str, file: UploadFile,
                               num_pages: int, auto_page: bool):
    """업로드된 .md → 슬라이드 원고(ResearchGraphResult) 생성."""
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    attachment_path = workspace / file.filename
    attachment_path.write_bytes(raw)

    return await _run_research_stage_from_paths(config, workspace, session_id, [attachment_path], num_pages, auto_page)


async def _run_design_hynix_stage(config, workspace, session_id: str, markdown_file: str, instruction: str):
    """원고(.md) → Design 에이전트(Hynix 템플릿)로 HTML 슬라이드 생성."""
    from deeppresenter.graph.callbacks import get_langfuse_handler
    from deeppresenter.graph.design_graph import run_design_graph
    from deeppresenter.utils.typings import InputRequest

    req = InputRequest(instruction=instruction, language=_LANGUAGE)

    template_content = ""
    tmpl_path = os.environ.get("DESIGN_TEMPLATE_FILE")
    if tmpl_path and Path(tmpl_path).exists():
        template_content = Path(tmpl_path).read_text(encoding="utf-8")

    config_file = os.environ.get("DESIGN_CONFIG_FILE") or None

    logger.info("[DesignHynixTemplate] session=%s lang=%s file=%s config=%s template=%s",
                session_id, _LANGUAGE, Path(markdown_file).name,
                Path(config_file).name if config_file else "Design.yaml",
                bool(template_content))

    try:
        return await run_design_graph(
            config=config,
            workspace=workspace,
            req=req,
            markdown_file=markdown_file,
            template_content=template_content,
            config_file=config_file,
            language=_LANGUAGE,
            langfuse_handler=get_langfuse_handler(session_id),
            session_id=session_id,
        )
    except Exception as e:
        logger.error("[DesignHynixTemplate] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Design agent failed: {e}")


async def _run_design_free_stage(config, workspace, session_id: str, markdown_file: str, instruction: str):
    """원고(.md) → Design 에이전트(자유 템플릿)로 HTML 슬라이드 생성."""
    from deeppresenter.graph.callbacks import get_langfuse_handler
    from deeppresenter.graph.design_graph import run_design_graph
    from deeppresenter.utils.constants import PACKAGE_DIR
    from deeppresenter.utils.typings import InputRequest

    req = InputRequest(instruction=instruction, language=_LANGUAGE)

    config_file = PACKAGE_DIR / "roles" / "DesignFreeTemplate.yaml"

    logger.info("[DesignFreeTemplate] session=%s lang=%s file=%s config=%s",
                session_id, _LANGUAGE, Path(markdown_file).name, config_file.name)

    try:
        return await run_design_graph(
            config=config,
            workspace=workspace,
            req=req,
            markdown_file=markdown_file,
            config_file=config_file,
            language=_LANGUAGE,
            langfuse_handler=get_langfuse_handler(session_id),
            session_id=session_id,
        )
    except Exception as e:
        logger.error("[DesignFreeTemplate] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Design agent failed: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _llm.model}


@app.get("/api/admin/appReady")
def app_ready():
    return {"status": "ok"}


@app.post("/research", tags=["dev"])
async def research(
    file: UploadFile = File(...),
    num_pages: int = Form(default=10, description="총 슬라이드 수 (표지 + 마지막 장 포함, auto_page=false일 때만 사용)"),
    auto_page: bool = Form(default=True, description="기본값 true. 문서 내용을 분석해 자동으로 슬라이드 수를 결정. false로 지정하면 num_pages 값을 그대로 사용"),
    model_size: Literal["big", "middle", "small"] = Form(default="big", description="사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    base_url: str | None = Form(default=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, description="OpenAI 호환 API 엔드포인트. 비워서 보내면 해당 티어의 기본 엔드포인트를 그대로 사용"),
    additional_request: str | None = Form(default="{}", description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """.md 파일 → Research 에이전트(LangGraph 엔진)로 슬라이드 원고 생성."""
    from deeppresenter.utils.constants import WORKSPACE_BASE

    tiered_llm = _resolve_tiered_llm(model_size, additional_request, base_url)
    config = _make_deep_config(research_llm=tiered_llm)

    # 세션별 workspace 생성
    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    result, resolved_num_pages = await _run_research_stage(
        config, workspace, session_id, file, num_pages, auto_page,
    )

    return FileResponse(
        path=result.manuscript_path,
        media_type="text/markdown",
        filename=Path(result.manuscript_path).name,
        headers={
            "X-Session-Id": session_id,
            "X-Turns": str(len(result.messages_log)),
            "X-Num-Pages": str(resolved_num_pages),
        },
    )


@app.post("/export", tags=["dev"])
async def export_pptx(
    slides_dir: str = Form(...),
    filename: str = Form(default="slides.pptx"),
    soft: bool = Form(default=True),
    emp_no: str | None = Form(default=None, description="미입력 시 'test_{랜덤5글자}'로 자동 생성"),
):
    """HTML 슬라이드 폴더(slides_dir) → PPTX 파일 변환 후 다운로드 (16:9 고정).
    soft=True(기본): 검증 경고는 로그로만 출력하고 PPTX 생성 계속.
    soft=False: 검증 오류 발생 시 변환 중단.
    생성된 PPTX는 MinIO에도 "{emp_no}/{filename stem}/ppt/{filename stem}.pptx" 로 업로드된다.
    """
    from deeppresenter.tools.export import html_slides_to_pptx
    from deeppresenter.tools.storage import upload_pptx

    slides_path = Path(slides_dir)
    if not slides_path.exists() or not slides_path.is_dir():
        raise HTTPException(status_code=400, detail=f"slides_dir not found: {slides_dir}")

    html_files = sorted(slides_path.glob("slide_*.html"))
    if not html_files:
        raise HTTPException(status_code=400, detail="No slide_*.html files found in slides_dir.")

    resolved_emp_no = emp_no or _random_emp_no()
    pptx_path = slides_path / filename
    logger.info("[Export] %d slides → %s (soft=%s, emp_no=%s)", len(html_files), pptx_path, soft, resolved_emp_no)

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

    try:
        object_name = upload_pptx(str(pptx_path), resolved_emp_no, filename)
    except Exception as e:
        logger.error("[Export] MinIO upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"MinIO upload failed: {e}")

    return FileResponse(
        path=str(pptx_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,
        headers={
            "X-Emp-No": resolved_emp_no,
            "X-Minio-Object": object_name,
        },
    )


@app.post("/download", tags=["dev"])
async def download_pptx(
    emp_no: str = Form(...),
    export_filename: str = Form(..., description="MinIO에 저장된 파일명 (예: slides.pptx 또는 slides)"),
):
    """MinIO의 '{emp_no}/{export_filename stem}/ppt/{export_filename stem}.pptx' 오브젝트를 조회해 다운로드."""
    from starlette.background import BackgroundTask

    from deeppresenter.tools.storage import download_pptx as fetch_pptx

    try:
        local_path, object_name = fetch_pptx(emp_no, export_filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("[Download] MinIO download failed: %s", e)
        raise HTTPException(status_code=500, detail=f"MinIO download failed: {e}")

    return FileResponse(
        path=local_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=Path(object_name).name,
        headers={"X-Minio-Object": object_name},
        background=BackgroundTask(lambda: Path(local_path).unlink(missing_ok=True)),
    )


@app.post("/download-separated-html", tags=["dev"])
async def download_separated_html(
    emp_no: str = Form(...),
    export_filename: str = Form(..., description="MinIO에 저장된 파일명 (예: slides.pptx 또는 slides)"),
):
    """MinIO의 '{emp_no}/{export_filename stem}/htmls/' 아래 개별 슬라이드 html + css 파일을
    모두 모아 zip으로 묶어 다운로드."""
    from starlette.background import BackgroundTask

    from deeppresenter.tools.storage import download_html_files

    try:
        local_path, prefix = download_html_files(emp_no, export_filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("[DownloadSeparatedHtml] MinIO download failed: %s", e)
        raise HTTPException(status_code=500, detail=f"MinIO download failed: {e}")

    zip_filename = f"{Path(prefix.rstrip('/')).name}.zip"
    return FileResponse(
        path=local_path,
        media_type="application/zip",
        filename=zip_filename,
        headers={"X-Minio-Prefix": prefix},
        background=BackgroundTask(lambda: Path(local_path).unlink(missing_ok=True)),
    )


@app.post("/download-combined-html", tags=["dev"])
async def download_combined_html(
    emp_no: str = Form(...),
    export_filename: str = Form(..., description="MinIO에 저장된 파일명 (예: slides.pptx 또는 slides)"),
):
    """MinIO의 '{emp_no}/{export_filename stem}/combined_html/' 아래 combined.html + 그 로컬 이미지
    (예: 하이닉스 커버 로고)를 모두 모아 zip으로 묶어 다운로드 — combined.html이 이미지를 상대경로로
    참조하므로 이미지 없이 combined.html만 받으면 렌더링이 깨진다."""
    from starlette.background import BackgroundTask

    from deeppresenter.tools.storage import download_combined_html as fetch_combined_html

    try:
        local_path, prefix = fetch_combined_html(emp_no, export_filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("[DownloadCombinedHtml] MinIO download failed: %s", e)
        raise HTTPException(status_code=500, detail=f"MinIO download failed: {e}")

    zip_filename = f"{Path(prefix.rstrip('/')).name}.zip"
    return FileResponse(
        path=local_path,
        media_type="application/zip",
        filename=zip_filename,
        headers={"X-Minio-Prefix": prefix},
        background=BackgroundTask(lambda: Path(local_path).unlink(missing_ok=True)),
    )


@app.post("/design-hynix-template", tags=["dev"])
async def design_hynix_template(
    file: UploadFile = File(...),
    instruction: str = Form(default="Create a professional presentation."),
    export_filename: str = Form(default="slides.pptx"),
    emp_no: str = Form(..., description="MinIO 저장 경로 '{emp_no}/{export_filename}/ppt/{export_filename}.pptx'에 사용되는 사번. 커버 슬라이드에도 표시됨"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
    model_size: Literal["big", "middle", "small"] = Form(default="big", description="사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    base_url: str | None = Form(default=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, description="OpenAI 호환 API 엔드포인트. 비워서 보내면 해당 티어의 기본 엔드포인트를 그대로 사용"),
    additional_request: str | None = Form(default="{}", description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """슬라이드 원고 .md → Design 에이전트(LangGraph 엔진) → HTML 슬라이드 생성 → PPTX 반환."""
    from deeppresenter.utils.constants import WORKSPACE_BASE

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    manuscript_path = workspace / file.filename
    manuscript_path.write_bytes(raw)

    tiered_llm = _resolve_tiered_llm(model_size, additional_request, base_url)
    config = _make_deep_config(design_llm=tiered_llm)

    full_instruction = f"{instruction}\n\n{_cover_info_block(presenter_name, emp_no, team_name)}"
    result = await _run_design_hynix_stage(config, workspace, session_id, str(manuscript_path), full_instruction)

    return await _design_response(result, session_id, export_filename, emp_no)


@app.post("/design-free-template", tags=["dev"])
async def design_free_template(
    file: UploadFile = File(...),
    instruction: str = Form(default="Create a professional presentation."),
    export_filename: str = Form(default="slides.pptx"),
    emp_no: str = Form(..., description="MinIO 저장 경로 '{emp_no}/{export_filename}/ppt/{export_filename}.pptx'에 사용되는 사번. 커버 슬라이드에도 표시됨"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
    model_size: Literal["big", "middle", "small"] = Form(default="big", description="사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    base_url: str | None = Form(default=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, description="OpenAI 호환 API 엔드포인트. 비워서 보내면 해당 티어의 기본 엔드포인트를 그대로 사용"),
    additional_request: str | None = Form(default="{}", description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """슬라이드 원고 .md → Design 에이전트(LangGraph 엔진) → HTML 슬라이드 생성 → PPTX 반환.
    템플릿 디렉토리 없이 Design 에이전트가 자유롭게 레이아웃을 설계한다.
    DESIGN_CONFIG_FILE env를 무시하고 항상 DesignFreeTemplate.yaml을 사용한다.
    """
    from deeppresenter.utils.constants import WORKSPACE_BASE

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    manuscript_path = workspace / file.filename
    manuscript_path.write_bytes(raw)

    tiered_llm = _resolve_tiered_llm(model_size, additional_request, base_url)
    config = _make_deep_config(design_llm=tiered_llm)

    full_instruction = f"{instruction}\n\n{_cover_info_block(presenter_name, emp_no, team_name)}"
    result = await _run_design_free_stage(config, workspace, session_id, str(manuscript_path), full_instruction)

    return await _design_response(result, session_id, export_filename, emp_no)


@app.post("/template-based-ppt-generation", tags=["main"], summary="Template-Based-PPT-Generation")
async def template_based_ppt_generation(
    file: UploadFile = File(...),
    num_pages: int = Form(default=10, description="총 슬라이드 수 (표지 + 마지막 장 포함, auto_page=false일 때만 사용)"),
    auto_page: bool = Form(default=True, description="기본값 true. 문서 내용을 분석해 자동으로 슬라이드 수를 결정. false로 지정하면 num_pages 값을 그대로 사용"),
    export_filename: str = Form(default="slides.pptx"),
    emp_no: str = Form(..., description="MinIO 저장 경로 '{emp_no}/{export_filename}/ppt/{export_filename}.pptx'에 사용되는 사번. 커버 슬라이드에도 표시됨"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
    research_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Research 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    design_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Design 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    base_url: str | None = Form(default=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, description="OpenAI 호환 API 엔드포인트. 비워서 보내면 해당 티어의 기본 엔드포인트를 그대로 사용"),
    additional_request: str | None = Form(default="{}", description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """.md 파일 → Research 에이전트로 원고 생성 → Design(Hynix 템플릿) 에이전트로 슬라이드 생성,
    변환까지 한 요청에서 이어서 처리하고 PPTX를 반환한다."""
    from deeppresenter.utils.constants import WORKSPACE_BASE

    research_llm = _resolve_tiered_llm(research_model_size, additional_request, base_url)
    design_llm = _resolve_tiered_llm(design_model_size, additional_request, base_url)
    config = _make_deep_config(research_llm=research_llm, design_llm=design_llm)

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    research_result, _ = await _run_research_stage(config, workspace, session_id, file, num_pages, auto_page)
    design_instruction = f"{_DESIGN_DEFAULT_INSTRUCTION}\n\n{_cover_info_block(presenter_name, emp_no, team_name)}"
    design_result = await _run_design_hynix_stage(
        config, workspace, session_id, research_result.manuscript_path, design_instruction,
    )

    return await _design_response(design_result, session_id, export_filename, emp_no)


@app.post("/template-free-ppt-generation", tags=["main"], summary="Template-Free-PPT-Generation")
async def template_free_ppt_generation(
    file: UploadFile = File(...),
    num_pages: int = Form(default=10, description="총 슬라이드 수 (표지 + 마지막 장 포함, auto_page=false일 때만 사용)"),
    auto_page: bool = Form(default=True, description="기본값 true. 문서 내용을 분석해 자동으로 슬라이드 수를 결정. false로 지정하면 num_pages 값을 그대로 사용"),
    export_filename: str = Form(default="slides.pptx"),
    emp_no: str = Form(..., description="MinIO 저장 경로 '{emp_no}/{export_filename}/ppt/{export_filename}.pptx'에 사용되는 사번. 커버 슬라이드에도 표시됨"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
    research_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Research 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    design_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Design 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    base_url: str | None = Form(default=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, description="OpenAI 호환 API 엔드포인트. 비워서 보내면 해당 티어의 기본 엔드포인트를 그대로 사용"),
    additional_request: str | None = Form(default="{}", description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
):
    """.md 파일 → Research 에이전트로 원고 생성 → Design(자유 템플릿) 에이전트로 슬라이드 생성,
    변환까지 한 요청에서 이어서 처리하고 PPTX를 반환한다."""
    from deeppresenter.utils.constants import WORKSPACE_BASE

    research_llm = _resolve_tiered_llm(research_model_size, additional_request, base_url)
    design_llm = _resolve_tiered_llm(design_model_size, additional_request, base_url)
    config = _make_deep_config(research_llm=research_llm, design_llm=design_llm)

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    research_result, _ = await _run_research_stage(config, workspace, session_id, file, num_pages, auto_page)
    design_instruction = f"{_DESIGN_DEFAULT_INSTRUCTION}\n\n{_cover_info_block(presenter_name, emp_no, team_name)}"
    design_result = await _run_design_free_stage(
        config, workspace, session_id, research_result.manuscript_path, design_instruction,
    )

    return await _design_response(design_result, session_id, export_filename, emp_no)


@app.post("/template-based-ppt-generation-db", tags=["main"], summary="Template-Based-PPT-Generation-DB")
async def template_based_ppt_generation_db(
    ids: list[str] = Form(..., description="sources 테이블에서 raw_text를 조회할 id 목록 (여러 개 전달 가능)"),
    num_pages: int = Form(default=10, description="총 슬라이드 수 (표지 + 마지막 장 포함, auto_page=false일 때만 사용)"),
    auto_page: bool = Form(default=True, description="기본값 true. 문서 내용을 분석해 자동으로 슬라이드 수를 결정. false로 지정하면 num_pages 값을 그대로 사용"),
    export_filename: str = Form(default="slides.pptx"),
    emp_no: str = Form(..., description="MinIO 저장 경로 '{emp_no}/{export_filename}/ppt/{export_filename}.pptx'에 사용되는 사번. 커버 슬라이드에도 표시됨"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
    research_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Research 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    design_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Design 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    base_url: str | None = Form(default=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, description="OpenAI 호환 API 엔드포인트. 비워서 보내면 해당 티어의 기본 엔드포인트를 그대로 사용"),
    additional_request: str | None = Form(default="{}", description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
    additional_instruction: str | None = Form(default=None, description="Research 매뉴스크립트 작성 시 반영할 추가 지시사항 (예: 특정 내용을 강조해달라는 지시)"),
):
    """id 목록 → PostgreSQL sources 테이블에서 title/raw_text 조회 → '{id}.md' 파일로 저장 →
    Research 에이전트(research-db.yaml)로 여러 소스를 하나의 원고로 합성 →
    Design(Hynix 템플릿) 에이전트로 슬라이드 생성,
    변환까지 한 요청에서 이어서 처리하고 PPTX를 반환한다."""
    from deeppresenter.tools.db import fetch_raw_texts
    from deeppresenter.utils.constants import PACKAGE_DIR, WORKSPACE_BASE

    ids = _split_ids(ids)
    if not ids:
        raise HTTPException(status_code=400, detail="ids must contain at least one id.")

    research_llm = _resolve_tiered_llm(research_model_size, additional_request, base_url)
    design_llm = _resolve_tiered_llm(design_model_size, additional_request, base_url)
    config = _make_deep_config(research_llm=research_llm, design_llm=design_llm)

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        sources = fetch_raw_texts(ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("[TemplateBasedDB] DB fetch failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {e}")

    md_paths = _write_sources_as_markdown(workspace, ids, sources)

    manifest = "\n".join(
        f"  {i}. id={sid} title={sources[sid]['title'] or '(untitled)'}"
        for i, sid in enumerate(ids, start=1)
    )
    research_instruction = (
        f"{_RESEARCH_DEFAULT_INSTRUCTION}\n\n"
        f"You have been given {len(ids)} independent source document(s) on the same topic:\n"
        f"{manifest}\n"
        "Each source's full content is also available as an attachment file, whose content "
        "starts with an in-file header repeating its id/title."
    )
    if additional_instruction:
        research_instruction += f"\n\nAdditional instructions from the requester:\n{additional_instruction}"

    research_result, _ = await _run_research_stage_from_paths(
        config, workspace, session_id, md_paths, num_pages, auto_page,
        instruction=research_instruction,
        config_file=PACKAGE_DIR / "roles" / "research-db.yaml",
    )
    design_instruction = f"{_DESIGN_DEFAULT_INSTRUCTION}\n\n{_cover_info_block(presenter_name, emp_no, team_name)}"
    design_result = await _run_design_hynix_stage(
        config, workspace, session_id, research_result.manuscript_path, design_instruction,
    )

    return await _design_response(design_result, session_id, export_filename, emp_no)


@app.post("/template-free-ppt-generation-db", tags=["main"], summary="Template-Free-PPT-Generation-DB")
async def template_free_ppt_generation_db(
    ids: list[str] = Form(..., description="sources 테이블에서 raw_text를 조회할 id 목록 (여러 개 전달 가능)"),
    num_pages: int = Form(default=10, description="총 슬라이드 수 (표지 + 마지막 장 포함, auto_page=false일 때만 사용)"),
    auto_page: bool = Form(default=True, description="기본값 true. 문서 내용을 분석해 자동으로 슬라이드 수를 결정. false로 지정하면 num_pages 값을 그대로 사용"),
    export_filename: str = Form(default="slides.pptx"),
    emp_no: str = Form(..., description="MinIO 저장 경로 '{emp_no}/{export_filename}/ppt/{export_filename}.pptx'에 사용되는 사번. 커버 슬라이드에도 표시됨"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
    research_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Research 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    design_model_size: Literal["big", "middle", "small"] = Form(default="big", description="Design 단계에 사용할 모델 티어 (.env의 MODEL_BIG/MODEL_MIDDLE/MODEL_SMALL)"),
    base_url: str | None = Form(default=os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL, description="OpenAI 호환 API 엔드포인트. 비워서 보내면 해당 티어의 기본 엔드포인트를 그대로 사용"),
    additional_request: str | None = Form(default="{}", description='LLM 요청에 병합할 추가 파라미터, JSON 문자열 (예: {"temperature":0.7,"max_tokens":4096})'),
    additional_instruction: str | None = Form(default=None, description="Research 매뉴스크립트 작성 시 반영할 추가 지시사항 (예: 특정 내용을 강조해달라는 지시)"),
):
    """id 목록 → PostgreSQL sources 테이블에서 title/raw_text 조회 → '{id}.md' 파일로 저장 →
    Research 에이전트(research-db.yaml)로 여러 소스를 하나의 원고로 합성 →
    Design(자유 템플릿) 에이전트로 슬라이드 생성,
    변환까지 한 요청에서 이어서 처리하고 PPTX를 반환한다."""
    from deeppresenter.tools.db import fetch_raw_texts
    from deeppresenter.utils.constants import PACKAGE_DIR, WORKSPACE_BASE

    ids = _split_ids(ids)
    if not ids:
        raise HTTPException(status_code=400, detail="ids must contain at least one id.")

    research_llm = _resolve_tiered_llm(research_model_size, additional_request, base_url)
    design_llm = _resolve_tiered_llm(design_model_size, additional_request, base_url)
    config = _make_deep_config(research_llm=research_llm, design_llm=design_llm)

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        sources = fetch_raw_texts(ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("[TemplateFreeDB] DB fetch failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {e}")

    md_paths = _write_sources_as_markdown(workspace, ids, sources)

    manifest = "\n".join(
        f"  {i}. id={sid} title={sources[sid]['title'] or '(untitled)'}"
        for i, sid in enumerate(ids, start=1)
    )
    research_instruction = (
        f"{_RESEARCH_DEFAULT_INSTRUCTION}\n\n"
        f"You have been given {len(ids)} independent source document(s) on the same topic:\n"
        f"{manifest}\n"
        "Each source's full content is also available as an attachment file, whose content "
        "starts with an in-file header repeating its id/title."
    )
    if additional_instruction:
        research_instruction += f"\n\nAdditional instructions from the requester:\n{additional_instruction}"

    research_result, _ = await _run_research_stage_from_paths(
        config, workspace, session_id, md_paths, num_pages, auto_page,
        instruction=research_instruction,
        config_file=PACKAGE_DIR / "roles" / "research-db.yaml",
    )
    design_instruction = f"{_DESIGN_DEFAULT_INSTRUCTION}\n\n{_cover_info_block(presenter_name, emp_no, team_name)}"
    design_result = await _run_design_free_stage(
        config, workspace, session_id, research_result.manuscript_path, design_instruction,
    )

    return await _design_response(design_result, session_id, export_filename, emp_no)


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
                os.environ.get("MODEL_BIG", "claude-opus-4-5"),
                os.environ.get("DESIGN_MODEL_NAME", "(none)"),
                os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL)
    logger.info("Server: host=%s port=%d reload=%s log_level=%s", host, port, reload, log_level)

    uvicorn.run(
        "main-ui:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
