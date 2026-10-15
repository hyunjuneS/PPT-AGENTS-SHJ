import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from deeppresenter.utils.config import get_llm
from deeppresenter.utils.constants import WORKSPACE_BASE
from deeppresenter.utils.log import SessionIdFilter, set_session_id

# .env 파일을 os.environ 에 주입.
load_dotenv()

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s session_id=%(session_id)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    # session_id — 동시에 여러 요청이 들어와도 로그 한 줄 한 줄이 어느 요청 것인지 구분할 수
    # 있도록 모든 로그 라인에 붙인다 (SessionIdFilter가 채움). Loki에서
    # `|= "session_id=abc12345"`로 필터링하면 그 요청의 로그만 시간순으로 모아 볼 수 있다.
    format=_LOG_FORMAT,
    # logging.StreamHandler()의 기본 스트림은 stdout이 아니라 stderr — 명시하지 않으면
    # 정상 운영 로그(logger.info)가 컨테이너의 STDERR로 나가서, 같은 stdout으로 나가는
    # deeppresenter/utils/log.py의 print() 기반 에이전트 진행 로그와 스트림이 갈라진다
    # (Loki 같은 로그 수집기에서 STDOUT/STDERR가 다른 스트림으로 분리돼 보이는 원인).
    stream=sys.stdout,
)

# 콘솔(stdout)에 찍히는 것과 같은 내용을 그대로 파일에도 남긴다 — turn/tool_call/tool_result 등
# deeppresenter.utils.log를 거치는 모든 진행 로그가 여기 포함된다(재시작해도 이어쓰기).
# WORKSPACE_BASE(기본: deeppresenter/output/YYYYMMDD)에 날짜별로 쌓인다.
# 이미 같은 파일 핸들러가 붙어 있으면(예: 테스트에서 모듈을 다시 import) 또 붙이지 않는다 —
# 붙이면 모든 로그 줄이 두 번씩 찍힌다.
_log_file_path = str((WORKSPACE_BASE / "server.log").resolve())
if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == _log_file_path
    for h in logging.root.handlers
):
    WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)
    _log_file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    _log_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.root.addHandler(_log_file_handler)

# 같은 이유로, 이미 붙어 있으면 다시 붙이지 않는다.
for _handler in logging.root.handlers:
    if not any(isinstance(f, SessionIdFilter) for f in _handler.filters):
        _handler.addFilter(SessionIdFilter())
logger = logging.getLogger(__name__)

# openai SDK가 내부적으로 쓰는 httpx의 요청/응답 로그를 켜둔다.
# "openai._base_client: Retrying request..." 메시지만으로는 재시도 원인(429 rate limit인지,
# 5xx인지, 커넥션 문제인지)을 알 수 없는데, httpx를 INFO로 올려두면 그 바로 옆에
# 'HTTP Request: POST .../chat/completions "HTTP/1.1 429 Too Many Requests"' 식으로
# 실제 상태 코드가 같이 찍혀서 원인을 바로 확인할 수 있다.
logging.getLogger("httpx").setLevel(logging.INFO)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """서버가 요청을 받기 시작할 때 한 번 찍는 시작 로그."""
    logger.info("LLM configured: model=%s base_url=%s language=%s", _LLM.model, _LLM.base_url, _LANGUAGE)
    yield


app = FastAPI(title="PPT Agent API", version="0.2.0", lifespan=_lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """모든 API 응답에 처리 소요시간(초)을 X-Process-Time 헤더로 추가."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    # uvicorn 자체 access log는 꺼두고(uvicorn.run의 access_log=False) 이 한 줄로 대신한다 —
    # 둘 다 켜 두면 요청마다 같은 내용이 두 번 찍힌다. 이쪽은 session_id가 붙고 server.log에도 남는다.
    logger.info("[Request] %s %s -> %d (%.3fs)", request.method, request.url.path, response.status_code, elapsed)
    return response

# ---------------------------------------------------------------------------
# LLM — 모델 하나만 사용 (.env의 LLM_MODEL / LLM_URL / LLM_API_KEY).
# Research, Design, 페이지 수 결정, heavy-reflect 시각 검토까지 전부 같은 모델이다.
# ---------------------------------------------------------------------------

# Research/Design 단계에 별도 instruction을 받지 않는 엔드포인트에서 사용하는 고정 지시문.
_RESEARCH_DEFAULT_INSTRUCTION = "Create presentation content based on the attached document."
_DESIGN_DEFAULT_INSTRUCTION = "Create a professional presentation."

# PPT_LANGUAGE env — 출력 언어 고정. 값: "en" (기본) 또는 "ko".
_LANGUAGE: str = os.environ.get("PPT_LANGUAGE", "en")

_LLM = get_llm()


def _make_deep_config():
    """모든 역할(research/design/long_context/vlm)이 같은 LLM을 쓰는 DeepPresenterConfig."""
    from deeppresenter.utils.config import DeepPresenterConfig

    llm = get_llm()
    return DeepPresenterConfig(
        research_agent=llm,
        design_agent=llm,
        long_context_model=llm,
        vlm_agent=llm,
    )


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


async def _design_response(result, session_id: str):
    """생성된 slide_*.html을 같은 요청 안에서 PPTX로 변환해 그대로 반환한다.
    PPTX와 슬라이드 HTML은 workspace(slides_dir)에 그대로 남는다."""
    from deeppresenter.tools.export import html_slides_to_pptx

    slides_dir = result.slides_dir
    html_files = sorted(Path(slides_dir).glob("slide_*.html"))

    pptx_filename = f"{session_id}.pptx"
    pptx_path = Path(slides_dir) / pptx_filename
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
        filename=pptx_filename,
        headers={
            "X-Session-Id": session_id,
            "X-Slides-Dir": slides_dir,
            "X-Slide-Count": str(len(html_files)),
            "X-Turns": str(len(result.messages_log)),
        },
    )


# ---------------------------------------------------------------------------
# Stage runners — Research/Design 각 단계의 실제 실행 로직. 두 생성 엔드포인트
# (/template-based-ppt-generation, /template-free-ppt-generation)가 공유한다.
# ---------------------------------------------------------------------------

async def _run_research_stage(config, workspace, session_id: str, file: UploadFile):
    """업로드된 .md → 슬라이드 원고(ResearchGraphResult) 생성. 슬라이드 수는 항상 문서 내용을
    분석해 자동으로 정한다(decide_num_pages)."""
    from deeppresenter.agents.page_planner import decide_num_pages
    from deeppresenter.graph.research_graph import run_research_graph
    from deeppresenter.utils.typings import InputRequest

    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    attachment_path = workspace / Path(file.filename).name
    attachment_path.write_bytes(raw)

    resolved_num_pages = await decide_num_pages(config.long_context_model, content, _RESEARCH_DEFAULT_INSTRUCTION)

    req = InputRequest(
        instruction=_RESEARCH_DEFAULT_INSTRUCTION,
        attachments=[str(attachment_path)],
        num_pages=str(resolved_num_pages),
        language=_LANGUAGE,
    )

    logger.info(
        "[Research] session=%s lang=%s resolved_num_pages=%d",
        session_id, _LANGUAGE, resolved_num_pages,
    )

    try:
        result = await run_research_graph(
            config=config,
            workspace=workspace,
            req=req,
            language=_LANGUAGE,
        )
    except Exception as e:
        logger.error("[Research] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Research agent failed: {e}")

    return result


async def _run_design_hynix_stage(config, workspace, session_id: str, markdown_file: str, instruction: str):
    """원고(.md) → Design 에이전트(Hynix 템플릿)로 HTML 슬라이드 생성.

    DESIGN_PARALLEL_MODE가 켜져 있으면 run_design_graph_parallel(슬라이드를 워커별로 나눠
    asyncio.Semaphore로 동시 실행)을 대신 쓴다 — 단, DESIGN_CONFIG_FILE이 기본 design-hynix.yaml이
    아닌 다른(커스텀) yaml을 가리킬 때나 DESIGN_TEMPLATE_FILE이 설정된 경우는, 병렬 워커 4개 yaml이
    design-hynix.yaml 구조를 그대로 손으로 옮겨 만든 것이라 그 가정이 안 맞을 수 있어 항상 기존
    직렬 경로(run_design_graph)로 폴백한다. DESIGN_CONFIG_FILE이 design-hynix.yaml 자체를 가리키는
    (일반적인) 경우는 병렬 모드 대상이다. template-free 경로(_run_design_free_stage)는 이 분기와
    무관하게 항상 직렬로만 동작한다."""
    from deeppresenter.graph.design_graph import run_design_graph, run_design_graph_parallel
    from deeppresenter.utils.constants import DESIGN_PARALLEL_MODE, PACKAGE_DIR
    from deeppresenter.utils.typings import InputRequest

    req = InputRequest(instruction=instruction, language=_LANGUAGE)

    template_content = ""
    tmpl_path = os.environ.get("DESIGN_TEMPLATE_FILE")
    if tmpl_path and Path(tmpl_path).exists():
        template_content = Path(tmpl_path).read_text(encoding="utf-8")

    config_file = os.environ.get("DESIGN_CONFIG_FILE") or None
    default_hynix_config_file = PACKAGE_DIR / "roles" / "design-hynix.yaml"
    is_default_hynix_config = bool(config_file) and Path(config_file).resolve() == default_hynix_config_file.resolve()
    use_parallel = DESIGN_PARALLEL_MODE and is_default_hynix_config and not template_content

    logger.info("[DesignHynixTemplate] session=%s lang=%s file=%s config=%s template=%s parallel=%s",
                session_id, _LANGUAGE, Path(markdown_file).name,
                Path(config_file).name if config_file else "(unset)",
                bool(template_content), use_parallel)

    try:
        if use_parallel:
            return await run_design_graph_parallel(
                config=config,
                workspace=workspace,
                req=req,
                markdown_file=markdown_file,
                language=_LANGUAGE,
            )
        return await run_design_graph(
            config=config,
            workspace=workspace,
            req=req,
            markdown_file=markdown_file,
            template_content=template_content,
            config_file=config_file,
            language=_LANGUAGE,
        )
    except Exception as e:
        logger.error("[DesignHynixTemplate] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Design agent failed: {e}")


async def _run_design_free_stage(config, workspace, session_id: str, markdown_file: str, instruction: str):
    """원고(.md) → Design 에이전트(자유 템플릿)로 HTML 슬라이드 생성."""
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
        )
    except Exception as e:
        logger.error("[DesignFreeTemplate] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Design agent failed: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _LLM.model}


@app.get("/api/admin/appReady")
def app_ready():
    return {"status": "ok"}


async def _generate(design_stage, file: UploadFile, emp_no: str, presenter_name: str, team_name: str):
    """.md 파일 → Research 에이전트로 원고 생성 → design_stage로 슬라이드 생성 → PPTX 반환."""
    from deeppresenter.utils.constants import WORKSPACE_BASE

    config = _make_deep_config()

    session_id = str(uuid.uuid4())[:8]
    set_session_id(session_id)
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    research_result = await _run_research_stage(config, workspace, session_id, file)
    design_instruction = f"{_DESIGN_DEFAULT_INSTRUCTION}\n\n{_cover_info_block(presenter_name, emp_no, team_name)}"
    design_result = await design_stage(
        config, workspace, session_id, research_result.manuscript_path, design_instruction,
    )

    return await _design_response(design_result, session_id)


@app.post("/template-based-ppt-generation", tags=["main"], summary="Template-Based-PPT-Generation")
async def template_based_ppt_generation(
    file: UploadFile = File(..., description="원본 문서 (.md, UTF-8)"),
    emp_no: str = Form(..., description="커버 슬라이드에 표시할 사번"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
):
    """.md 파일 → Research 에이전트로 원고 생성 → Design(Hynix 템플릿) 에이전트로 슬라이드 생성,
    변환까지 한 요청에서 이어서 처리하고 PPTX를 반환한다."""
    return await _generate(_run_design_hynix_stage, file, emp_no, presenter_name, team_name)


@app.post("/template-free-ppt-generation", tags=["main"], summary="Template-Free-PPT-Generation")
async def template_free_ppt_generation(
    file: UploadFile = File(..., description="원본 문서 (.md, UTF-8)"),
    emp_no: str = Form(..., description="커버 슬라이드에 표시할 사번"),
    presenter_name: str = Form(..., description="커버 슬라이드에 표시할 이름"),
    team_name: str = Form(..., description="커버 슬라이드에 표시할 팀명"),
):
    """.md 파일 → Research 에이전트로 원고 생성 → Design(자유 템플릿) 에이전트로 슬라이드 생성,
    변환까지 한 요청에서 이어서 처리하고 PPTX를 반환한다."""
    return await _generate(_run_design_free_stage, file, emp_no, presenter_name, team_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 필수 환경변수 검증
    if not os.environ.get("LLM_API_KEY"):
        print("error: LLM_API_KEY is required (set in .env or environment)", file=sys.stderr)
        sys.exit(1)

    # 경로 검증
    for env_key in ("DESIGN_CONFIG_FILE", "DESIGN_TEMPLATE_FILE"):
        val = os.environ.get(env_key)
        if val and not Path(val).exists():
            print(f"error: {env_key} not found: {val}", file=sys.stderr)
            sys.exit(1)

    host      = os.environ.get("HOST", "0.0.0.0")
    port      = int(os.environ.get("PORT", "5000"))
    log_level = os.environ.get("LOG_LEVEL", "info")

    logger.info("Server: host=%s port=%d log_level=%s", host, port, log_level)

    # 리로드는 쓰지 않는다. 앱 객체를 직접 넘기므로 uvicorn이 이 파일을 다시 import하지 않는다.
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
