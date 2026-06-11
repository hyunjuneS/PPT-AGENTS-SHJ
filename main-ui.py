import argparse
import logging
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from agents.agent import Agent
from agents.llms import AsyncLLM

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPT Agent API", version="0.2.0")

# ---------------------------------------------------------------------------
# LLM — 환경변수에서 읽음.
# __main__ 블록에서 args → os.environ 에 먼저 쓴 뒤 uvicorn을 띄우기 때문에
# reload worker가 이 모듈을 다시 import해도 올바른 값을 가져간다.
# ---------------------------------------------------------------------------
_llm = AsyncLLM(
    model=os.environ.get("MODEL_NAME", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
)

LLM_MAPPING: dict[str, AsyncLLM] = {"language": _llm}

logger.info("LLM configured: %s", _llm)


def _make_deep_config():
    """DeepPresenterConfig을 현재 _llm 설정으로 생성."""
    from deeppresenter.utils.config import DeepPresenterConfig, LLM
    deep_llm = LLM(
        model=_llm.model,
        base_url=_llm.base_url,
        api_key=_llm.api_key,
    )
    return DeepPresenterConfig(
        research_agent=deep_llm,
        design_agent=deep_llm,
        long_context_model=deep_llm,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _llm.model}


@app.post("/analyze")
async def analyze_markdown(file: UploadFile = File(...)):
    """[pptagent] .md 파일을 doc_extractor 에이전트로 분석해 JSON 반환."""
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    raw = await file.read()
    try:
        markdown_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    if not markdown_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("Analyzing file: %s (%d chars)", file.filename, len(markdown_text))
    agent = Agent(name="doc_extractor", llm_mapping=LLM_MAPPING)

    try:
        turn_id, result = await agent(markdown_document=markdown_text)
    except Exception as e:
        logger.error("Agent error: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return JSONResponse(content={"turn_id": turn_id, "result": result})


@app.post("/research")
async def research(
    file: UploadFile = File(...),
    instruction: str = Form(...),
    language: str = Form(default="ko"),
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
        language=language,
    )

    logger.info("[Research] session=%s instruction=%r", session_id, instruction[:80])

    config = _make_deep_config()
    manuscript_path = None
    messages_log = []

    try:
        async with AgentEnv(workspace) as env:
            agent = Research(config=config, agent_env=env, workspace=workspace, language=language)
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
):
    """HTML 슬라이드 폴더(slides_dir) → PPTX 파일 변환 후 다운로드."""
    from deeppresenter.tools.export import html_slides_to_pptx

    slides_path = Path(slides_dir)
    if not slides_path.exists() or not slides_path.is_dir():
        raise HTTPException(status_code=400, detail=f"slides_dir not found: {slides_dir}")

    html_files = sorted(slides_path.glob("slide_*.html"))
    if not html_files:
        raise HTTPException(status_code=400, detail="No slide_*.html files found in slides_dir.")

    pptx_path = slides_path / filename
    logger.info("[Export] %d slides → %s", len(html_files), pptx_path)

    try:
        await html_slides_to_pptx(
            slides_dir=str(slides_path),
            output_path=str(pptx_path),
            aspect_ratio=aspect_ratio,
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
    language: str = Form(default="ko"),
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

    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_BASE / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    manuscript_path = workspace / file.filename
    manuscript_path.write_bytes(raw)

    req = InputRequest(instruction=instruction, language=language)

    logger.info("[Design] session=%s file=%s", session_id, file.filename)

    config = _make_deep_config()
    slides_dir = None
    messages_log = []

    try:
        async with AgentEnv(workspace) as env:
            agent = Design(config=config, agent_env=env, workspace=workspace, language=language)
            async for item in agent.loop(req, markdown_file=str(manuscript_path)):
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPT Agent FastAPI server")
    # LLM
    parser.add_argument("--apikey",  required=True, help="LLM API key")
    parser.add_argument("--llmurl",  default=None,  help="LLM base URL (OpenAI-compatible)")
    parser.add_argument("--model",   default="claude-opus-4-5", help="Model name (default: claude-opus-4-5)")
    parser.add_argument("--timeout", type=int, default=120, help="LLM request timeout in seconds (default: 120)")
    # Server
    parser.add_argument("--host",      default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",      type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument("--reload",    action=argparse.BooleanOptionalAction, default=True, help="Auto-reload (default: on)")
    parser.add_argument("--log-level", default="debug",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Uvicorn log level (default: debug)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["OPENAI_API_KEY"]  = args.apikey
    os.environ["MODEL_NAME"]      = args.model
    os.environ["LLM_TIMEOUT"]     = str(args.timeout)
    if args.llmurl:
        os.environ["OPENAI_BASE_URL"] = args.llmurl

    logger.info("LLM  : model=%s url=%s", args.model, args.llmurl)
    logger.info("Server: host=%s port=%d reload=%s log_level=%s",
                args.host, args.port, args.reload, args.log_level)

    uvicorn.run(
        "main-ui:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
