import argparse
import logging
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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

    manuscript_content = Path(manuscript_path).read_text(encoding="utf-8")
    return JSONResponse(content={
        "session_id": session_id,
        "manuscript_path": manuscript_path,
        "manuscript": manuscript_content,
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
