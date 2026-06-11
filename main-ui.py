import logging
import os

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agents.agent import Agent
from agents.llms import AsyncLLM

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPT Agent API", version="0.1.0")

# ---------------------------------------------------------------------------
# LLM setup — configure via environment variables
# ---------------------------------------------------------------------------
_llm = AsyncLLM(
    model=os.environ.get("MODEL_NAME", "claude-opus-4-5"),
    base_url=os.environ.get("OPENAI_BASE_URL", None),
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

LLM_MAPPING: dict[str, AsyncLLM] = {"language": _llm}

logger.info("LLM configured: %s", _llm)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _llm.model}


@app.post("/analyze")
async def analyze_markdown(file: UploadFile = File(...)):
    """Receive a .md file and return a structured analysis as JSON."""
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

    agent = Agent(name="md_analyzer", llm_mapping=LLM_MAPPING)

    try:
        turn_id, result = await agent(markdown_document=markdown_text)
    except Exception as e:
        logger.error("Agent error: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return JSONResponse(content={"turn_id": turn_id, "result": result})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main-ui:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
    )
