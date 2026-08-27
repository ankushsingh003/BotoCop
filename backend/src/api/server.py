import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, AliasChoices
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("botocop-web")

app = FastAPI(title="BotoCop Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_checks():
    """
    Safety net so 'forgot to run init_db() first' can't turn into a live
    500 on the very first case-layer write. Idempotent -- create_all()
    only creates tables that don't already exist, so this is cheap and
    harmless to run on every boot, including every container restart.

    Also fails loudly (a clear log line, not a silent per-request
    failure) if GEMINI_API_KEY is unset, since every pipeline and both eval
    judges depend on it -- without this, the app boots fine and just
    quietly returns final_status="failed" on every single audit.
    """
    from backend.src.case.db import init_db
    init_db()
    logger.info("Case DB tables verified/created on startup.")

    if not os.getenv("GEMINI_API_KEY"):
        logger.warning(
            "GEMINI_API_KEY is not set. The server will start, but every "
            "pipeline audit and both eval-agent judges call Gemini and will "
            "fail on every request until this is configured."
        )


from typing import Optional


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint. Point prometheus.yml at this path --
    see monitoring/prometheus.yml for the scrape config."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    Live event ingestion. Each incoming message is:
        {"channel": "transaction" | "call", "payload": {...}}
    routed through the orchestrator (case linking -> specialist pipeline
    -> eval loop -> cross-channel risk), one response per event:
        {"status": "ok", "result": {...}}  or  {"status": "error", "error": "..."}
    Kept as a plain per-connection loop rather than a Kafka consumer for
    now -- swap the receive_json() loop for a Kafka consumer without
    touching orchestrator.handle_event once real throughput requires it.
    """
    await websocket.accept()
    from backend.src.orchestrator.orchestrator import handle_event

    logger.info("WebSocket client connected to /ws/events")
    try:
        while True:
            data = await websocket.receive_json()
            channel = data.get("channel")
            payload = data.get("payload", {})
            try:
                result = handle_event(channel, payload)
                await websocket.send_json({"status": "ok", "result": result})
            except Exception as e:
                logger.error(f"Event handling failed: {e}")
                await websocket.send_json({"status": "error", "error": str(e)})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/events")

@app.get("/")
async def root():
    return {"status": "botocop-api online", "message": "API is strictly event-driven. Use WebSocket or Kafka for ingestion."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
