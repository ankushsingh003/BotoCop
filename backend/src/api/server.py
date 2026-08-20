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

# Load environment variables
load_dotenv(override=True)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("botocop-web")

app = FastAPI(title="BotoCop Web API")

# Setup CORS
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
    failure) if GROQ_API_KEY is unset, since every pipeline and both eval
    judges depend on it -- without this, the app boots fine and just
    quietly returns final_status="failed" on every single audit.
    """
    from backend.src.case.db import init_db
    init_db()
    logger.info("Case DB tables verified/created on startup.")

    if not os.getenv("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY is not set. The server will start, but every "
            "pipeline audit and both eval-agent judges call Groq and will "
            "fail on every request until this is configured."
        )


from typing import Optional

# Request & Response Models
class AuditRequest(BaseModel):
    url: str = Field(validation_alias=AliasChoices("url", "video_url"))

class ViolationItem(BaseModel):
    rule: str
    severity: str        # "high" / "medium" / "low"
    description: str
    source: str          # "audio" or "visual"

class AuditResponse(BaseModel):
    video_url: str
    status: str                        # "pass" or "fail"
    domain: str                        # "finance" / "healthcare" / "food_and_beverage" / "general"
    audio_violations: list[ViolationItem]
    visual_violations: list[ViolationItem]
    rag_sources: list[str]
    total_violations: int

@app.post("/audit", response_model=AuditResponse)
@app.post("/api/audit", response_model=AuditResponse)
async def audit_video(request: AuditRequest):
    try:
        # Lazy load the heavy graph only when needed
        logger.info("Importing video audit graph...")
        from backend.src.graph.workflow import video_audit_graph
        
        session_id = str(uuid.uuid4())
        url = request.url
        logger.info(f"Audit requested for: {url} (Session: {session_id})")
        
        input_data = {
            "video_url": url,
            "video_id": session_id[:8],
            "compliance_result": [],
            "error": []
        }
        
        # Invoke the graph
        result = video_audit_graph.invoke(input_data)
        merged = result.get("merged_report", {})
        
        # Ensure merged_report has expected audio/visual keys (for fallback/legacy support)
        if "audio" not in merged or "visual" not in merged:
            visual_list = result.get("visual_violations") or []
            compliance_list = result.get("compliance_result") or []
            visual_desc_set = {v.get("description") for v in visual_list if v.get("description")}
            audio_list = []
            v_len = len(visual_list)
            if v_len > 0 and len(compliance_list) >= v_len:
                match = True
                for i in range(v_len):
                    if compliance_list[-v_len + i].get("description") != visual_list[i].get("description"):
                        match = False
                        break
                if match:
                    audio_list = compliance_list[:-v_len]
                else:
                    audio_list = [item for item in compliance_list if item.get("description") not in visual_desc_set]
            else:
                audio_list = compliance_list
                
            def map_severity(raw_sev: str) -> str:
                if not raw_sev:
                    return "low"
                raw_sev_lower = raw_sev.lower()
                if raw_sev_lower in ("high", "medium", "low"):
                    return raw_sev_lower
                if "critical" in raw_sev_lower:
                    return "high"
                elif "warning" in raw_sev_lower:
                    return "medium"
                elif "info" in raw_sev_lower:
                    return "low"
                return "low"
                
            merged["audio"] = [
                {
                    "rule": item.get("category") or "General",
                    "severity": map_severity(item.get("severity")),
                    "description": item.get("description") or "",
                    "source": "audio"
                }
                for item in audio_list
            ]
            merged["visual"] = [
                {
                    "rule": item.get("category") or "General",
                    "severity": map_severity(item.get("severity")),
                    "description": item.get("description") or "",
                    "source": "visual"
                }
                for item in visual_list
            ]
            merged["status"] = "pass" if result.get("final_status") == "success" else "fail"
            
        return AuditResponse(
            video_url=url,
            status=merged.get("status", "unknown"),
            domain=result.get("domain") or "general",
            audio_violations=merged.get("audio", []),
            visual_violations=merged.get("visual", []),
            rag_sources=result.get("rag_sources", []),
            total_violations=len(merged.get("audio", [])) + len(merged.get("visual", []))
        )
        
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Basic health check
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

# Serve Frontend
# server.py is in backend/src/api/
# static is in backend/static/
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static"))
if not os.path.exists(static_path):
    os.makedirs(static_path)

logger.info(f"Serving static files from: {static_path}")
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
