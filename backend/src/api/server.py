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
    Safety net & automatic background engine launcher on boot.
    Initializes case database tables and automatically starts the 5-layer
    telephony threat simulation engine so deployed environments immediately
    stream live telemetry and metrics to the admin dashboard.
    """
    from backend.src.case.db import init_db
    init_db()
    logger.info("Case DB tables verified/created on startup.")

    # Automatically start simulation engine daemon on server boot
    try:
        from backend.src.simulation.engine import get_simulation_engine
        sim_engine = get_simulation_engine()
        sim_engine.start_background_simulation()
        logger.info("Cyber Security Threat Simulation Engine automatically launched on server boot.")
    except Exception as e:
        logger.warning(f"Simulation engine startup note: {e}")

    if not os.getenv("GEMINI_API_KEY"):
        logger.warning(
            "GEMINI_API_KEY is not set. Employing high-confidence rule-based ML evaluation."
        )


from typing import Optional


from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/analytics", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_analytics_dashboard():
    """Serve live BotoCop Telephony Fraud Engine Analytics & Grafana Dashboard UI."""
    analytics_file = STATIC_DIR / "analytics.html"
    if analytics_file.exists():
        return FileResponse(analytics_file)
    return HTMLResponse("<h2>Analytics Dashboard HTML loading...</h2>")


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

@app.post("/api/call-fraud/analyze")
async def analyze_call_fraud(payload: dict):
    """
    Direct REST endpoint to submit a call fraud event for automated ML analysis,
    identity correlation graph resolution, and LLM forensic audit.
    """
    from backend.src.orchestrator.orchestrator import handle_event
    try:
        result = handle_event("call", payload)
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Call fraud analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/hitl/pending")
async def get_pending_hitl_reviews():
    """Layer 5: Fetch all call fraud cases pending Human-In-The-Loop analyst verification."""
    from backend.src.pipelines.call_fraud.hitl_queue import get_hitl_queue
    queue = get_hitl_queue()
    return {"status": "ok", "pending_cases": queue.get_pending_reviews()}


class HITLResolveRequest(BaseModel):
    case_id: str
    analyst_id: str
    decision: str = Field(description="'CONFIRM_FRAUD' or 'OVERRIDE_FALSE_POSITIVE'")
    notes: Optional[str] = ""


@app.post("/api/v1/hitl/resolve")
async def resolve_hitl_review(req: HITLResolveRequest):
    """
    Layer 5: Submit human analyst verification decision.
    CONFIRM_FRAUD automatically populates Layer 4 Blocklist.
    """
    from backend.src.pipelines.call_fraud.hitl_queue import get_hitl_queue
    queue = get_hitl_queue()
    try:
        resolved_case = queue.resolve_review(
            case_id=req.case_id,
            analyst_id=req.analyst_id,
            decision=req.decision,
            notes=req.notes or ""
        )
        return {"status": "ok", "resolved_case": resolved_case}
    except Exception as e:
        logger.error(f"HITL resolution failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/blocklist")
async def get_scam_blocklist_numbers():
    """Layer 4: Retrieve all confirmed scam caller IDs in the deterministic blocklist."""
    from backend.src.pipelines.call_fraud.blocklist import get_scam_blocklist
    bl = get_scam_blocklist()
    return {"status": "ok", "blocklist": bl._blocklist}


class BlocklistAddRequest(BaseModel):
    phone: str
    source: Optional[str] = "Manual_Admin"
    reason: Optional[str] = "Reported Scam Caller"


@app.post("/api/v1/blocklist")
async def add_scam_number_to_blocklist(req: BlocklistAddRequest):
    """Layer 4: Add a new confirmed scam caller ID to the deterministic blocklist."""
    from backend.src.pipelines.call_fraud.blocklist import get_scam_blocklist
    bl = get_scam_blocklist()
    bl.add_scam_number(phone=req.phone, source=req.source, reason=req.reason)
    return {"status": "ok", "message": f"Phone {req.phone} added to Layer 4 Blocklist."}


@app.get("/api/v1/evidence/{case_id}")
async def get_case_evidence(case_id: str):
    """Fetch court-admissible SHA-256 chain-of-custody evidence package by case_id."""
    from backend.src.pipelines.call_fraud.evidence_store import get_evidence_vault
    vault = get_evidence_vault()
    record = vault.get_evidence(case_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Evidence record for case {case_id} not found.")
    return {"status": "ok", "evidence": record}


# --- SOC Simulation Engine & Real-Time Threat Feed Endpoints ---
class ScenarioInjectRequest(BaseModel):
    scenario: str = "digital_arrest"


class IntervalRequest(BaseModel):
    seconds: float = 6.0


@app.get("/api/simulation/status")
async def get_simulation_status():
    from backend.src.simulation.engine import get_simulation_engine
    return get_simulation_engine().get_status()


@app.post("/api/simulation/toggle")
async def toggle_simulation():
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    engine.set_running(not engine.is_running)
    return engine.get_status()


@app.post("/api/simulation/interval")
async def set_simulation_interval(req: IntervalRequest):
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    engine.set_interval(req.seconds)
    return engine.get_status()


@app.post("/api/simulation/inject")
async def inject_simulation_scenario(req: ScenarioInjectRequest):
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    result = engine.inject_scenario(req.scenario)
    return {"status": "ok", "event": result}


@app.post("/api/simulation/reset")
async def reset_simulation_stats():
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    engine.reset_stats()
    return engine.get_status()


@app.get("/api/threats/live")
async def get_live_threats():
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    return {"status": "ok", "threats": list(engine.recent_events)}


@app.get("/api/threats/stats")
async def get_threat_stats():
    from backend.src.case.store import get_case_statistics
    from backend.src.pipelines.call_fraud.hitl_queue import get_hitl_queue
    from backend.src.pipelines.call_fraud.blocklist import get_scam_blocklist
    from backend.src.simulation.engine import get_simulation_engine

    engine = get_simulation_engine()
    db_stats = get_case_statistics()
    queue = get_hitl_queue()
    bl = get_scam_blocklist()

    return {
        "status": "ok",
        "simulation": engine.get_status(),
        "database": db_stats,
        "pending_hitl_count": len(queue.get_pending_reviews()),
        "blocklist_count": len(bl._blocklist),
    }


@app.get("/api/cases/recent")
async def get_recent_cases_endpoint(limit: int = 20):
    from backend.src.case.store import get_recent_cases
    cases = get_recent_cases(limit=limit)
    return {"status": "ok", "cases": cases}


@app.get("/api/cases/{case_id}")
async def get_case_details_endpoint(case_id: str):
    from backend.src.case.store import get_case_with_events
    case = get_case_with_events(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "ok", "case": case}


@app.get("/")
@app.get("/admin", response_class=HTMLResponse)
@app.get("/analytics", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve live BotoCop Telephony Fraud Engine SOC Dashboard UI."""
    analytics_file = STATIC_DIR / "analytics.html"
    if analytics_file.exists():
        return FileResponse(analytics_file)
    return HTMLResponse("<h2>BotoCop SOC Security Dashboard loading...</h2>")


@app.get("/api")
async def root_api():
    return {
        "status": "botocop-api online",
        "message": "BotoCop Fraud Engine active with automated 5-Layer ML Call Fraud Pipeline.",
        "endpoints": [
            "/ws/events",
            "/api/call-fraud/analyze",
            "/api/v1/hitl/pending",
            "/api/v1/hitl/resolve",
            "/api/v1/blocklist",
            "/api/v1/evidence/{case_id}",
            "/api/simulation/status",
            "/api/simulation/inject",
            "/api/threats/live",
            "/api/threats/stats",
            "/api/cases/recent",
            "/health",
            "/metrics"
        ]
    }





if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
