import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
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
from backend.src.api.auth import get_api_key


from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/analytics", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_analytics_dashboard():
    """
    Serves the main web dashboard that the security team sees on their screens.
    It returns the raw HTML file from the static folder so the browser can load the UI.
    Calls the local file system to read analytics.html.
    """
    analytics_file = STATIC_DIR / "analytics.html"
    if analytics_file.exists():
        return FileResponse(analytics_file)
    return HTMLResponse("<h2>Analytics Dashboard HTML loading...</h2>")


@app.get("/health")
@app.get("/api/health")
async def health():
    """Render health check endpoint — must stay fast and dependency-free."""
    import time
    return {"status": "healthy", "service": "botocop-fraud-engine", "version": "2.4.0"}



@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint. Point prometheus.yml at this path --
    see monitoring/prometheus.yml for the scrape config."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    A live open connection where incoming events (like new phone calls) stream in.
    When an event arrives, it hands it off to the main brain to figure out if it's fraud.
    Calls handle_event in backend.src.orchestrator.orchestrator.
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

@app.post("/api/call-fraud/analyze", dependencies=[Depends(get_api_key)])
async def analyze_call_fraud(payload: dict):
    """
    An alternative way to send a single call event to the fraud engine via a normal web request instead of WebSocket.
    It returns the final fraud decision directly to the caller.
    Calls handle_event in backend.src.orchestrator.orchestrator.
    """
    from backend.src.orchestrator.orchestrator import handle_event
    try:
        result = handle_event("call", payload)
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Call fraud analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/{entity_id}")
async def fast_lane_risk_check(entity_id: str):
    """
    The super-fast check that the payment system calls before allowing money to move.
    It looks up the customer's current fraud status without asking the slow AI.
    Calls get_open_case_for_entity in backend.src.case.store.
    """
    from backend.src.case.store import get_open_case_for_entity, pending_count
    import time
    start = time.time()
    
    # Fast DB read (bypasses LLM/pipelines)
    case = get_open_case_for_entity(entity_id)
    pending = pending_count(entity_id)
    
    # Apply simple action policy based on case state
    decision = "allow"
    reason = "No high-risk case found"
    risk_score = 0.0
    
    if case:
        risk_score = case.risk_score
        if case.status == "ESCALATED":
            decision = "hold"
            reason = f"Customer has an ESCALATED cross-channel case ({case.case_id})"
        elif case.status == "PENDING_REVIEW" or risk_score >= 0.6:
            decision = "step_up"
            reason = f"Customer has a high-risk open case ({case.case_id})"
            
    latency_ms = int((time.time() - start) * 1000)
    logger.info(f"Fast-lane risk check for {entity_id}: {decision} ({latency_ms}ms)")
    
    return {
        "status": "ok",
        "entity_id": entity_id,
        "decision": decision,
        "reason": reason,
        "risk_score": risk_score,
        "analysis_pending": pending > 0,
        "latency_ms": latency_ms,
        "case_id": case.case_id if case else None
    }

@app.get("/api/v1/hitl/pending")
async def get_pending_hitl_reviews():
    """
    Gets the list of escalated cases that are waiting for a human analyst to review them.
    This populates the 'Pending Reviews' list on the dashboard.
    Calls get_hitl_queue in backend.src.pipelines.call_fraud.hitl_queue.
    """
    from backend.src.pipelines.call_fraud.hitl_queue import get_hitl_queue
    queue = get_hitl_queue()
    return {"status": "ok", "pending_cases": queue.get_pending_reviews()}


class HITLResolveRequest(BaseModel):
    case_id: str
    analyst_id: str
    decision: str = Field(description="'closed_fraud' or 'closed_cleared'")
    notes: Optional[str] = ""
    blocklist_entries: Optional[list] = [] # list of {"type": "phone", "value": "123"}


@app.post("/api/v1/hitl/resolve", dependencies=[Depends(get_api_key)])
async def resolve_hitl_review(req: HITLResolveRequest):
    """
    When a human analyst clicks 'Confirm Fraud' or 'Clear' on the dashboard, this handles it.
    It closes the case, saves their notes, and blocks the scammer's number all in one step.
    Calls resolve_case in backend.src.case.store.
    """
    from backend.src.case.store import resolve_case
    try:
        success = resolve_case(
            case_id=req.case_id,
            new_status=req.decision,
            actor=req.analyst_id,
            reason=req.notes or "Resolved by analyst",
            blocklist_entries=req.blocklist_entries
        )
        if not success:
            raise HTTPException(status_code=404, detail="Case not found or could not be locked")
        return {"status": "ok", "message": f"Case {req.case_id} resolved as {req.decision}"}
    except Exception as e:
        logger.error(f"HITL resolution failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/blocklist")
async def get_scam_blocklist_numbers():
    """
    Gets the full list of phone numbers that have been permanently blocked because they belong to confirmed scammers.
    This tells the phone system who to hang up on immediately.
    Calls get_scam_blocklist in backend.src.pipelines.call_fraud.blocklist.
    """
    from backend.src.pipelines.call_fraud.blocklist import get_scam_blocklist
    bl = get_scam_blocklist()
    return {"status": "ok", "blocklist": bl._blocklist}


class BlocklistAddRequest(BaseModel):
    phone: str
    source: Optional[str] = "Manual_Admin"
    reason: Optional[str] = "Reported Scam Caller"


@app.post("/api/v1/blocklist", dependencies=[Depends(get_api_key)])
async def add_scam_number_to_blocklist(req: BlocklistAddRequest):
    """
    Manually adds a bad phone number to the blocklist so it can't call anyone again.
    Usually done by an analyst if they discover a new scam number outside the normal system.
    Calls get_scam_blocklist in backend.src.pipelines.call_fraud.blocklist.
    """
    from backend.src.pipelines.call_fraud.blocklist import get_scam_blocklist
    bl = get_scam_blocklist()
    bl.add_scam_number(phone=req.phone, source=req.source, reason=req.reason)
    return {"status": "ok", "message": f"Phone {req.phone} added to Layer 4 Blocklist."}


@app.get("/api/v1/evidence/{case_id}")
async def get_case_evidence(case_id: str):
    """
    Retrieves the secure, tamper-proof record of why we flagged a specific case as fraud.
    This is used when police or lawyers ask for proof of why an account was frozen.
    Calls get_evidence_vault in backend.src.pipelines.call_fraud.evidence_store.
    """
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
    """
    Checks if the fake event generator (which simulates attacks) is currently running.
    Used by the dashboard to show the 'Simulation Active' green light.
    Calls get_simulation_engine in backend.src.simulation.engine.
    """
    from backend.src.simulation.engine import get_simulation_engine
    return get_simulation_engine().get_status()


@app.post("/api/simulation/toggle", dependencies=[Depends(get_api_key)])
async def toggle_simulation():
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    engine.set_running(not engine.is_running)
    return engine.get_status()


@app.post("/api/simulation/interval", dependencies=[Depends(get_api_key)])
async def set_simulation_interval(req: IntervalRequest):
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    engine.set_interval(req.seconds)
    return engine.get_status()


@app.post("/api/simulation/inject", dependencies=[Depends(get_api_key)])
async def inject_simulation_scenario(req: ScenarioInjectRequest):
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    result = engine.inject_scenario(req.scenario)
    return {"status": "ok", "event": result}


@app.post("/api/simulation/reset", dependencies=[Depends(get_api_key)])
async def reset_simulation_stats():
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    engine.reset_stats()
    return engine.get_status()


@app.get("/api/threats/live")
async def get_live_threats():
    """
    Gets the most recent simulated attacks so they can be shown in a live scrolling feed.
    Used by the dashboard's 'Live Threat Feed' component.
    Calls get_simulation_engine in backend.src.simulation.engine.
    """
    from backend.src.simulation.engine import get_simulation_engine
    engine = get_simulation_engine()
    return {"status": "ok", "threats": list(engine.recent_events)}


@app.get("/api/threats/stats")
async def get_threat_stats():
    """
    Gathers all the high-level numbers (total cases, total blocked) from different parts of the system.
    This provides the data for the big summary charts on the dashboard.
    Calls multiple functions across store.py, hitl_queue.py, blocklist.py, and engine.py.
    """
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


# --- Action layer endpoints (Gap 1) ---

@app.get("/api/v1/actions/log")
async def get_action_log():
    """Get the audit trail of all action decisions the engine has made."""
    from backend.src.actions.policy_engine import get_action_log
    return {"status": "ok", "actions": get_action_log()}


@app.post("/api/v1/actions/evaluate", dependencies=[Depends(get_api_key)])
async def evaluate_action_endpoint(payload: dict):
    """
    Check what action the engine would take for a given entity.
    Useful for testing policies without actually executing them.
    """
    from backend.src.actions.policy_engine import evaluate_action
    from backend.src.case.store import get_open_case_for_entity, get_case_with_events

    entity_id = payload.get("entity_id")
    if not entity_id:
        raise HTTPException(status_code=400, detail="entity_id is required")

    case = get_open_case_for_entity(entity_id)
    case_data = None
    if case:
        case_data = get_case_with_events(case.case_id)

    event_data = {
        "channel": payload.get("channel", "transaction"),
        "is_first_time_payee": payload.get("is_first_time_payee", False),
    }

    decision = evaluate_action(case_data, event_data)
    return {"status": "ok", "decision": decision.to_dict()}


# --- Notification layer endpoints (Gap 4) ---

@app.get("/api/v1/notifications/pending")
async def get_pending_notifications_endpoint(priority: Optional[str] = None):
    """Get all unacknowledged analyst alerts, sorted by urgency."""
    from backend.src.notifications.dispatcher import get_pending_notifications
    return {"status": "ok", "notifications": get_pending_notifications(priority)}


@app.post("/api/v1/notifications/acknowledge", dependencies=[Depends(get_api_key)])
async def acknowledge_notification_endpoint(payload: dict):
    """Mark a notification as seen by an analyst (stops the SLA timer)."""
    from backend.src.notifications.dispatcher import acknowledge_notification
    notif_id = payload.get("notification_id")
    analyst_id = payload.get("analyst_id")
    if not notif_id or not analyst_id:
        raise HTTPException(status_code=400, detail="notification_id and analyst_id are required")
    success = acknowledge_notification(notif_id, analyst_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "ok", "acknowledged": True}


@app.get("/api/v1/notifications/sla-breaches")
async def get_sla_breaches():
    """Check for analyst alerts where nobody acted within the SLA window."""
    from backend.src.notifications.dispatcher import check_sla_breaches
    return {"status": "ok", "breaches": check_sla_breaches()}


@app.get("/api/v1/notifications/stats")
async def get_notification_stats_endpoint():
    """Summary stats for the notification system."""
    from backend.src.notifications.dispatcher import get_notification_stats
    return {"status": "ok", "stats": get_notification_stats()}


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
            "/api/v1/risk/{entity_id}",
            "/api/v1/hitl/pending",
            "/api/v1/hitl/resolve",
            "/api/v1/blocklist",
            "/api/v1/evidence/{case_id}",
            "/api/v1/actions/log",
            "/api/v1/actions/evaluate",
            "/api/v1/notifications/pending",
            "/api/v1/notifications/acknowledge",
            "/api/v1/notifications/sla-breaches",
            "/api/v1/notifications/stats",
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
