"""
Wraps the existing video_audit_graph (the original BotoCop pipeline,
untouched) so it can be called through orchestrator.handle_event() the
same way as the other three channels, normalized to the
{"violations": [...], "final_status": ...} shape the case aggregator
expects.

Honest limitation: retrying this pipeline re-runs the ENTIRE video
audit -- download, transcription, frame extraction, both LLM audits --
not just the part that might have been wrong. That's expensive per
retry compared to the other channels (a single LLM call). Left as-is
for now rather than over-engineering a partial-retry/caching layer
before there's a real need for one; worth revisiting if video volume
ever makes the retry cost matter.
"""
import logging
from typing import Dict, Any, Optional

from backend.src.graph.workflow import video_audit_graph

logger = logging.getLogger("video-compliance")


def run_video_compliance_pipeline(video_event: Dict[str, Any], retry_feedback: Optional[str] = None) -> dict:
    video_url = video_event.get("video_url")
    video_id = video_event.get("video_id") or video_url

    result = video_audit_graph.invoke({
        "video_url": video_url,
        "video_id": video_id,
        "domain": video_event.get("domain"),
        "compliance_result": [],
        "error": [],
    })

    merged = result.get("merged_report") or {}
    audio = merged.get("audio", [])
    visual = merged.get("visual", [])

    violations = [
        {
            "category": v.get("rule", "General"),
            "description": v.get("description", ""),
            "severity": v.get("severity", "low"),
            "suggestion": f"Review flagged {v.get('source', 'content')} segment.",
        }
        for v in (audio + visual)
    ]

    return {
        "violations": violations,
        "final_status": result.get("final_status", "failed"),
        "rag_sources": result.get("rag_sources", []),
    }
