import uuid
import json
import logging
from dotenv import load_dotenv

load_dotenv(override=True)
from backend.src.graph.workflow import video_audit_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug-audit")

def debug_audit():
    video_url = "https://youtu.be/lvFfHH6unkc"
    session_id = str(uuid.uuid4())[:8]
    
    input_data = {
        "video_url": video_url,
        "video_id": session_id,
        "compliance_result": [],
        "error": []
    }
    
    print(f"Starting debug audit for: {video_url}")
    try:
        result = video_audit_graph.invoke(input_data)
        print("\nAudit Result:")
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print("\nAudit Failed with Exception:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_audit()
