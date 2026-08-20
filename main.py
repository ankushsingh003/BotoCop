def main():
    print("Hello from complianceoapipeline!")


import uuid
import json 
import logging 
import pprint
from pprint import PrettyPrinter
from dotenv import load_dotenv

load_dotenv(override=True)
from backend.src.graph.workflow import video_audit_graph
logging.basicConfig( level = logging.INFO , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("brand-compliance-rules")

def run_cli_simulation():
    session_id = str(uuid.uuid4())
    logger.info(f" Starting the Audit Session : {session_id}")

    input_data = {
        "video_url": "https://youtu.be/yx39ed__8ZA", # Example URL
        "video_id": str(uuid.uuid4())[:8],
        "compliance_result": [],
        "error": []
    }
    
    print("Initializing workflow")
    print(f" Input Payload : {json.dumps(input_data , indent=2)}")

    try:
        final_state = video_audit_graph.invoke(input_data)
        print("\n" + "="*60)
        print("Workflow Completed")
        print("="*60)
        print(f"video id : {final_state.get('video_id')}")
        print(f"Final Status : {final_state.get('final_status')}")
        print(f"Compliance Results : {final_state.get('compliance_result')}")
        print(f"Errors : {final_state.get('error')}")
        print(f"RAG Sources Used : {final_state.get('rag_sources')}")
        print(f"Visual Status : {final_state.get('visual_status')}")
        print(f"Visual Violations : {final_state.get('visual_violations')}")
        print(f"Selected Frames Count : {len(final_state.get('selected_frames') or [])}")
        print(f"Merged Report Summary : {final_state.get('merged_report')}")
        print(f"Final Consolidated Message:\n{final_state.get('final_message')}")
        results = final_state.get('compliance_result',[])

        if results:
            for issue in results:
                print(f"- [{issue.get('severity')}] [{issue.get('category')}] {issue.get('description')}")

        else:
            print("No compliance issues found.")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        print(f"Workflow failed: {e}")

    
if __name__ == "__main__":
    # main()
    run_cli_simulation()
