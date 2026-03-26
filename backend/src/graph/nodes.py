from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage , SystemMessage
import requests

from backend.src.graph.state import VideoAuditState , complianceIssue
from backend.src.services.video_index import VideoIndexerServices

logger = logging.getLogger("brand-compliance-rules")
logging.basicConfig(level=logging.INFO)

# INDEXER

def index_video_node( state: VideoAuditState) -> Dict[str , Any]:
    """
    Download video -> Extract Frames -> Transcribe Audio
    """
    video_url = state.get("video_url")
    video_id = state.get("video_id")

    logger.info(f" processing video : {video_url}")

    local_filename = f"temp_{video_id}.mp4"
    try:
        vi_service = VideoIndexerServices()
        
        # 1. Download
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url , output_path=local_filename)
        else:
            raise Exception("Please provide a valid youtube URL")

        # 2. Extract Frames (Visual Context)
        frames = vi_service.extract_frames(local_path, max_frames=8)
        
        # 3. Transcribe (Audio Context)
        transcript = vi_service.transcribe_audio(local_path)

        # Cleanup
        if os.path.exists(local_path):
            os.remove(local_path)

        logger.info(f"-----[NODE : Indexer] Data Collection Completed-------")
        return {
            "transcript": transcript,
            "ocr_text": [], # No longer needed separately as GPT-4o sees it
            "video_metadata": [{"type": "multimodal_context", "frames_count": len(frames)}],
            "final_status": "success",
            "local_file_path": None, # Frames are already extracted
            "frames": frames # Pass frames to auditor
        }

    except Exception as e:
        logger.error(f"Error in index_video_node: {str(e)}")
        return {
            "error": [str(e)],
            "final_status": "failed",
            "final_message": f"Failed to process video: {str(e)}"
        }



# Compliance 

def auto_content_node( state: VideoAuditState) -> Dict[str , Any]:
    """
    Multimodal Auditor using GPT-4o
    """

    logger.info("----[NODE: Auditor] Analyzing with Multimodal GPT-4o---")

    transcript = state.get("transcript")
    frames = state.get("frames", [])
    
    if not transcript and not frames:
        logger.warning("No data available for analysis")
        return {
            "error": ["No video data available for analysis"],
            "final_status": "failed",
        }

    # Initialize GPT-4o
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=api_key
    )

    # Simplified regulatory context (can be expanded with RAG if needed)
    regulation_rules = "Audit against brand integrity, ensuring no explicit content, correct disclaimers, and premium visual quality."

    system_prompt = f"""
    You are a Brand Compliance Auditor. Your job is to analyze video data (frames and transcript) based on the regulation rules.
    Rules context: {regulation_rules}
    
    Return your response in JSON format:
    {{
        "compliance_result": [
            {{
                "category": "string",
                "description": "string",
                "severity": "Warning/Critical/Info",
                "suggestion": "string"
            }}
        ],
        "final_status": "success/warning/failed",
        "final_report": "Summary of the audit"
    }}
    """

    # Multimodal Content for OpenAI
    content = [
        {"type": "text", "text": f"VIDEO TRANSCRIPT: {transcript}"},
        {"type": "text", "text": "Below are key frames from the video for visual audit:"}
    ]
    
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
        })

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        )
        response_content = response.content
        
        # Extract JSON from potential markdown blocks
        import re
        import json
        if "```" in response_content:
            response_content = re.search(r"```json(.*?)```", response_content, re.DOTALL).group(1).strip()
        
        data = json.loads(response_content)
        return {
            "compliance_result": data.get("compliance_result" , []),
            "final_status": data.get("final_status" , "success"),
            "final_report": data.get("final_report" , "Audit completed successfully."),
        }
    except Exception as e:
        logger.error(f"Error in auditor Multimodal phase: {str(e)}")
        return {
            "error": [str(e)],
            "final_status": "failed",
            "final_report": f"Audit error: {str(e)}",
            "compliance_result": []
        }



        
        

    
