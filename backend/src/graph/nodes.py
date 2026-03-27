import json
import os
import logging
import re
from typing import Dict, Any, List
from langchain_groq import ChatGroq
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
    Download video -> Extract Frames -> Transcribe Audio (Groq)
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

        # 2. Extract Frames 
        frames = vi_service.extract_frames(local_path, max_frames=8)
        
        # 3. Transcribe (Audio Context - Groq Whisper)
        transcript = vi_service.transcribe_audio(local_path)

        # Cleanup
        if os.path.exists(local_path):
            os.remove(local_path)

        logger.info(f"-----[NODE : Indexer] Data Collection Completed-------")
        return {
            "transcript": transcript,
            "ocr_text": [],
            "video_metadata": [{"type": "text_audit_fallback", "frames_count": len(frames)}],
            "final_status": "success",
            "local_file_path": None,
            "frames": frames
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
    Auditor using Groq (Llama 3.3 Multi-Domain Edition)
    Specialized for Financial and Healthcare Compliance.
    """

    logger.info("----[NODE: Auditor] Analyzing with Multi-Domain Groq Llama 3.3---")

    transcript = state.get("transcript")
    
    if not transcript:
        logger.warning("No transcript available for analysis")
        return {
            "error": ["No transcription data available for analysis. Audit cannot continue."],
            "final_status": "failed",
        }

    # Initialize Groq
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.0,
        groq_api_key=api_key
    )

    # Multi-Domain Compliance Rules
    regulation_rules = """
    DOMAIN: FINANCIAL SERVICES
    1. PROHIBITED: Guaranteed returns (e.g. "Risk-free profits").
    2. REQUIRED: "Not financial advice" or "For educational purposes" disclaimer.
    3. REQUIRED: High-risk/Potential Loss warnings for trading mentions.
    
    DOMAIN: HEALTHCARE
    1. REQUIRED: "Always consult a doctor before starting any treatment" disclaimer.
    2. PROHIBITED: "Miracle Cure" or "Zero side effects" claims.
    3. CHECK: Misleading "Clinically Proven" language without scientific context.
    4. REQUIRED: FDA/Regulatory disclaimers if medical products are mentioned.
    """

    system_prompt = f"""
    You are a Professional Compliance Auditor specializing in Financial Services and Healthcare.
    Analyze the video transcript based on these rules:
    {regulation_rules}
    
    Your task:
    1. Identify if the content is Financial, Healthcare, or General.
    2. Apply the relevant domain rules.
    3. Flag missing disclaimers as 'Critical' or 'Warning'.
    4. Flag 'Guaranteed' profit or 'Miracle' cures as 'Critical' violations.
    
    Return your response in JSON format:
    {{
        "compliance_result": [
            {{
                "category": "Financial_Compliance/Healthcare_Compliance/General",
                "description": "Specific finding in the transcript",
                "severity": "Warning/Critical/Info",
                "suggestion": "How to fix the violation"
            }}
        ],
        "final_status": "success/warning/failed",
        "final_report": "Summary of the multi-domain audit"
    }}
    """

    content = f"VIDEO TRANSCRIPT FOR AUDIT: {transcript}"

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        )
        response_content = response.content
        
        # Extract JSON from potential markdown blocks
        if "```" in response_content:
            match = re.search(r"```json(.*?)```", response_content, re.DOTALL)
            if match:
                response_content = match.group(1).strip()
            else:
                response_content = response_content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(response_content)
        return {
            "compliance_result": data.get("compliance_result" , []),
            "final_status": data.get("final_status" , "success"),
            "final_report": data.get("final_report" , "Multi-domain compliance audit completed."),
        }
    except Exception as e:
        logger.error(f"Error in Multi-Domain Auditor Groq phase: {str(e)}")
        return {
            "error": [str(e)],
            "final_status": "failed",
            "final_report": f"Audit error: {str(e)}",
            "compliance_result": []
        }



        
        

    
