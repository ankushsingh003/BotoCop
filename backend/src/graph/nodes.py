import json
import os
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
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

def classify_domain_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Classify the video transcript to route it to domain-specific rulebooks.
    """
    transcript = state.get("transcript") or ""
    if not transcript.strip():
        logger.info("No transcript found, defaulting domain to 'general'")
        return {"domain": "general"}

    # Initialize Groq
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    
    logger.info(f"Classifying domain using: {model_name}")
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.0,
        groq_api_key=api_key
    )

    system_prompt = "You are a specialized compliance classification agent."
    content = f"""
    Classify the video transcript into one of these exact categories:
    - finance (if the video discusses investment, crypto, stocks, trading, insurance, banking, or wealth advice)
    - healthcare (if the video discusses medicine, health treatments, pharmaceuticals, clinical services, or medical advice)
    - food_and_beverage (if the video discusses recipes, restaurants, packaged food, dietary products, nutrition, or beverages)
    - general (if it fits none of the above, or is generic brand advertising / lifestyle content)

    Transcript:
    "{transcript[:4000]}"

    Response MUST be exactly one of the following words: [finance, healthcare, food_and_beverage, general]. Do not include any punctuation, markdown formatting, or extra text.
    """

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        )
        domain = response.content.strip().lower()
        # Clean any accidental markdown or punctuation
        domain = re.sub(r'[^a-z_]', '', domain)
        
        valid_domains = {"finance", "healthcare", "food_and_beverage", "general"}
        if domain not in valid_domains:
            logger.warning(f"Invalid domain classified: '{domain}'. Defaulting to 'general'.")
            domain = "general"
            
        logger.info(f"Classified video domain as: '{domain}'")
        return {"domain": domain}
    except Exception as e:
        logger.error(f"Error classifying domain: {e}. Defaulting to 'general'.")
        return {"domain": "general"}

class ComplianceIssueModel(BaseModel):
    category: str = Field(description="The compliance category, e.g. Financial_Compliance, Healthcare_Compliance, or General")
    description: str = Field(description="Detailed finding or violation description based on the transcript and guidelines")
    severity: str = Field(description="Severity level of the issue: Warning, Critical, or Info")
    suggestion: str = Field(description="Actionable recommendation/suggestion to resolve the issue")
    timestamp: Optional[str] = Field(None, description="Timestamp of the issue in HH:MM:SS format if applicable, or None")

class AuditReportModel(BaseModel):
    compliance_result: List[ComplianceIssueModel] = Field(default_factory=list, description="List of all detected compliance issues")
    final_status: str = Field(description="Overall status of the audit: success, warning, or failed")
    final_report: str = Field(description="Summary narrative report of the multi-domain compliance audit")

# Compliance 

def run_domain_audit(state: VideoAuditState, domain: str) -> Dict[str, Any]:
    """
    Core auditing node function invoked conditionally based on classified domain.
    """
    from backend.src.rag.retriever import RuleRetriever, sanitize_text
    retriever = RuleRetriever()

    logger.info(f"----[NODE: {domain.upper()} Auditor] Analyzing with Groq Llama 3.3---")

    transcript = state.get("transcript")
    
    if not transcript:
        logger.warning("No transcript available for analysis")
        return {
            "error": [f"No transcription data available for {domain} audit. Audit cannot continue."],
            "final_status": "failed",
            "retrieved_rules": None,
            "rag_sources": [],
        }

    # Initialize Groq
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    
    logger.info(f"Using Groq model: {model_name}, API Key prefix: {api_key[:10] if api_key else 'None'}")
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.0,
        groq_api_key=api_key
    )
    
    transcript = sanitize_text(transcript)

    # Retrieve relevant rules from the domain-specific collection using the transcript
    logger.info(f"Retrieving compliance rules for domain '{domain}' via MMR RAG...")
    try:
        retrieved_rules = retriever.retrieve(transcript[:2000], domain=domain)
        import re
        rag_sources = re.findall(r'\[Source: ([^\]]+)\]', retrieved_rules)
    except Exception as e:
        logger.error(f"Failed to retrieve rules: {e}. Falling back to empty rules.")
        retrieved_rules = "No matching rules found."
        rag_sources = []

    import uuid
    cache_buster = str(uuid.uuid4())
    system_prompt = f"Session ID: {cache_buster}. You are a Professional Compliance Auditor specializing in {domain.upper()} regulations. Analyze the provided transcript against the retrieved guidelines and return a structured compliance report."

    content = f"""Request ID: {cache_buster}
Analyze the following video transcript against the regulatory guidelines.

<guidelines>
{retrieved_rules}
</guidelines>

<transcript>
{transcript}
</transcript>

You MUST perform the compliance audit and output ONLY a valid JSON object matching the schema below. Do not include any preamble, introduction, markdown blocks, or explanation. Start your response directly with the JSON object.
Do NOT use any external knowledge or retrieve guidelines from your training data. Base your analysis ONLY on the guidelines listed under the <guidelines> tag. Do not analyze any other guidelines.
Analyze the EXACT transcript provided under the <transcript> tag. Do not make up, hallucinate, or analyze a different transcript.

JSON Schema:
{{
    "compliance_result": [
        {{
            "category": "Financial_Compliance/Healthcare_Compliance/Food_Compliance/General",
            "description": "Specific finding in the transcript",
            "severity": "Warning/Critical/Info",
            "suggestion": "How to fix the violation",
            "timestamp": "HH:MM:SS format or null"
        }}
    ],
    "final_status": "success/warning/failed",
    "final_report": "Summary of the compliance audit"
}}"""

    logger.info(f"System Prompt sent to LLM:\n{system_prompt}")
    logger.info(f"User Content sent to LLM:\n{content}")

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        )
        response_content = response.content
        logger.info(f"Raw LLM Response: {response_content}")
        
        # Robustly extract the outermost JSON object
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = response_content[start_idx:end_idx+1]
        else:
            json_str = response_content
        
        logger.info(f"Extracted JSON for loading: {json_str}")
        data = json.loads(json_str)
        
        # Validate using Pydantic model
        report = AuditReportModel(**data)
        
        compliance_result = []
        for issue in report.compliance_result:
            issue_dict = {
                "category": issue.category,
                "description": issue.description,
                "severity": issue.severity,
                "suggestion": issue.suggestion,
            }
            if issue.timestamp:
                issue_dict["timestamp"] = issue.timestamp
            compliance_result.append(issue_dict)
            
        return {
            "compliance_result": compliance_result,
            "final_status": report.final_status,
            "final_report": report.final_report,
            "retrieved_rules": retrieved_rules,
            "rag_sources": rag_sources,
        }
    except Exception as e:
        logger.error(f"Error in Multi-Domain Auditor Groq phase: {str(e)}")
        return {
            "error": [str(e)],
            "final_status": "failed",
            "final_report": f"Audit error: {str(e)}",
            "compliance_result": [],
            "retrieved_rules": retrieved_rules,
            "rag_sources": rag_sources,
        }

def auto_content_node(state: VideoAuditState) -> Dict[str, Any]:
    domain = state.get("domain") or "general"
    return run_domain_audit(state, domain)


def visual_compliance_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Visual Compliance Auditor using Groq Vision Model (Llama 4 Scout)
    Analyzes extracted frames for on-screen text, disclaimers, logo placement, etc.
    """
    logger.info("----[NODE: Visual Auditor] Analyzing video frames with Groq Vision---")

    frames = state.get("frames")
    if not frames:
        logger.info("No frames available for visual analysis.")
        return {
            "visual_violations": [],
            "visual_status": "success",
            "selected_frames": [],
        }

    # Initialize Groq Vision Model
    api_key = os.getenv("GROQ_API_KEY")
    from backend.src.config import settings
    # Default to the vision model configured in settings
    model_name = getattr(settings, "groq_vision_model", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    logger.info(f"Using Groq Vision model: {model_name}")
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.0,
        groq_api_key=api_key
    )

    # Sample at most 5 evenly spaced frames using the selection utility
    from backend.src.services.video_index import select_keyframes
    sampled_frames = select_keyframes(frames, n=5)

    logger.info(f"Sending {len(sampled_frames)} frames to Groq Vision API...")

    # Construct the message content
    message_content = [
        {
            "type": "text",
            "text": "You are a professional brand compliance and regulatory auditor. Analyze the provided sequence of video frames to identify compliance issues."
        }
    ]

    for idx, frame in enumerate(sampled_frames):
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}"
            }
        })

    import uuid
    cache_buster = str(uuid.uuid4())
    system_prompt = f"Session ID: {cache_buster}. You are a Professional Visual Compliance Auditor. Analyze the provided video frames for compliance issues and return a structured report."

    human_prompt = f"""Request ID: {cache_buster}
Carefully inspect the provided sequence of video frames. Check the following:
1. On-screen disclaimer/disclosure text (is it present, legible, correctly timed/placed, or missing when required?)
2. Logo/brand placement compliance (e.g. correct logo version, proper styling, prominence)
3. Any text overlays making claims (price, health, guarantees, or financial statements)
4. Inappropriate, restricted, or unsafe visual content

You MUST perform the compliance audit and output ONLY a valid JSON object matching the schema below. Do not include any preamble, introduction, markdown blocks, or explanation. Start your response directly with the JSON object.

JSON Schema:
{{
    "compliance_result": [
        {{
            "category": "Financial_Compliance/Healthcare_Compliance/General",
            "description": "Specific visual finding or violation description based on the frames",
            "severity": "Warning/Critical/Info",
            "suggestion": "How to fix the visual compliance issue",
            "timestamp": "HH:MM:SS format or null"
        }}
    ],
    "final_status": "success/warning/failed",
    "final_report": "Summary narrative report of the visual compliance audit"
}}"""

    message_content.append({
        "type": "text",
        "text": human_prompt
    })

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_content)
        ])
        response_content = response.content
        logger.info(f"Raw Vision LLM Response: {response_content}")

        # Robustly extract the outermost JSON object
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = response_content[start_idx:end_idx+1]
        else:
            json_str = response_content

        data = json.loads(json_str)

        # Validate using Pydantic model
        report = AuditReportModel(**data)

        compliance_result = []
        for issue in report.compliance_result:
            issue_dict = {
                "category": issue.category,
                "description": issue.description,
                "severity": issue.severity,
                "suggestion": issue.suggestion,
            }
            if issue.timestamp:
                issue_dict["timestamp"] = issue.timestamp
            compliance_result.append(issue_dict)

        return {
            "visual_violations": compliance_result,
            "visual_status": report.final_status,
            "selected_frames": sampled_frames,
        }
    except Exception as e:
        logger.error(f"Error in visual_compliance_node: {str(e)}")
        return {
            "error": [str(e)],
            "visual_violations": [],
            "visual_status": "failed",
            "selected_frames": [],
        }


def merge_results_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Consolidates the audio/text compliance results and the visual compliance violations
    into a single unified report. Computes the overall final_status and final_message.
    """
    logger.info("----[NODE: Result Merger] Consolidating audio and visual audit results---")

    audio_status = state.get("final_status") or "success"
    visual_status = state.get("visual_status") or "success"

    audio_violations = state.get("compliance_result") or []
    visual_violations = state.get("visual_violations") or []

    # Determine overall status: if either fails, the whole audit fails.
    if audio_status == "failed" or visual_status == "failed":
        overall_status = "failed"
    elif audio_status == "warning" or visual_status == "warning":
        overall_status = "warning"
    else:
        overall_status = "success"

    # Construct the final unified report / message
    summary_parts = []
    summary_parts.append(f"Overall Audit Status: {overall_status.upper()}")
    
    total_audio = len(audio_violations)
    total_visual = len(visual_violations)
    summary_parts.append(f"Audio/Text Violations Found: {total_audio}")
    summary_parts.append(f"Visual Frame Violations Found: {total_visual}")

    if overall_status == "success":
        summary_parts.append("Compliance Audit completed successfully. No major brand or regulatory violations detected.")
    else:
        summary_parts.append("Action required: brand or regulatory compliance issues were identified in the audio or visual content.")

    final_message = "\n".join(summary_parts)

    logger.info(f"Consolidated overall status: {overall_status}")
    logger.info(f"Consolidated final message: {final_message}")

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

    audio_mapped = [
        {
            "rule": item.get("category") or "General",
            "severity": map_severity(item.get("severity")),
            "description": item.get("description") or "",
            "source": "audio"
        }
        for item in audio_violations
    ]
    visual_mapped = [
        {
            "rule": item.get("category") or "General",
            "severity": map_severity(item.get("severity")),
            "description": item.get("description") or "",
            "source": "visual"
        }
        for item in visual_violations
    ]

    status = "pass" if overall_status == "success" else "fail"

    merged_report = {
        "status": status,
        "audio": audio_mapped,
        "visual": visual_mapped,
        "final_status": overall_status,
        "final_message": final_message,
        "audio_violations_count": len(audio_violations),
        "visual_violations_count": len(visual_violations),
        "total_violations_count": len(audio_violations) + len(visual_violations),
    }

    # Return the merged fields. 
    # Returning the visual_violations in compliance_result will append them to the existing list via the graph reducer.
    return {
        "compliance_result": visual_violations,
        "final_status": overall_status,
        "final_message": final_message,
        "merged_report": merged_report,
    }




        
        

    
