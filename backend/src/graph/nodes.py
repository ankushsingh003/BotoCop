import json
import os
import logging
import re
import boto3
from typing import Dict , Any , List

from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage , SystemMessage
from opensearchpy import AWSV4SignerAuth
import requests

from backend.src.graph.state import VideoAuditState , complianceIssue

from backend.src.services.video_index import VideoIndexerServices

logger = logging.getLogger("brand-compliance-rules")
logging.basicConfig(level=logging.INFO)

# INDEXER

def index_video_node( state: VideoAuditState) -> Dict[str , Any]:
    """
    download -> stores blob storage -> extract insights
    """
    video_url = state.get("video_url")
    video_id = state.get("video_id")

    logger.info(f" processing video : {video_url}")

    local_filename = "temp_audit_video.mp4"
    try:
        vi_service = VideoIndexerServices()
        # download part
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url , output_path=local_filename)

        else:
            raise Exception("Please provide a valid youtube URL")
        # upload part
        vi_service.upload_to_s3(local_path, video_id)

        # start Rekognition analysis (Labels)
        job_id = vi_service.start_video_analysis("orchestra-frankfurt", f"videos/{video_id}.mp4")
        
        # start Rekognition analysis (Text)
        text_job_id = vi_service.start_text_detection("orchestra-frankfurt", f"videos/{video_id}.mp4")
        
        # start Transcribe analysis
        transcribe_job_name = f"audit_{video_id}"
        vi_service.start_transcription_job("orchestra-frankfurt", f"videos/{video_id}.mp4", transcribe_job_name)

        logger.info(f"Analysis started. Rekognition Labels ID: {job_id}, Text ID: {text_job_id}, Transcribe ID: {transcribe_job_name}")

        # cleaning
        if os.path.exists(local_path):
            os.remove(local_path)

        # Polling for results (simplified for simulation)
        import time
        max_retries = 30
        transcript_text = ""
        raw_insights = {}
        text_insights = {}
        print("Polling for analysis results (this may take a minute)...")
        for i in range(max_retries):
            time.sleep(10) # wait 10 seconds between polls
            
            # Check Rekognition Labels
            if not raw_insights or raw_insights.get("JobStatus") != "SUCCEEDED":
                raw_insights = vi_service.get_insights(job_id)
            
            # Check Rekognition Text
            if not text_insights or text_insights.get("JobStatus") != "SUCCEEDED":
                text_insights = vi_service.get_text_detection(text_job_id)

            # Check Transcribe
            if not transcript_text:
                transcript_text = vi_service.get_transcription_text(transcribe_job_name)
            
            # Fail early if any job fails
            label_status = raw_insights.get("JobStatus")
            text_status = text_insights.get("JobStatus")
            
            if label_status == "FAILED" or text_status == "FAILED":
                logger.error(f"Analysis job failed. Labels: {label_status}, Text: {text_status}")
                break

            if (label_status == "SUCCEEDED") and \
               (text_status == "SUCCEEDED") and \
               transcript_text:
                print("Analysis completed successfully.")
                break
            
            print(f"Still processing... (Attempt {i+1}/{max_retries})")

        # extract
        clean_data = vi_service.extract_data(raw_insights, transcript_text, text_insights=text_insights)
        logger.info(f"-----[NODE : Indexer] Extraction Completed-------")
        return clean_data

    except Exception as e:
        logger.error(f"Error in index_video_node: {str(e)}")
        
        return {
            "error": [str(e)],
            "final_status": "failed",
            "final_message": f"Failed to index video: {str(e)}",
            "transcript": "",
            "ocr_text": [],
            "video_metadata": [],
            "compliance_result": []
        }



# Compliance 

def auto_content_node( state: VideoAuditState) -> Dict[str , Any]:
    """
    RAG
    """

    logger.info("----[NODE: Auditor] querying the knowledges based and LLM---")

    transcript = state.get("transcript")
    if not transcript:
        logger.warning("No transcript available ")
        return {
            "error": ["No transcript available for analysis"],
            "final_status": "failed",
            "final_message": "Failed to index video",
            "transcript": "",
            "ocr_text": [],
            "video_metadata": [],
            "compliance_result": []
        }

    # initialising clients 
    access_key = (os.getenv("AWS_STORAGE_CONNECTION_STRING") or "").strip().strip('"').strip("'")
    secret_key = (os.getenv("AWS_OPEN_AI_KEY") or "").strip().strip('"').strip("'")
    region = os.getenv("REGION", "eu-central-1")

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    llm = ChatBedrock(
        model_id=os.getenv("AWS_OPENAI_MODEL"),
        model_kwargs={"temperature": 0.0},
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    embeddings = BedrockEmbeddings(
        model_id=os.getenv("AWS_OPENAI_EMBEDDING_DEPLOYMENT"),
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # ---- Vector Store (OpenSearch) ----
    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, region)

    docs = []
    try:
        logger.info(f"Connecting to OpenSearch at {os.getenv('AWS_SEARCH_ENDPOINT')}...")
        vector_store = OpenSearchVectorSearch(
            opensearch_url=os.getenv("AWS_SEARCH_ENDPOINT"),
            index_name=os.getenv("AWS_SEARCH_INDEX_NAME"),
            embedding_function=embeddings,
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=requests.Session
        )
        
        # rag retrieval
        ocr_text = state.get("ocr_text" , [])
        query_text = f"{transcript} {' '.join(ocr_text)}"
        docs = vector_store.similarity_search(query_text, k=3, timeout=10)
        logger.info(f"Successfully retrieved {len(docs)} documents.")
    except Exception as e:
        logger.warning(f"Knowledge base search failed: {e}. Falling back to internal audit model.")
        docs = []

    regulation_rules = "\n\n".join([doc.page_content for doc in docs]) if docs else "No specific regulatory context found. Audit against general brand integrity."

    system_prompt = f"""
    You are a Brand Compliance Auditor. Your job is to analyze video data based on the provided regulation rules.
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
        "final_report": "Summary"
    }}
    """

    user_message = f"""
    VIDEO_METADATA : {state.get("video_metadata")}
    TRANSCRIPT : {state.get("transcript")}
    OCR_TEXT : {state.get("ocr_text")}
    """
    
    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt) , HumanMessage(content=user_message)]
        )
        response_content = response.content
        
        if "```" in response_content:
            response_content = re.search(r"```json(.*?)```" , response_content , re.DOTALL).group(1).strip()
        
        data = json.loads(response_content)
        return {
            "compliance_result": data.get("compliance_result" , []),
            "final_status": data.get("final_status" , "success"),
            "final_report": data.get("final_report" , "Audit completed successfully."),
        }
    except Exception as e:
        logger.error(f"Error in auditor LLM phase: {str(e)}")
        return {
            "error": [str(e)],
            "final_status": "failed",
            "final_report": f"Audit error: {str(e)}",
            "compliance_result": []
        }



        
        

    
