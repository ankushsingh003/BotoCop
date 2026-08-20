import operator

from typing import TypedDict, Annotated , Type , List , Dict , Any , Optional

# schema for the compliance result  
class complianceIssue(TypedDict):
    category: str
    description: str
    severity: str  # ---> Warning
    timestamp: Optional[str]
    

# global graph state

class VideoAuditState(TypedDict):

    # what it takes from the input we provide
    video_url: str
    video_id: str
    domain: Optional[str]
    
    # ingestion and extraction
    local_file_path: Optional[str]
    video_metadata: List[Dict[str, Any]]
    transcript: Optional[str]
    ocr_text: List[str]
    frames: List[str] # Base64 encoded frames


    # analysis
    compliance_result: Annotated[List[complianceIssue], operator.add]
    retrieved_rules: Optional[str]
    rag_sources: Optional[List[str]]
    visual_violations: Optional[List[Dict[str, Any]]]
    visual_status: Optional[str]
    selected_frames: Optional[List[str]]
    merged_report: Optional[dict]


    # final  
    final_status: str
    final_message: str

    # api timeout , system level errors
    error: Annotated[List[str] , operator.add]
    
    