'''
Connector between Python and AWS for video analysis.
'''

import os 
import logging 
import yt_dlp 
import boto3

logger = logging.getLogger("video-indexer")

class VideoIndexerService:
    """
    Service for handling video analysis workflows using AWS (S3, Rekognition) 
    and YouTube integration (yt-dlp).
    """
    def __init__(self):
        self.region = os.getenv("REGION", "eu-central-1")
        
        # Mapping AWS credentials from environment variables
        # Note: Using AWS_STORAGE_CONNECTION_STRING as Access Key and 
        # AWS_OPEN_AI_KEY as Secret Key as per current .ENV configuration
        self.aws_access_key = (os.getenv("AWS_STORAGE_CONNECTION_STRING") or "").strip().strip('"').strip("'")
        self.aws_secret_key = (os.getenv("AWS_OPEN_AI_KEY") or "").strip().strip('"').strip("'")
        self.default_bucket = "orchestra-frankfurt"

        if not self.aws_access_key or not self.aws_secret_key:
            logger.warning("AWS credentials not fully found in environment variables.")

        # Initialize AWS Session and Clients
        try:
            # Strip quotes if present in .ENV
            access_key = self.aws_access_key.strip('"').strip("'") if self.aws_access_key else None
            secret_key = self.aws_secret_key.strip('"').strip("'") if self.aws_secret_key else None

            self.session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=self.region
            )
            self.s3 = self.session.client("s3")
            self.rekognition = self.session.client("rekognition")
            self.transcribe = self.session.client("transcribe")
            logger.info(f"AWS clients initialized successfully in {self.region}.")
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {e}")
            raise

    def download_youtube_video(self, url: str, output_path: str = "temp_video.mp4") -> str:
        """Downloads a video from YouTube using yt-dlp with browser cookie authentication."""
        logger.info(f"Downloading YouTube video: {url}")
        
        # Attempt with cookies if available/requested
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_path,
                'quiet': False,
                'no_warnings': True,
                'noplaylist': True,
                'cookiesfrombrowser': ('chrome',),  # Try Chrome cookies
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.warning(f"Download with Chrome cookies failed: {e}. Retrying without cookies.")
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_path,
                'quiet': False,
                'no_warnings': True,
                'noplaylist': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as e2:
                logger.error(f"Failed to download YouTube video even without cookies: {e2}")
                raise
            
            if not os.path.exists(output_path):
                # Fallback if merger failed or output template varied
                if os.path.exists(output_path + ".mp4"):
                    os.rename(output_path + ".mp4", output_path)
                else:
                    raise FileNotFoundError(f"Downloaded file not found at {output_path}")
                    
            return output_path
        except Exception as e:
            logger.error(f"Failed to download YouTube video: {e}")
            raise

    def upload_to_s3(self, local_path: str, video_id: str, bucket: str = None) -> str:
        """Uploads a local file to S3 and returns the S3 URI."""
        bucket = bucket or self.default_bucket
        key = f"videos/{video_id}.mp4"
        
        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
        try:
            self.s3.upload_file(local_path, bucket, key)
            return f"s3://{bucket}/{key}"
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

    def start_video_analysis(self, bucket: str, video_key: str) -> str:
        """Starts a Rekognition label detection job."""
        logger.info(f"Starting label detection for s3://{bucket}/{video_key}")
        try:
            response = self.rekognition.start_label_detection(
                Video={"S3Object": {"Bucket": bucket, "Name": video_key}}
            )
            return response["JobId"]
        except Exception as e:
            logger.error(f"Failed to start Rekognition analysis: {e}")
            raise

    def get_analysis_results(self, job_id: str):
        """Retrieves results of a Rekognition label detection job."""
        try:
            return self.rekognition.get_label_detection(JobId=job_id)
        except Exception as e:
            logger.error(f"Failed to get Rekognition results: {e}")
            return {}

    def start_text_detection(self, bucket: str, video_key: str) -> str:
        """Starts a Rekognition text detection job."""
        logger.info(f"Starting text detection for s3://{bucket}/{video_key}")
        try:
            response = self.rekognition.start_text_detection(
                Video={"S3Object": {"Bucket": bucket, "Name": video_key}}
            )
            return response["JobId"]
        except Exception as e:
            logger.error(f"Failed to start Rekognition text detection: {e}")
            raise

    def get_text_detection(self, job_id: str):
        """Retrieves results of a Rekognition text detection job."""
        try:
            return self.rekognition.get_text_detection(JobId=job_id)
        except Exception as e:
            logger.error(f"Failed to get Rekognition text results: {e}")
            return {}

    def get_insights(self, job_id: str):
        """Alias for get_analysis_results to maintain compatibility with nodes.py."""
        return self.get_analysis_results(job_id)

    def start_transcription_job(self, bucket: str, video_key: str, job_name: str) -> str:
        """Starts an AWS Transcribe job for a video file."""
        video_uri = f"s3://{bucket}/{video_key}"
        logger.info(f"Starting transcription job {job_name} for {video_uri}")
        
        try:
            # delete existing job if name collision occurs
            try:
                self.transcribe.delete_transcription_job(TranscriptionJobName=job_name)
            except:
                pass

            self.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': video_uri},
                MediaFormat='mp4',
                LanguageCode='en-US'
            )
            return job_name
        except Exception as e:
            logger.error(f"Failed to start transcription job: {e}")
            raise

    def get_transcription_text(self, job_name: str) -> str:
        """Retrieves the transcript text from a finished job."""
        try:
            response = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            status = response['TranscriptionJob']['TranscriptionJobStatus']
            
            if status == 'COMPLETED':
                transcript_url = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                # Download transcript JSON
                import requests
                transcript_data = requests.get(transcript_url).json()
                return transcript_data.get('results', {}).get('transcripts', [{}])[0].get('transcript', "")
            elif status == 'FAILED':
                logger.error(f"Transcription job failed: {response['TranscriptionJob'].get('FailureReason')}")
                return ""
            else:
                logger.info(f"Transcription job {job_name} still in progress (status: {status})")
                return ""
        except Exception as e:
            logger.error(f"Error fetching transcription results: {e}")
            return ""

    def extract_data(self, rek_insights: dict, transcript_text: str = "", text_insights: dict = None) -> dict:
        """Extracts and cleans relevant data from Rekognition insights (Labels & Text) and Transcribe results."""
        logger.info("Extracting combined data from Rekognition and Transcribe")
        
        # Labels
        labels = rek_insights.get("Labels", []) if rek_insights else []
        clean_labels = [
            {
                "name": label.get("Label", {}).get("Name"),
                "confidence": label.get("Label", {}).get("Confidence"),
                "timestamp": label.get("Timestamp")
            }
            for label in labels
        ]
        
        # OCR Text
        ocr_lines = []
        if text_insights and "TextDetections" in text_insights:
            for detection in text_insights["TextDetections"]:
                if detection.get("Type") == "LINE":
                    text = detection.get("TextDetection", {}).get("DetectedText")
                    if text and text not in ocr_lines:
                        ocr_lines.append(text)
        
        # Check JobStatus
        job_status = rek_insights.get("JobStatus", "IN_PROGRESS") if rek_insights else "FAILED"
        if text_insights and text_insights.get("JobStatus") == "FAILED":
            job_status = "FAILED"
            
        return {
            "transcript": transcript_text or "",
            "ocr_text": ocr_lines,
            "video_metadata": clean_labels,
            "final_status": "success" if job_status == "SUCCEEDED" else "failed" if job_status == "FAILED" else "processing"
        }

    def _empty_response(self, message: str) -> dict:
        return {
            "transcript": "",
            "ocr_text": [],
            "video_metadata": [],
            "final_status": "failed",
            "message": message
        }

# Alias for backward compatibility with existing code (e.g., nodes.py)
VideoIndexerServices = VideoIndexerService
