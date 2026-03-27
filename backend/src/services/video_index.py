'''
Connector between Python and AI Services for video analysis.
Replaces OpenAI with Groq (Whisper).
'''

import os 
import logging 
import yt_dlp 
import cv2
import base64
from typing import List, Dict, Any
from groq import Groq

logger = logging.getLogger("video-indexer")

class VideoIndexerService:
    """
    Service for handling video analysis workflows using Groq
    and YouTube integration (yt-dlp).
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def download_youtube_video(self, url: str, output_path: str = "temp_video.mp4") -> str:
        """Downloads a video from YouTube using yt-dlp."""
        logger.info(f"Downloading YouTube video: {url}")
        
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
            
            if not os.path.exists(output_path):
                # Fallback check
                if os.path.exists(output_path + ".mp4"):
                    os.rename(output_path + ".mp4", output_path)
                else:
                    raise FileNotFoundError(f"Downloaded file not found at {output_path}")
                    
            return output_path
        except Exception as e:
            logger.error(f"Failed to download YouTube video: {e}")
            raise

    def extract_frames(self, video_path: str, max_frames: int = 10) -> List[str]:
        """Extracts high-quality key frames from a video for visual analysis."""
        logger.info(f"Extracting {max_frames} frames from {video_path}")
        base64_frames = []
        video = cv2.VideoCapture(video_path)
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error("No frames found in video.")
            return []

        interval = max(1, total_frames // max_frames)
        
        for i in range(0, total_frames, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = video.read()
            if not success:
                break
            
            # Convert frame to jpg and then to base64
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            
            if len(base64_frames) >= max_frames:
                break
        
        video.release()
        logger.info(f"Successfully extracted {len(base64_frames)} frames.")
        return base64_frames

    def transcribe_audio(self, video_path: str) -> str:
        """Transcribes the audio from a video file using Groq Whisper."""
        if not self.client:
            raise ValueError("Groq client not initialized. Missing API key.")
        
        logger.info(f"Transcribing audio from {video_path} using Groq Whisper")
        try:
            with open(video_path, "rb") as audio_file:
                # Groq Whisper API call
                transcript = self.client.audio.transcriptions.create(
                    file=(os.path.basename(video_path), audio_file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            return str(transcript)
        except Exception as e:
            logger.error(f"Groq Transcription failed: {e}")
            return ""

    def extract_data(self, transcript_text: str, frames: List[str] = None) -> dict:
        return {
            "transcript": transcript_text or "",
            "frames_count": len(frames) if frames else 0,
            "final_status": "success" if transcript_text else "failed"
        }

VideoIndexerServices = VideoIndexerService
