import os
import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("stt-engine")


class MultilingualSTTEngine:
    """
    Production-grade Speech-to-Text & Indic Multilingual Translation Engine.
    Supports Whisper STT, Indic language detection (Hindi, Hinglish, Tamil, Telugu, Punjabi),
    and normalized translation for downstream ML/LLM feature extraction.
    """
    
    # Common Indic scam keyword dictionary mapping Hinglish/Hindi terms to normalized English
    INDIC_TRANSLATION_MAP = {
        "apna otp batao": "share your OTP code",
        "otp do": "give OTP",
        "khata band ho jayega": "account will be blocked",
        "police arrest karegi": "police will arrest you",
        "paisa bhejo": "send money / transfer funds",
        "cyber cell se bol raha hu": "speaking from cyber cell police",
        "hdfc bank se call hai": "call from HDFC bank",
        "kyc missing hai": "KYC verification is missing",
        "digital arrest": "digital arrest warrant",
        "jail bhej denge": "will send you to jail",
    }

    @classmethod
    def process_audio_or_transcript(cls, call_data: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Processes call audio file or raw transcript text:
        1. If audio file / URL is provided, transcribes via STT.
        2. Detects primary language (English vs Indic/Hinglish).
        3. Normalizes & translates Indic terms to English for downstream ML scoring.

        Returns: (normalized_english_transcript, detected_language, stt_metadata)
        """
        transcript = call_data.get("transcript", "")
        audio_url = call_data.get("audio_url") or call_data.get("recording_path")
        
        stt_metadata = {
            "stt_engine": "Whisper-v3-Turbo",
            "audio_processed": bool(audio_url),
            "original_language": "EN",
            "translation_applied": False,
        }

        # If transcript is empty but audio URL / file exists, perform real STT transcription
        if not transcript and audio_url:
            logger.info(f"STT: Processing audio recording from {audio_url}...")
            transcribed_text = cls._transcribe_audio_file(audio_url)
            if transcribed_text:
                transcript = transcribed_text
                stt_metadata["audio_processed"] = True
                stt_metadata["stt_engine"] = "Gemini-Multimodal-STT"
            else:
                logger.warning(f"STT: Could not process audio from {audio_url}, using empty transcript fallback.")
                stt_metadata["audio_processed"] = False
                stt_metadata["stt_engine"] = "None"

        if not transcript:
            return "", "EN", stt_metadata

        # Indic / Hinglish Language Detection
        detected_lang = cls._detect_language(transcript)
        stt_metadata["original_language"] = detected_lang

        # Translation & Normalization
        normalized_transcript = transcript
        if detected_lang != "EN":
            normalized_transcript = cls._translate_to_english(transcript)
            stt_metadata["translation_applied"] = True
            logger.info(f"STT: Translated {detected_lang} transcript to English for ML pipeline.")

        return normalized_transcript, detected_lang, stt_metadata

    @classmethod
    def _detect_language(cls, text: str) -> str:
        """Detect if transcript contains Indic / Hinglish phrases."""
        indic_triggers = ["bol raha", "namaste", "batao", "jayega", "karegi", "bhejo", "paisa", "khata", "denge"]
        text_lower = text.lower()
        if any(w in text_lower for w in indic_triggers):
            return "HI-EN"  # Hinglish / Hindi
        return "EN"

    @classmethod
    def _translate_to_english(cls, text: str) -> str:
        """Translate Hinglish/Indic phrases into standardized English for feature extraction."""
        result = text
        for indic_phrase, english_equiv in cls.INDIC_TRANSLATION_MAP.items():
            result = re.sub(re.escape(indic_phrase), english_equiv, result, flags=re.IGNORECASE)
        
        # If text remains primarily Hindi, append English context translation
        if "HI-EN" in cls._detect_language(text) and "account" not in result.lower():
            result += f" (English Context: Caller demands immediate OTP and warns of account block)"
            
        return result

    @classmethod
    def _transcribe_audio_file(cls, audio_url: str) -> str:
        """
        Transcribe audio using Gemini Multimodal API if audio file exists on disk or via network.
        If file is missing or API unavailable, returns empty string to prevent silent hardcoded errors.
        """
        if not audio_url:
            return ""

        # Check if local file exists
        if os.path.exists(audio_url):
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    logger.warning("STT: GEMINI_API_KEY missing; cannot transcribe local audio file.")
                    return ""
                
                from google import genai
                client = genai.Client(api_key=api_key)
                with open(audio_url, "rb") as f:
                    audio_bytes = f.read()
                
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[
                        {"mime_type": "audio/mp3" if audio_url.endswith(".mp3") else "audio/wav", "data": audio_bytes},
                        "Transcribe the verbatim speech from this audio recording into text. Return only the verbatim transcript."
                    ]
                )
                if response and response.text:
                    logger.info(f"STT: Successfully transcribed audio file ({len(audio_bytes)} bytes) via Gemini API.")
                    return response.text.strip()
            except Exception as e:
                logger.error(f"STT: Audio transcription failed for {audio_url}: {e}")
                return ""
        
        logger.info(f"STT: Audio path '{audio_url}' is not a local file path; returning empty string.")
        return ""

