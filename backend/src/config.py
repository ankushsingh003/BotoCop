import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-3.6-flash"
    GEMINI_VISION_MODEL_NAME: str = "gemini-3.6-flash"

    
    @property
    def gemini_vision_model(self) -> str:
        return self.GEMINI_VISION_MODEL_NAME

    RAG_DATA_DIR: str = str(Path(__file__).resolve().parent.parent / "data")
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "brand-compliance-rules"
    PORT: int = 8000
    RAG_MMR_LAMBDA: float = 0.5
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
