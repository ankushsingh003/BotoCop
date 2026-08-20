import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    @property
    def groq_vision_model(self) -> str:
        return self.GROQ_VISION_MODEL_NAME

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
