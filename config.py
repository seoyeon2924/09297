"""
Application configuration.
Reads from .env file and provides sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralized application settings."""

    # ── Paths ──
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
    CHROMA_HOST = "localhost"
    CHROMA_PORT = 8000
    SQLITE_URL = f"sqlite:///{DATA_DIR / 'compliance.db'}"

    # ── OpenAI (Stage 2+) ──
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    # ── RAG Parameters ──
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── Mock Mode ──
    # When True, LLM/Embedding calls return dummy data (no API key needed).
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "true").lower() == "true"

    def __init__(self) -> None:
        """Ensure required directories exist at startup."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "chroma_db").mkdir(parents=True, exist_ok=True)

    # ── Cohere (Reranker) ──
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

    # ── FastAPI Backend ──
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "localhost")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8001"))

    @property
    def FASTAPI_BASE_URL(self) -> str:
        return f"http://{self.FASTAPI_HOST}:{self.FASTAPI_PORT}"


# Singleton – directories are created on first import.
settings = Settings()
