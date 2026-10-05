"""Configuration module for Indian Legal RAG Pipeline.

Loads configuration from environment variables (.env) with robust validation,
production defaults, and automatic filesystem initialization.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Locate Project Root and load .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_val: Optional[str], default_rel: str) -> Path:
    """Resolve directory path against PROJECT_ROOT if relative."""
    raw = env_val.strip() if env_val else default_rel
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class Settings(BaseModel):
    """Production settings and hyperparameters for Indian Legal RAG."""

    # Gemini API Configuration
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )
    gemini_embedding_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    )

    # Vector Store & Collection
    chroma_db_dir: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("CHROMA_DB_DIR"), "chromadb_store")
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "indian_legal_precedents")
    )

    # Hierarchical Legal Chunking Hyperparameters
    parent_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("PARENT_CHUNK_SIZE", "1800"))
    )
    child_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_SIZE", "450"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "80"))
    )

    # Reranker Model
    reranker_model: str = Field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )

    # Data Storage Directories
    parent_store_dir: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("PARENT_STORE_DIR"), "data/parent_store")
    )
    raw_data_dir: Path = Field(
        default_factory=lambda: _resolve_path(
            os.getenv("RAW_DATA_DIR"),
            "data/raw_judgments",
        )
    )
    processed_data_dir: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("PROCESSED_DATA_DIR"), "data/processed")
    )
    benchmarks_dir: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("BENCHMARKS_DIR"), "data/benchmarks")
    )

    @classmethod
    def validate(cls, instance: Optional["Settings"] = None) -> "Settings":
        """Validate critical configuration and ensure all storage directories exist on disk."""
        cfg = instance or cls()

        # Validate Gemini API Key presence
        if not cfg.gemini_api_key or cfg.gemini_api_key in ("your_gemini_api_key_here", ""):
            raise ValueError(
                "GEMINI_API_KEY is not set or set to placeholder in .env. "
                "Please configure a valid Google Gemini API key."
            )

        # Validate chunking parameters
        if cfg.child_chunk_size >= cfg.parent_chunk_size:
            raise ValueError(
                f"CHILD_CHUNK_SIZE ({cfg.child_chunk_size}) must be smaller than "
                f"PARENT_CHUNK_SIZE ({cfg.parent_chunk_size})."
            )
        if cfg.chunk_overlap >= cfg.child_chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({cfg.chunk_overlap}) must be smaller than "
                f"CHILD_CHUNK_SIZE ({cfg.child_chunk_size})."
            )

        # Automatically create all required directories
        target_directories = [
            cfg.chroma_db_dir,
            cfg.parent_store_dir,
            cfg.raw_data_dir,
            cfg.processed_data_dir,
            cfg.benchmarks_dir,
        ]

        for directory in target_directories:
            os.makedirs(str(directory), exist_ok=True)

        return cfg


@lru_cache(maxsize=1)
def get_settings(validate: bool = False) -> Settings:
    """Return cached singleton configuration settings."""
    settings = Settings()
    if validate:
        Settings.validate(settings)
    return settings
