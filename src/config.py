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

    # Resource Guardrails (Strict Memory & Storage Ceilings)
    max_ram_gb: float = Field(
        default_factory=lambda: float(os.getenv("MAX_RAM_GB", "20.0"))
    )
    warn_ram_gb: float = Field(
        default_factory=lambda: float(os.getenv("WARN_RAM_GB", "16.0"))
    )
    max_storage_gb: float = Field(
        default_factory=lambda: float(os.getenv("MAX_STORAGE_GB", "30.0"))
    )
    vram_safety_fraction: float = Field(
        default_factory=lambda: float(os.getenv("VRAM_SAFETY_FRACTION", "0.85"))
    )
    num_threads: int = Field(
        default_factory=lambda: int(os.getenv("NUM_THREADS", "8"))
    )

    # Embedding Provider Configuration (local vs. gemini)
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local").lower()
    )
    local_embedding_model: str = Field(
        default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "zeroentropy/zembed-1")
    )
    gemini_embedding_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    )

    # Synthesis Provider Configuration (gemini vs. ollama)
    synthesis_provider: str = Field(
        default_factory=lambda: os.getenv("SYNTHESIS_PROVIDER", "gemini").lower()
    )
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )
    # Gemini free-tier guard (see src/llm_budget.py). Check your limits in Google AI Studio.
    gemini_rpm: float = Field(
        default_factory=lambda: float(os.getenv("GEMINI_RPM", "5"))
    )
    gemini_daily_budget: int = Field(
        default_factory=lambda: int(os.getenv("GEMINI_DAILY_BUDGET", "200"))
    )
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )

    # Vector Store & Collection
    chroma_db_dir: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("CHROMA_DB_DIR"), "chromadb_store")
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "indian_legal_precedents")
    )

    # 3-Tier Hierarchical Legal Chunking Hyperparameters
    parent_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("PARENT_CHUNK_SIZE", "1500"))
    )
    child_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_SIZE", "400"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "80"))
    )

    # Reranker Model
    reranker_model: str = Field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )

    # Data Storage Directories & SQLite Path
    sqlite_db_path: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("SQLITE_DB_PATH"), "data/parent_store.db")
    )
    checkpoint_path: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("CHECKPOINT_PATH"), "data/processed/indexing_checkpoint.json")
    )
    parent_store_dir: Path = Field(
        default_factory=lambda: _resolve_path(os.getenv("PARENT_STORE_DIR"), "data/parent_store")
    )
    write_parent_json: bool = Field(
        default_factory=lambda: os.getenv("WRITE_PARENT_JSON", "false").lower() in ("1", "true", "yes")
    )
    # Strength of the Stare Decisis / role / judge authority prior. It is applied as a bounded
    # log-scale bonus (at most +/- AUTHORITY_MAX_BONUS) so it breaks ties between equally
    # relevant chunks without overriding semantic relevance. 0 disables it (pure relevance).
    authority_strength: float = Field(
        default_factory=lambda: float(os.getenv("AUTHORITY_STRENGTH", "0.1"))
    )
    authority_max_bonus: float = Field(
        default_factory=lambda: float(os.getenv("AUTHORITY_MAX_BONUS", "0.05"))
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

        # Only require GEMINI_API_KEY if Gemini is selected for either embeddings or synthesis
        if (cfg.embedding_provider == "gemini" or cfg.synthesis_provider == "gemini"):
            if not cfg.gemini_api_key or cfg.gemini_api_key in ("your_gemini_api_key_here", ""):
                # Issue warning or raise depending on usage
                import logging
                logging.getLogger("IndianLegalRAG.Config").warning(
                    "GEMINI_API_KEY is not set or using placeholder. Cloud Gemini operations will be disabled."
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
            cfg.sqlite_db_path.parent,
            cfg.checkpoint_path.parent,
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
