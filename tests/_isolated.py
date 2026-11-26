"""Test helper: Settings whose every directory lives in a temporary folder and that has no LLM keys.

Unit tests must never write into the real workspace (Chroma, SQLite, processed text, benchmarks)
and must never call an LLM API. Use:

    self.tmp, self.settings = isolated_settings()
    ...
    self.tmp.cleanup()
"""

import tempfile
from pathlib import Path
from typing import Tuple

from src.config import Settings, get_settings


def isolated_settings(**overrides) -> Tuple[tempfile.TemporaryDirectory, Settings]:
    tmp = tempfile.TemporaryDirectory(prefix="legalrag_test_", ignore_cleanup_errors=True)
    root = Path(tmp.name)
    update = {
        "chroma_db_dir": root / "chroma",
        "sqlite_db_path": root / "store" / "parent_store.db",
        "checkpoint_path": root / "store" / "indexing_checkpoint.json",
        "parent_store_dir": root / "parent_store",
        "raw_data_dir": root / "raw",
        "processed_data_dir": root / "processed",
        "benchmarks_dir": root / "benchmarks",
        "metadata_dir": root / "metadata",
        "logs_dir": root / "logs",
        # No LLM calls from unit tests
        "llm_api_key": "",
        "gemini_api_key": "",
        "local_llm_fallback": False,
        "embedding_provider": "local",
    }
    update.update(overrides)
    settings = get_settings().model_copy(update=update)
    Settings.validate(settings)
    return tmp, settings
