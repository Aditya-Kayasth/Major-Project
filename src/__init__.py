"""Indian Legal Precedents RAG (Supreme Court criminal judgments, CPU build).

Kept import-free on purpose: PDF extraction workers are spawned processes that import only
src.judgment_text (+ pypdf), so importing the package must not pull in pydantic, torch, etc.
"""

__version__ = "0.2.0"


def __getattr__(name):  # lazy access: `from src import get_settings` still works
    if name in ("Settings", "get_settings"):
        from src import config

        return getattr(config, name)
    raise AttributeError(name)
