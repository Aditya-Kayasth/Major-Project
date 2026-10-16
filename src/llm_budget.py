"""Gemini Free-Tier Budget Guard: rate limiting, daily request budget and response caching.

Every Gemini text-generation call in the project (grounded synthesis, LLM-as-a-judge,
synthetic benchmark generation) goes through ``GeminiBudget.generate`` so that:

- Requests are spaced to stay under GEMINI_RPM requests per minute.
- At most GEMINI_DAILY_BUDGET requests are spent per calendar day (persisted on disk, so the
  budget survives restarts and is shared by every script run on the same machine).
- Identical prompts are answered from a disk cache (data/benchmarks/llm_cache.json), so
  re-running a benchmark does not spend quota again.
- HTTP 429/503 responses back off (honouring the server's "retry in Xs" hint); a per-day
  quota error stops further calls for the day instead of hammering the API.

Calls return None when the budget is exhausted or the API keeps failing; callers treat
that as "LLM unavailable" and never substitute fabricated scores.
"""

import datetime
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types

from src.config import Settings, get_settings

logger = logging.getLogger("IndianLegalRAG.LLMBudget")

_lock = threading.Lock()
_last_call_ts = 0.0


class GeminiBudget:
    """Process-wide guard around google-genai ``models.generate_content``."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.rpm = max(1.0, float(self.settings.gemini_rpm))
        self.daily_budget = int(self.settings.gemini_daily_budget)
        self.cache_path = self.settings.benchmarks_dir / "llm_cache.json"
        self.usage_path = self.settings.benchmarks_dir / "gemini_usage.json"
        self._cache: Optional[Dict[str, str]] = None
        self._daily_quota_hit = False

    # ------------------------------------------------------------------ cache
    def _load_cache(self) -> Dict[str, str]:
        if self._cache is None:
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)
        except Exception as err:
            logger.warning(f"Could not persist LLM cache: {err}")

    # ------------------------------------------------------------------ usage
    def _usage(self) -> Dict[str, int]:
        try:
            return json.loads(self.usage_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def requests_today(self) -> int:
        return int(self._usage().get(datetime.date.today().isoformat(), 0))

    def _record_request(self) -> None:
        usage = self._usage()
        today = datetime.date.today().isoformat()
        usage = {today: int(usage.get(today, 0)) + 1}  # keep only today's counter
        try:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            self.usage_path.write_text(json.dumps(usage), encoding="utf-8")
        except Exception:
            pass

    def remaining_today(self) -> int:
        return max(0, self.daily_budget - self.requests_today())

    # ------------------------------------------------------------------ calls
    def _wait_for_slot(self) -> None:
        global _last_call_ts
        min_interval = 60.0 / self.rpm
        with _lock:
            wait = _last_call_ts + min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call_ts = time.monotonic()

    def generate(
        self,
        client: Any,
        models: List[str],
        contents: Any,
        config: Optional[types.GenerateContentConfig] = None,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Generate text with rate limiting, budget accounting and caching. None on failure."""
        if client is None:
            return None

        system = getattr(config, "system_instruction", None) if config else None
        key_src = json.dumps([models[0] if models else "", str(system), str(contents)], ensure_ascii=False)
        cache_key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()

        if use_cache:
            cached = self._load_cache().get(cache_key)
            if cached is not None:
                return cached

        for model_name in [m for m in dict.fromkeys(models) if m]:
            for attempt in range(max_retries):
                if self._daily_quota_hit or self.remaining_today() <= 0:
                    logger.warning(
                        f"Gemini daily budget exhausted ({self.requests_today()}/{self.daily_budget}). "
                        "Skipping LLM call."
                    )
                    return None

                self._wait_for_slot()
                self._record_request()
                try:
                    response = client.models.generate_content(model=model_name, contents=contents, config=config)
                    text = (response.text or "") if response else ""
                    if text:
                        if use_cache:
                            self._load_cache()[cache_key] = text
                            self._save_cache()
                        return text
                    break
                except Exception as err:
                    msg = str(err)
                    if "404" in msg or "NOT_FOUND" in msg:
                        logger.info(f"Model '{model_name}' unavailable (404); trying next candidate.")
                        break
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                        if "PerDay" in msg or "per day" in msg.lower():
                            logger.error("Gemini per-day quota reached; no further LLM calls today.")
                            self._daily_quota_hit = True
                            return None
                        hint = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE) or re.search(
                            r"retryDelay['\"]?:\s*['\"]?(\d+)s", msg
                        )
                        backoff = float(hint.group(1)) + 1.0 if hint else 10.0 * (attempt + 1)
                        if backoff > 120:
                            logger.error(f"Gemini asked to wait {backoff:.0f}s; giving up on this call.")
                            return None
                        logger.warning(f"Gemini rate-limited/busy on {model_name}; retrying in {backoff:.0f}s...")
                        time.sleep(backoff)
                        continue
                    logger.warning(f"Gemini call failed on {model_name}: {msg[:200]}")
                    break
        return None


_budget_singleton: Optional[GeminiBudget] = None


def get_budget(settings: Optional[Settings] = None) -> GeminiBudget:
    """Return the shared process-wide budget guard."""
    global _budget_singleton
    if _budget_singleton is None:
        _budget_singleton = GeminiBudget(settings)
    return _budget_singleton
