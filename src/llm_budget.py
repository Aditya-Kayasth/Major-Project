"""LLM Router & Free-Tier Budget Guard: provider fallback, rate limiting, budget and caching.

Every LLM text-generation call in the project (grounded synthesis, LLM-as-a-judge,
synthetic benchmark generation) goes through ``GeminiBudget.generate``. It tries the
primary SYNTHESIS_PROVIDER (an OpenAI-compatible API such as Groq, Gemini, or a local model)
and falls back to the other configured providers. For Gemini specifically:

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
_last_call_ts_by_provider: Dict[str, float] = {}


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
    def _wait_for_slot(self, provider: str = "gemini", rpm: Optional[float] = None) -> None:
        min_interval = 60.0 / max(1.0, rpm or self.rpm)
        with _lock:
            wait = _last_call_ts_by_provider.get(provider, 0.0) + min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call_ts_by_provider[provider] = time.monotonic()

    # ------------------------------------------------------------------ routing
    @property
    def provider(self) -> str:
        """Primary provider: 'openai' (any OpenAI-compatible API), 'gemini' or 'local'."""
        p = (self.settings.synthesis_provider or "gemini").lower()
        if p in ("openai", "groq", "cerebras", "openrouter", "mistral", "compat"):
            return "openai"
        return p if p in ("gemini", "local") else "gemini"

    def _chain(self, gemini_client: Any) -> List[str]:
        """Primary provider first, then every other configured provider as fallback."""
        available = {
            "openai": bool(self.settings.llm_api_key),
            "gemini": gemini_client is not None,
            "local": self.provider == "local" or self.settings.local_llm_fallback,
        }
        order = [self.provider] + [p for p in ("openai", "gemini", "local") if p != self.provider]
        return [p for p in order if available[p]]

    def _engine_label(self, provider: str) -> str:
        if provider == "openai":
            return f"{self.settings.llm_model} @ {self.settings.llm_base_url.split('//')[-1].split('/')[0]}"
        if provider == "local":
            return f"local {self.settings.local_llm_model}"
        return self.settings.gemini_model

    def llm_label(self, gemini_client: Any = None) -> str:
        chain = self._chain(gemini_client)
        return " -> ".join(self._engine_label(p) for p in chain) if chain else "none configured"

    def generate(
        self,
        client: Any,
        models: List[str],
        contents: Any,
        config: Optional[types.GenerateContentConfig] = None,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Generate text via the provider chain with caching. None if every provider fails.

        ``client`` is the Gemini client (may be None). The chain is the primary
        SYNTHESIS_PROVIDER followed by any other configured provider, so a rate-limited or
        out-of-quota provider transparently hands over to the next one.
        """
        system = getattr(config, "system_instruction", None) if config else None
        json_mode = getattr(config, "response_mime_type", None) == "application/json" if config else False
        temperature = getattr(config, "temperature", None) if config else None

        chain = self._chain(client)
        engine = self._engine_label(chain[0]) if chain else ""
        key_src = json.dumps([engine, str(system), str(contents)], ensure_ascii=False)
        cache_key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        if use_cache:
            cached = self._load_cache().get(cache_key)
            if cached is not None:
                return cached

        text: Optional[str] = None
        for i, provider in enumerate(chain):
            if i > 0:
                logger.info(f"Falling back to {self._engine_label(provider)} for this call.")
            if provider == "openai":
                text = self._generate_compat(contents, system, json_mode, temperature, max_retries)
            elif provider == "gemini":
                text = self._generate_gemini(client, models, contents, config, max_retries)
            else:
                text = self._generate_local(contents, system, json_mode, temperature)
            if text:
                break

        if text and use_cache:
            self._load_cache()[cache_key] = text
            self._save_cache()
        return text

    def _generate_compat(
        self,
        contents: Any,
        system: Optional[str],
        json_mode: bool,
        temperature: Optional[float],
        max_retries: int,
    ) -> Optional[str]:
        """Call any OpenAI-compatible /chat/completions endpoint (Groq, Cerebras, OpenRouter, ...)."""
        import httpx

        from src.local_llm import LocalLLM

        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": str(contents)})
        payload: Dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0.1 if temperature is None else float(temperature),
            "max_tokens": self.settings.llm_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}", "Content-Type": "application/json"}

        for attempt in range(max_retries + 1):
            self._wait_for_slot("openai", self.settings.llm_rpm)
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=180.0)
            except Exception as err:
                logger.warning(f"LLM API request failed ({err}); retrying...")
                time.sleep(3 * (attempt + 1))
                continue

            if resp.status_code == 200:
                try:
                    text = resp.json()["choices"][0]["message"]["content"] or ""
                except Exception as err:
                    logger.warning(f"Unexpected LLM API response shape: {err} :: {resp.text[:200]}")
                    return None
                # Reasoning models may emit their scratchpad inline; keep only the final answer
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if not text:
                    return None
                return LocalLLM._extract_json(text) if json_mode else text

            body = resp.text[:300]
            if resp.status_code == 400 and json_mode and "response_format" in payload:
                payload.pop("response_format")  # provider/model without JSON mode: rely on the prompt
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 5.0 * (attempt + 1)
                if wait > 120:
                    logger.error(f"LLM API asked to wait {wait:.0f}s (quota likely exhausted): {body}")
                    return None
                logger.warning(f"LLM API {resp.status_code}; retrying in {wait:.0f}s ... {body[:120]}")
                time.sleep(wait)
                continue
            logger.error(f"LLM API error {resp.status_code}: {body}")
            return None
        return None

    def _generate_local(
        self, contents: Any, system: Optional[str], json_mode: bool, temperature: Optional[float]
    ) -> Optional[str]:
        from src.local_llm import get_local_llm

        try:
            return get_local_llm(self.settings.local_llm_model).generate(
                str(contents), system=system, json_mode=json_mode,
                temperature=0.1 if temperature is None else float(temperature),
            )
        except Exception as err:
            logger.error(f"Local LLM generation failed: {err}")
            return None

    def _generate_gemini(
        self,
        client: Any,
        models: List[str],
        contents: Any,
        config: Optional[types.GenerateContentConfig],
        max_retries: int,
    ) -> Optional[str]:
        """Gemini call with rate limiting, daily budget and 429 handling. None on failure."""
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
                        return text
                    break
                except Exception as err:
                    msg = str(err)
                    if "404" in msg or "NOT_FOUND" in msg:
                        logger.info(f"Model '{model_name}' unavailable (404); trying next candidate.")
                        break
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                        if "PerDay" in msg or "per day" in msg.lower():
                            logger.error(f"Gemini per-day quota reached; no further Gemini calls today. ({msg[:300]})")
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
