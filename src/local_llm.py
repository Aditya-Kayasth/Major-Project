"""Local Instruction-Tuned LLM (Hugging Face Transformers) for synthesis, judging and query generation.

Runs an open-weights chat model (default: Qwen/Qwen2.5-7B-Instruct, ~15 GB in BF16) on the
local GPU so the system has no dependency on API quotas. Used when SYNTHESIS_PROVIDER=local,
or automatically as a fallback when Gemini is rate-limited or out of daily quota
(LOCAL_LLM_FALLBACK=true). All calls go through src/llm_budget.py, which caches responses.
"""

import json
import logging
import re
import threading
from typing import Any, Optional

logger = logging.getLogger("IndianLegalRAG.LocalLLM")

JSON_SUFFIX = "\n\nRespond with a single valid JSON object only. No markdown, no code fences, no extra text."


class LocalLLM:
    """Lazy-loaded chat model with deterministic-by-default generation."""

    def __init__(self, model_name: str, max_new_tokens: int = 768):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.resource_guard import configure_torch_backends

        configure_torch_backends()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        logger.info(f"Loading local LLM '{self.model_name}' on {self.device} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(self.device)
        self.model.eval()
        logger.info(f"Local LLM '{self.model_name}' ready.")

    @staticmethod
    def _extract_json(text: str) -> str:
        """Return the first balanced JSON object in text (models sometimes wrap it in prose/fences)."""
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            pass
        start = cleaned.find("{")
        depth = 0
        for i in range(start, len(cleaned)) if start >= 0 else []:
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
        return cleaned

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.1,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[str]:
        import torch

        with self._lock:
            self._load()
            messages = []
            if system or json_mode:
                messages.append({"role": "system", "content": (system or "You are a helpful assistant.") + (JSON_SUFFIX if json_mode else "")})
            messages.append({"role": "user", "content": str(prompt)})

            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens or (400 if json_mode else self.max_new_tokens),
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            if temperature and temperature > 0.05:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                gen_kwargs.update(do_sample=False)

            with torch.no_grad():
                output = self.model.generate(**inputs, **gen_kwargs)
            text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if not text:
            return None
        return self._extract_json(text) if json_mode else text


_instance: Optional[LocalLLM] = None


def get_local_llm(model_name: str) -> LocalLLM:
    global _instance
    if _instance is None or _instance.model_name != model_name:
        _instance = LocalLLM(model_name)
    return _instance
