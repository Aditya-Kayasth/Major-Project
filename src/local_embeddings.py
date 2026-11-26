"""Local Neural Embedding Engine for Indian Legal RAG.

CPU plan (CLAUDE.md D4): BAAI/bge-small-en-v1.5, fp32 on CPU, auto device.
- Queries get the BGE instruction prefix; documents get none (asymmetric retrieval).
  Models whose sentence-transformers config defines a "query" prompt (e.g. zembed-1) use
  that prompt instead.
- torch threads are set explicitly (NUM_THREADS / INDEX_TORCH_THREADS).
- The CUDA branches (bf16/fp16, VRAM check) are kept for the stretch GPU run only.
"""

import logging
from typing import Any, List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("IndianLegalRAG.LocalEmbeddings")

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def resolve_query_instruction(model_name: str, setting: Optional[str] = "auto") -> str:
    """'auto' -> the BGE prefix for English BGE v1.5-style models, else ''. Any other value is used as-is."""
    if setting is None or setting == "auto":
        name = (model_name or "").lower()
        return BGE_QUERY_INSTRUCTION if ("bge-" in name and "-en" in name) else ""
    return setting


def _mean_pooling(model_output: Any, attention_mask: Any) -> Any:
    """Mean pooling over token embeddings taking attention mask into account."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class LocalEmbeddingEngine:
    """Embedding engine with automatic device selection (cuda if available, else cpu fp32)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
        use_bf16: bool = True,
        num_threads: Optional[int] = None,
        query_instruction: Optional[str] = "auto",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_bf16 = use_bf16
        self.num_threads = num_threads
        self.query_prefix = resolve_query_instruction(model_name, query_instruction)

        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._st_model = None
        self._initialized = False

    def _dtype(self) -> Any:
        """fp32 on CPU; bf16 (or fp16) only on CUDA."""
        if self.device == "cuda":
            return torch.bfloat16 if (self.use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
        return torch.float32

    def initialize(self) -> None:
        """Lazy loader for the embedding model."""
        if self._initialized:
            return
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for local neural embeddings.")

        from src.resource_guard import configure_torch_backends

        configure_torch_backends()
        if self.num_threads:
            torch.set_num_threads(int(self.num_threads))

        if self.device == "cuda":  # stretch GPU run only
            try:
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if "zembed" in self.model_name.lower() and total_vram_gb < 7.0:
                    logger.warning(f"GPU VRAM {total_vram_gb:.1f} GB < 7 GB for '{self.model_name}'; using bge-small.")
                    self.model_name = "BAAI/bge-small-en-v1.5"
                    self.query_prefix = resolve_query_instruction(self.model_name, "auto")
            except Exception:
                pass

        dtype = self._dtype()
        logger.info(f"Loading embedding model '{self.model_name}' on '{self.device}' ({dtype}), "
                    f"threads={torch.get_num_threads()}")

        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {"dtype": dtype} if self.device == "cuda" else {}
            self._st_model = SentenceTransformer(
                self.model_name, device=self.device, model_kwargs=model_kwargs, trust_remote_code=True
            )
            self._initialized = True
            return
        except Exception as st_err:
            logger.warning(f"SentenceTransformer load failed ({st_err}); trying transformers AutoModel.")

        from transformers import AutoModel, AutoTokenizer  # transformers 5: `dtype=` (was `torch_dtype=`)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, dtype=dtype, trust_remote_code=True).to(self.device)
        self.model.eval()
        self._initialized = True

    def _st_prompts(self) -> dict:
        return (getattr(self._st_model, "prompts", None) or {}) if self._st_model is not None else {}

    def prepare_texts(self, texts: List[str], is_query: bool) -> List[str]:
        """Apply the query prefix (queries only) unless the model ships its own 'query' prompt."""
        if is_query and self.query_prefix and "query" not in self._st_prompts():
            return [self.query_prefix + t for t in texts]
        return list(texts)

    def embed_texts(self, texts: List[str], show_progress: bool = False, is_query: bool = False) -> List[List[float]]:
        """Normalized embeddings for a list of texts, in batches."""
        if not texts:
            return []
        self.initialize()
        inputs = self.prepare_texts(texts, is_query)

        if self._st_model is not None:
            prompts = self._st_prompts()
            prompt_name = None
            if is_query and "query" in prompts:
                prompt_name = "query"
            elif not is_query and "document" in prompts:
                prompt_name = "document"
            embeddings = self._st_model.encode(
                inputs,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
                prompt_name=prompt_name,
            )
            return embeddings.tolist()

        import torch.nn.functional as F

        all_embeddings: List[List[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            encoded = self.tokenizer(
                inputs[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = _mean_pooling(outputs, encoded["attention_mask"])
                all_embeddings.extend(F.normalize(pooled, p=2, dim=1).cpu().float().numpy().tolist())
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        results = self.embed_texts([query], show_progress=False, is_query=True)
        return results[0] if results else []
