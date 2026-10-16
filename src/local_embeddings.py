"""Local and High-Capacity Neural Embedding Engine for Indian Legal RAG.

Optimized for:
1. zeroentropy/zembed-1 (4B parameter distilled model running in BF16 on Blackwell/CUDA)
2. BAAI/bge-base-en-v1.5 or bge-small-en-v1.5 (Local open-source baseline)
3. Fallback to Google Gemini Embedding API when configured

Engineered with:
- Batch-wise inference with tqdm progress bar
- Strict PyTorch VRAM safety bounds and garbage collection
- Truncation to avoid token overflow
"""

import gc
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("IndianLegalRAG.LocalEmbeddings")


def _mean_pooling(model_output: Any, attention_mask: Any) -> Any:
    """Mean pooling over token embeddings taking attention mask into account."""
    token_embeddings = model_output[0]  # First element has all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class LocalEmbeddingEngine:
    """High-performance embedding engine with automatic device selection and precision management."""

    def __init__(
        self,
        model_name: str = "zeroentropy/zembed-1",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        use_bf16: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_bf16 = use_bf16

        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._st_model = None  # SentenceTransformer instance if available
        self._initialized = False

    def initialize(self) -> None:
        """Lazy loader for embedding model onto selected device with safe precision."""
        if self._initialized:
            return

        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for local neural embeddings.")

        from src.resource_guard import configure_torch_backends

        configure_torch_backends()

        # Check VRAM capacity before loading 4B models on GPU
        if self.device == "cuda":
            try:
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if "zembed" in self.model_name.lower() and total_vram_gb < 7.0:
                    logger.warning(
                        f"[HardwareGuard] GPU VRAM ({total_vram_gb:.1f} GB) is below 7.0 GB required for 4B '{self.model_name}'. "
                        "Switching to 'BAAI/bge-small-en-v1.5' to operate within hardware constraints."
                    )
                    self.model_name = "BAAI/bge-small-en-v1.5"
            except Exception:
                pass

        logger.info(f"Loading neural embedding model '{self.model_name}' on device '{self.device}'...")

        # Determine precision
        dtype = torch.float32
        if self.device == "cuda":
            if self.use_bf16 and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                logger.info("Using native BF16 precision for embedding engine.")
            else:
                dtype = torch.float16
                logger.info("Using FP16 half-precision for embedding engine.")

        # 1. First attempt: Load via SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            model_kwargs = {"torch_dtype": dtype} if self.device == "cuda" else {}
            self._st_model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
            )
            self._initialized = True
            logger.info(f"Successfully initialized '{self.model_name}' via SentenceTransformers.")
            return
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM during model load. Re-attempting on CPU...")
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model_name = "BAAI/bge-small-en-v1.5"
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name, device="cpu")
            self._initialized = True
            return
        except Exception as st_err:
            logger.debug(f"SentenceTransformer loader skipped ({st_err}). Falling back to HuggingFace Transformers...")

        # 2. Second attempt: Load via Transformers AutoModel
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info(f"Successfully initialized '{self.model_name}' via Transformers AutoModel.")
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM during AutoModel load. Falling back to CPU...")
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model_name = "BAAI/bge-small-en-v1.5"
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name, device="cpu")
            self._initialized = True
        except Exception as hf_err:
            logger.error(f"Failed to load local embedding model '{self.model_name}': {hf_err}")
            raise RuntimeError(
                f"Could not load embedding model '{self.model_name}'. Check internet connectivity or cache: {hf_err}"
            )

    def embed_texts(
        self, texts: List[str], show_progress: bool = False, is_query: bool = False
    ) -> List[List[float]]:
        """Compute normalized vector embeddings for a list of text strings in memory-safe batches.

        Asymmetric retrieval models (zembed-1, e5, bge-m3 style) ship separate 'query' and
        'document' prompts. Queries MUST be encoded with the query prompt, otherwise the
        model's default ('document') prompt is applied and dense recall degrades.
        """
        if not texts:
            return []

        self.initialize()

        # If loaded via SentenceTransformer
        if self._st_model is not None:
            prompts = getattr(self._st_model, "prompts", None) or {}
            prompt_name = None
            if is_query and "query" in prompts:
                prompt_name = "query"
            elif not is_query and "document" in prompts:
                prompt_name = "document"
            embeddings = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
                prompt_name=prompt_name,
            )
            return embeddings.tolist()

        # Otherwise use Transformers pipeline with mean pooling
        all_embeddings: List[List[float]] = []
        import torch.nn.functional as F

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = _mean_pooling(outputs, encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                all_embeddings.extend(normalized.cpu().float().numpy().tolist())

            # Clear temporary GPU memory after each batch
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single legal query."""
        results = self.embed_texts([query], show_progress=False, is_query=True)
        return results[0] if results else []

