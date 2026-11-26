"""Hardware, Memory, and Storage Guardrail Monitor for Indian Legal RAG.

Enforces strict resource limits:
- Max RAM Cap: MAX_RAM_GB (default 18) measured over the whole process tree (parent + workers)
- Max Storage Cap: optional (MAX_STORAGE_GB <= 0 disables it); the walk is throttled
- GPU VRAM Protection: Configures PyTorch memory fractions and safe allocation buffers
- Adaptive Throttling: Enforces cooldown intervals and garbage collection when nearing thresholds
"""

import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("IndianLegalRAG.ResourceGuard")


def configure_torch_backends() -> None:
    """Route GPU attention away from cuDNN unless CUDNN_ENABLED=true.

    On the shared B200 MIG slice, cuDNN fails to initialise (CUDNN_STATUS_NOT_INITIALIZED)
    inside scaled_dot_product_attention because the system torch build and the cuDNN found
    on PYTHONPATH do not match. Text transformers do not need cuDNN: with it disabled, SDPA
    uses the flash / memory-efficient / math kernels instead, with identical outputs.
    """
    if not HAS_TORCH:
        return
    if os.getenv("CUDNN_ENABLED", "false").strip().lower() in ("1", "true", "yes"):
        return
    torch.backends.cudnn.enabled = False
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except AttributeError:
        pass


class ResourceGuardrailException(Exception):
    """Raised when memory or disk usage crosses absolute safety thresholds."""
    pass


class ResourceGuard:
    """Monitors and throttles system resources to guarantee operation within strict boundaries."""

    def __init__(
        self,
        max_ram_gb: float = 18.0,
        warn_ram_gb: float = 15.0,
        max_storage_gb: float = 0.0,
        storage_check_paths: Optional[list] = None,
        vram_safety_fraction: float = 0.85,
        num_cpu_threads: int = 14,
        storage_check_every: int = 50,
    ):
        self.max_ram_gb = max_ram_gb
        self.warn_ram_gb = warn_ram_gb
        self.max_storage_gb = max_storage_gb
        self.storage_check_paths = storage_check_paths or []
        self.vram_safety_fraction = vram_safety_fraction
        self.num_cpu_threads = num_cpu_threads
        self.storage_check_every = max(1, storage_check_every)
        self._storage_calls = 0
        self._storage_cached_gb = 0.0
        self.peak_ram_gb = 0.0
        self._initialized = False

    def initialize_hardware(self) -> Dict[str, Any]:
        """Configure hardware constraints (PyTorch threads, CUDA memory fractions) on startup."""
        telemetry: Dict[str, Any] = {
            "cuda_available": False,
            "device_name": "CPU",
            "threads_set": self.num_cpu_threads,
            "vram_allocated_cap_gb": 0.0,
        }

        if HAS_TORCH:
            configure_torch_backends()
            # Pin CPU threads to avoid CPU starvation
            torch.set_num_threads(self.num_cpu_threads)
            telemetry["threads_set"] = self.num_cpu_threads

            if torch.cuda.is_available():
                telemetry["cuda_available"] = True
                dev_name = torch.cuda.get_device_name(0)
                telemetry["device_name"] = dev_name

                # Check total VRAM
                total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
                total_vram_gb = total_vram_bytes / (1024 ** 3)
                telemetry["total_vram_gb"] = round(total_vram_gb, 2)

                # Set hard per-process fraction to guarantee safety buffer
                try:
                    torch.cuda.set_per_process_memory_fraction(self.vram_safety_fraction, 0)
                    cap_gb = total_vram_gb * self.vram_safety_fraction
                    telemetry["vram_allocated_cap_gb"] = round(cap_gb, 2)
                    logger.info(
                        f"[HardwareGuard] Bound PyTorch CUDA to {self.vram_safety_fraction*100:.0f}% of VRAM "
                        f"(~{cap_gb:.1f} GB / {total_vram_gb:.1f} GB) on {dev_name}"
                    )
                except Exception as err:
                    logger.warning(f"[HardwareGuard] Could not set per_process_memory_fraction: {err}")

        self._initialized = True
        return telemetry

    @staticmethod
    def process_tree_rss_bytes(process: Optional["psutil.Process"] = None) -> int:
        """RSS of a process plus all its descendants (e.g. PDF extraction workers)."""
        process = process or psutil.Process()
        total = 0
        try:
            total += process.memory_info().rss
            children = process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return total
        for child in children:
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue  # a worker may exit between listing and reading
        return total

    def get_ram_usage_gb(self) -> Tuple[float, float]:
        """Return (process_tree_rss_gb, system_used_ram_gb). Checks cgroup v2 if running inside container."""
        proc_rss_gb = self.process_tree_rss_bytes() / (1024 ** 3)
        self.peak_ram_gb = max(self.peak_ram_gb, proc_rss_gb)

        # Check cgroup v2 container memory usage if available
        cgroup_mem_current = Path("/sys/fs/cgroup/memory.current")
        if cgroup_mem_current.exists():
            try:
                with open(cgroup_mem_current, "r") as f:
                    cgroup_bytes = int(f.read().strip())
                    sys_used_gb = cgroup_bytes / (1024 ** 3)
                    return round(proc_rss_gb, 2), round(sys_used_gb, 2)
            except Exception:
                pass

        sys_used_gb = psutil.virtual_memory().used / (1024 ** 3)
        return round(proc_rss_gb, 2), round(sys_used_gb, 2)

    def get_monitored_storage_gb(self) -> float:
        """Disk used by the monitored directories. Skipped (0.0) when MAX_STORAGE_GB <= 0, and
        re-walked only every `storage_check_every` calls (a full walk is slow on NFS)."""
        if self.max_storage_gb <= 0 or not self.storage_check_paths:
            return 0.0
        self._storage_calls += 1
        if self._storage_calls > 1 and (self._storage_calls - 1) % self.storage_check_every:
            return self._storage_cached_gb
        total_bytes = 0
        for path_obj in self.storage_check_paths:
            p = Path(path_obj).resolve()
            if p.exists():
                if p.is_file():
                    total_bytes += p.stat().st_size
                elif p.is_dir():
                    for root, _, files in os.walk(p):
                        for file in files:
                            fp = os.path.join(root, file)
                            try:
                                total_bytes += os.path.getsize(fp)
                            except OSError:
                                pass
        self._storage_cached_gb = round(total_bytes / (1024 ** 3), 3)
        return self._storage_cached_gb

    def get_vram_usage_gb(self) -> Tuple[float, float]:
        """Return (vram_allocated_gb, vram_reserved_gb) if CUDA is available."""
        if HAS_TORCH and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            return round(alloc, 2), round(reserved, 2)
        return 0.0, 0.0

    def enforce_guardrails(self, step_name: str = "Ingestion") -> Dict[str, Any]:
        """Check all resources. Perform progressive cooling, GC, or raise exception if limits breached."""
        proc_ram, sys_ram = self.get_ram_usage_gb()
        vram_alloc, vram_res = self.get_vram_usage_gb()
        storage_gb = self.get_monitored_storage_gb()

        status: Dict[str, Any] = {
            "proc_ram_gb": proc_ram,
            "sys_ram_gb": sys_ram,
            "vram_alloc_gb": vram_alloc,
            "storage_gb": storage_gb,
            "action_taken": "NONE",
        }

        # 1. Check RAM Cap
        if proc_ram >= self.max_ram_gb:
            logger.critical(
                f"[GUARDRAIL TRIGGERED] Process RAM ({proc_ram:.2f} GB) exceeded hard limit ({self.max_ram_gb:.1f} GB) "
                f"at step '{step_name}'! Forcing emergency GC & cache purge..."
            )
            self.purge_memory()
            proc_ram, _ = self.get_ram_usage_gb()
            if proc_ram >= self.max_ram_gb:
                raise ResourceGuardrailException(
                    f"Process RAM ({proc_ram:.2f} GB) exceeds hard cap of {self.max_ram_gb:.1f} GB after purge. "
                    "Halting step to prevent cgroup OOM kill."
                )
            status["action_taken"] = "EMERGENCY_PURGE"

        elif proc_ram >= self.warn_ram_gb:
            logger.warning(
                f"[GUARDRAIL WARNING] Process RAM ({proc_ram:.2f} GB) reached warning threshold ({self.warn_ram_gb:.1f} GB). "
                f"Slowing down execution and releasing memory caches..."
            )
            self.purge_memory()
            time.sleep(1.5)  # Cooldown delay
            status["action_taken"] = "COOLDOWN_GC"

        # 2. Check Storage Cap
        if self.max_storage_gb > 0 and storage_gb >= self.max_storage_gb:
            logger.critical(
                f"[GUARDRAIL TRIGGERED] Monitored storage ({storage_gb:.2f} GB) exceeded limit ({self.max_storage_gb:.1f} GB)!"
            )
            raise ResourceGuardrailException(
                f"Storage usage ({storage_gb:.2f} GB) exceeded limit of {self.max_storage_gb:.1f} GB. "
                "Halting to avoid disk quota exhaustion."
            )

        return status

    def purge_memory(self) -> None:
        """Force garbage collection and clear PyTorch CUDA caches."""
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def format_telemetry(self, step_name: str = "") -> str:
        """Generate human-readable single-line telemetry string."""
        proc_ram, sys_ram = self.get_ram_usage_gb()
        vram_alloc, _ = self.get_vram_usage_gb()
        storage_gb = self.get_monitored_storage_gb()
        prefix = f"[{step_name}] " if step_name else ""
        storage = f"{storage_gb:.2f}GB/{self.max_storage_gb:.0f}GB" if self.max_storage_gb > 0 else "off"
        return (
            f"{prefix}TreeRAM: {proc_ram:.1f}GB/{self.max_ram_gb:.0f}GB (peak {self.peak_ram_gb:.1f}) | "
            f"SysRAM: {sys_ram:.1f}GB | VRAM: {vram_alloc:.1f}GB | Storage: {storage}"
        )

