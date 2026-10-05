#!/usr/bin/env python3
"""Environment and Pre-flight Verification Script for Indian Legal RAG.

Validates virtual environment, Python version, dependencies, hardware acceleration,
API connectivity, and directory write permissions.
"""

import importlib
import os
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_env_file(filepath: Path) -> dict:
    """Simple parser for .env without external dependencies."""
    env_vars = {}
    if not filepath.exists():
        return env_vars
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            env_vars[key.strip()] = val.strip().strip("'\"")
    return env_vars


def check_virtual_environment() -> Tuple[str, str, str]:
    """Verify execution inside an active virtual environment."""
    is_venv = (sys.prefix != getattr(sys, "base_prefix", sys.prefix)) or hasattr(sys, "real_prefix")
    status = "PASS" if is_venv else "FAIL"
    details = f"Prefix: {sys.prefix}" if is_venv else "Global interpreter detected (Not in .venv)"
    return "Virtual Environment", status, details


def check_python_version() -> Tuple[str, str, str]:
    """Verify Python version >= 3.10."""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    passed = (v.major == 3 and v.minor >= 10) or v.major > 3
    status = "PASS" if passed else "FAIL"
    details = f"v{ver_str} (>= 3.10 required)" if passed else f"v{ver_str} (Upgrade required)"
    return "Python Version", status, details


def check_dependencies() -> List[Tuple[str, str, str]]:
    """Test importing of all required packages."""
    required_packages = [
        ("google-genai", "google.genai"),
        ("chromadb", "chromadb"),
        ("rank-bm25", "rank_bm25"),
        ("sentence-transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("pypdf", "pypdf"),
        ("python-dotenv", "dotenv"),
        ("pydantic", "pydantic"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
    ]

    results = []
    for pkg_name, module_name in required_packages:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "installed")
            results.append((f"Import: {pkg_name}", "PASS", f"v{version}"))
        except ImportError as err:
            results.append((f"Import: {pkg_name}", "FAIL", f"Missing ({err})"))
        except Exception as err:
            results.append((f"Import: {pkg_name}", "FAIL", f"Error ({err})"))
    return results


def check_cuda_pytorch() -> Tuple[str, str, str]:
    """Inspect CUDA/GPU acceleration capability."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return (
                "PyTorch CUDA / GPU",
                "PASS",
                f"{gpu_name} ({device_count} GPU, {vram_gb:.1f} GB VRAM)",
            )
        else:
            return "PyTorch CUDA / GPU", "PASS", "CPU fallback (CUDA not available)"
    except ImportError:
        return "PyTorch CUDA / GPU", "FAIL", "torch not installed"
    except Exception as err:
        return "PyTorch CUDA / GPU", "FAIL", f"Detection error: {err}"


def check_gemini_api() -> Tuple[str, str, str]:
    """Validate GEMINI_API_KEY and test API connectivity with client.models.list()."""
    env_vars = parse_env_file(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or env_vars.get("GEMINI_API_KEY", "")

    if not api_key or api_key == "your_gemini_api_key_here":
        return (
            "Gemini API Connectivity",
            "FAIL",
            "GEMINI_API_KEY not configured or set to placeholder in .env",
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        # Attempt minimal models.list call
        models = list(client.models.list(config={"page_size": 1}))
        return "Gemini API Connectivity", "PASS", f"Connected ({len(models)}+ models accessible)"
    except ImportError:
        return "Gemini API Connectivity", "FAIL", "google-genai package not installed"
    except Exception as err:
        return "Gemini API Connectivity", "FAIL", f"API Call Failed: {str(err)[:50]}"


def check_directory_write_access(dir_name: str, rel_path: str) -> Tuple[str, str, str]:
    """Verify read and write access to required storage directories."""
    target_dir = PROJECT_ROOT / rel_path
    target_dir.mkdir(parents=True, exist_ok=True)
    probe_file = target_dir / f".write_test_{uuid.uuid4().hex[:8]}.tmp"

    try:
        with open(probe_file, "w", encoding="utf-8") as f:
            f.write("write_permission_probe")
        with open(probe_file, "r", encoding="utf-8") as f:
            content = f.read()
        probe_file.unlink(missing_ok=True)

        if content == "write_permission_probe":
            return f"Write Access: {dir_name}", "PASS", f"{rel_path} writable"
        return f"Write Access: {dir_name}", "FAIL", "Data corruption on read-back"
    except Exception as err:
        return f"Write Access: {dir_name}", "FAIL", f"Write failed: {err}"


def render_ascii_table(rows: List[Tuple[str, str, str]]) -> str:
    """Format check results as an aligned ASCII table."""
    col_w_step = max(len(r[0]) for r in rows + [("Verification Step", "", "")]) + 2
    col_w_status = 10
    col_w_details = max(len(r[2]) for r in rows + [("", "", "Details")]) + 2

    # Restrict max detail column width for terminal readability
    col_w_details = min(max(col_w_details, 25), 65)

    sep_border = (
        f"+-{'-' * col_w_step}-+-{'-' * col_w_status}-+-{'-' * col_w_details}-+"
    )
    header = (
        f"| {'Verification Step'.ljust(col_w_step)} "
        f"| {'Status'.center(col_w_status)} "
        f"| {'Details'.ljust(col_w_details)} |"
    )

    lines = [
        sep_border,
        header,
        sep_border,
    ]

    for step, status, details in rows:
        clipped_details = details if len(details) <= col_w_details else details[: col_w_details - 3] + "..."
        status_styled = f"[{status}]"
        lines.append(
            f"| {step.ljust(col_w_step)} "
            f"| {status_styled.center(col_w_status)} "
            f"| {clipped_details.ljust(col_w_details)} |"
        )

    lines.append(sep_border)
    return "\n".join(lines)


def main():
    print("\n" + "=" * 80)
    print(" INDIAN LEGAL RAG WORKSPACE: ENVIRONMENT PRE-FLIGHT VERIFICATION ".center(80))
    print("=" * 80 + "\n")

    checks: List[Tuple[str, str, str]] = []

    # 1. Virtual Environment & Python Version
    checks.append(check_virtual_environment())
    checks.append(check_python_version())

    # 2. Directory Permissions
    checks.append(check_directory_write_access("ChromaDB Store", "chromadb_store"))
    checks.append(check_directory_write_access("Parent Store", "data/parent_store"))

    # 3. Dependencies
    dep_checks = check_dependencies()
    checks.extend(dep_checks)

    # 4. Hardware Acceleration
    checks.append(check_cuda_pytorch())

    # 5. Gemini API Connectivity
    checks.append(check_gemini_api())

    # Render Table
    table_output = render_ascii_table(checks)
    print(table_output)

    # Summary
    total_checks = len(checks)
    passed_checks = sum(1 for c in checks if c[1] == "PASS")
    failed_checks = total_checks - passed_checks

    print(f"\nExecution Summary: {passed_checks}/{total_checks} PASSED ({failed_checks} FAILED/PENDING)")
    if failed_checks > 0:
        print("[!] Note: Missing dependencies or unset GEMINI_API_KEY will resolve after installation & setup.\n")
    else:
        print("[*] Environment verified and production-ready!\n")


if __name__ == "__main__":
    main()
