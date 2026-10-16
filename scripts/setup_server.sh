#!/usr/bin/env bash
# One-time environment setup on the GPU server.
#
# Creates ./venv with --system-site-packages so the server's preinstalled PyTorch (CUDA),
# Transformers, NumPy, Pandas, FastAPI etc. are reused as-is. Only packages that are
# missing get installed, into the venv only, and the server's torch / transformers /
# numpy versions are pinned via a constraints file so pip can never upgrade or shadow them.
#
# Usage:  bash scripts/setup_server.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}

if [ ! -f venv/bin/activate ]; then
  echo "[setup] Creating venv (reusing system site-packages)..."
  rm -rf venv
  if ! "$PY" -m venv --system-site-packages venv 2>/dev/null; then
    # Some containers lack python3-venv/ensurepip and we have no sudo: use virtualenv from --user
    echo "[setup] 'python3 -m venv' unavailable; falling back to virtualenv (installed with --user)"
    rm -rf venv
    "$PY" -m pip install --user --quiet virtualenv
    "$PY" -m virtualenv --system-site-packages venv
  fi
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "[setup] Pinning preinstalled core libraries..."
python - <<'EOF' > /tmp/legalrag_constraints.txt
import importlib.metadata as md
for dist in ["torch", "torchvision", "torchaudio", "transformers", "numpy", "pandas", "fastapi", "pydantic", "tokenizers", "safetensors", "huggingface-hub"]:
    try:
        print(f"{dist}=={md.version(dist)}")
    except md.PackageNotFoundError:
        pass
EOF
cat /tmp/legalrag_constraints.txt

# import name -> pip requirement
declare -A NEEDED=(
  [chromadb]="chromadb>=0.5.0"
  [rank_bm25]="rank-bm25==0.2.2"
  [google.genai]="google-genai>=1.0.0"
  [sentence_transformers]="sentence-transformers"
  [pypdf]="pypdf>=5.0.0"
  [dotenv]="python-dotenv>=1.0.0"
  [tqdm]="tqdm"
  [psutil]="psutil"
  [kagglehub]="kagglehub"
  [uvicorn]="uvicorn"
)

MISSING=()
for mod in "${!NEEDED[@]}"; do
  if python -c "import ${mod}" 2>/dev/null; then
    echo "[setup] ok       ${mod}"
  else
    echo "[setup] missing  ${mod}"
    MISSING+=("${NEEDED[$mod]}")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "[setup] Installing into venv only: ${MISSING[*]}"
  pip install -c /tmp/legalrag_constraints.txt "${MISSING[@]}"
fi

python - <<'EOF'
import torch, transformers, chromadb, sentence_transformers
print(f"[setup] torch {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"[setup] transformers {transformers.__version__} | sentence-transformers {sentence_transformers.__version__} | chromadb {chromadb.__version__}")
EOF

[ -f .env ] || { cp .env.example .env; echo "[setup] Created .env from .env.example -> edit GEMINI_API_KEY"; }
echo "[setup] Done. Activate with: source venv/bin/activate"
