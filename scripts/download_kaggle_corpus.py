#!/usr/bin/env python3
"""Kaggle Corpus Downloader for Indian Supreme Court Judgments.

Kaggle does not allow downloading a single folder of a dataset, so this script:
1. Downloads the full dataset once via ``kagglehub`` (~7 GB, cached in ~/.cache/kagglehub;
   re-running reuses the cache instead of downloading again).
2. Copies ONLY the PDFs inside the requested year folders (default: 2025, ~400 judgments)
   into data/raw_judgments/sc_<year>/.
3. Optionally deletes the kagglehub cache afterwards (--cleanup) to free the disk space.

Credentials: a Kaggle API token in ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY
environment variables (Kaggle -> Settings -> API -> Create New Token).

Usage:
    python scripts/download_kaggle_corpus.py --years 2025
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402

DEFAULT_DATASET = "adarshsingh0903/legal-dataset-sc-judgments-india-19502024"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected year folders of the Kaggle SC judgments dataset")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Kaggle dataset handle 'owner/name'")
    parser.add_argument("--years", type=str, nargs="+", default=["2025"], help="Year folders to keep")
    parser.add_argument("--dest-dir", type=str, default=None, help="Destination (defaults to data/raw_judgments)")
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help="Skip downloading and copy from an already downloaded/extracted dataset directory",
    )
    parser.add_argument("--cleanup", action="store_true", help="Delete the kagglehub download cache afterwards")
    return parser.parse_args()


def collect_year_pdfs(root: Path, years: List[str], dest_dir: Path) -> Dict[str, int]:
    """Copy every PDF whose parent folders include one of ``years`` into dest_dir/sc_<year>/."""
    copied: Dict[str, int] = {y: 0 for y in years}
    for pdf in root.rglob("*"):
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            continue
        folders = pdf.relative_to(root).parts[:-1]
        year = next((y for y in years if y in folders), None)
        if year is None:
            continue
        target_dir = dest_dir / f"sc_{year}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / pdf.name
        if not target.exists() or target.stat().st_size != pdf.stat().st_size:
            shutil.copy2(pdf, target)
        copied[year] += 1
    return copied


def main() -> None:
    args = parse_arguments()
    settings = get_settings()
    dest_dir = Path(args.dest_dir) if args.dest_dir else settings.raw_data_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  KAGGLE LEGAL CORPUS DOWNLOADER")
    print("=" * 70)
    print(f"Dataset     : {args.dataset}")
    print(f"Years       : {', '.join(args.years)}")
    print(f"Destination : {dest_dir.resolve()}")
    print("=" * 70)

    if args.source_dir:
        root = Path(args.source_dir).expanduser().resolve()
    else:
        try:
            import kagglehub
        except ImportError:
            print("[INFO] 'kagglehub' not found. Installing into the active environment...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "--quiet"])
            import kagglehub

        print("\n[Downloading] Kaggle only allows whole-dataset downloads (~7 GB). This runs once;")
        print("             later runs reuse ~/.cache/kagglehub. Only the selected years are kept.\n")
        try:
            root = Path(kagglehub.dataset_download(args.dataset))
        except Exception as err:
            print(f"[ERROR] Download failed: {err}")
            print("If this is an authentication error, create a token at Kaggle -> Settings -> API ->")
            print("'Create New Token' and save it as ~/.kaggle/kaggle.json (chmod 600).")
            sys.exit(1)

    print(f"[OK] Dataset available at: {root}")
    copied = collect_year_pdfs(root, args.years, dest_dir)

    total = sum(copied.values())
    if total == 0:
        print("[ERROR] No PDFs found inside the requested year folders. Top-level layout:")
        for child in sorted(root.iterdir())[:20]:
            print(f"   {child.name}{'/' if child.is_dir() else ''}")
        sys.exit(1)

    print("\n" + "=" * 70)
    for year, n in copied.items():
        print(f"  {year}: {n} PDFs -> {dest_dir / f'sc_{year}'}")
    size_mb = sum(p.stat().st_size for p in dest_dir.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"  Corpus size on disk: {size_mb:.1f} MB")
    print("=" * 70)

    if args.cleanup and not args.source_dir:
        shutil.rmtree(root, ignore_errors=True)
        print(f"[CLEANUP] Removed kagglehub cache at {root}")

    print("\nNext: python index_corpus.py --data-dir data/raw_judgments --reset")


if __name__ == "__main__":
    main()
