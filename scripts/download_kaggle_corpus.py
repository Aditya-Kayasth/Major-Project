#!/usr/bin/env python3
"""Automated Kaggle Corpus Downloader for Indian Legal Judgments.

Pulls legal PDF datasets directly from Kaggle into data/raw_judgments/ on the remote server:
- Checks for credentials in ~/.kaggle/kaggle.json or environment variables
- Downloads and unzips archives safely
- Reports total downloaded PDFs, disk consumption, and directory structure
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from src.config import get_settings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download legal PDF dataset from Kaggle")
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcontractor/supreme-court-judgment-dataset",
        help="Kaggle dataset identifier in format 'username/dataset-name'",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Destination directory (defaults to data/raw_judgments)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    settings = get_settings()

    dest_dir = Path(args.dest_dir) if args.dest_dir else settings.raw_data_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  KAGGLE LEGAL CORPUS DOWNLOADER")
    print("=" * 70)
    print(f"Target Dataset : {args.dataset}")
    print(f"Target Directory: {dest_dir.resolve()}")
    print("=" * 70)

    # 1. Verify Kaggle Credentials
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

    if not kaggle_json_path.exists() and not has_env:
        print("\n[ERROR] Kaggle credentials not found!")
        print("Please configure one of the following:")
        print("  1. Place your 'kaggle.json' API token file at: ~/.kaggle/kaggle.json")
        print("     chmod 600 ~/.kaggle/kaggle.json")
        print("  2. Or set environment variables:")
        print("     export KAGGLE_USERNAME='your_username'")
        print("     export KAGGLE_KEY='your_api_key'")
        sys.exit(1)

    # 2. Import or Install kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[INFO] 'kaggle' package not found. Installing via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "--quiet"])
        from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("[OK] Authenticated successfully with Kaggle API.")

    # 3. Download Dataset
    zip_dest = dest_dir / "dataset.zip"
    print(f"\n[Downloading] Fetching '{args.dataset}' from Kaggle...")

    try:
        api.dataset_download_files(args.dataset, path=str(dest_dir), unzip=True, quiet=False)
        print("[OK] Download and extraction completed successfully!")
    except Exception as err:
        print(f"[ERROR] Failed to download dataset: {err}")
        sys.exit(1)

    # Clean up residual zip if any
    for zf in dest_dir.glob("*.zip"):
        try:
            zf.unlink()
        except Exception:
            pass

    # 4. Inventory Extracted Files
    pdf_files = list(dest_dir.rglob("*.pdf"))
    total_bytes = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    print("\n" + "=" * 70)
    print("  DOWNLOAD INVENTORY & STATISTICS")
    print("=" * 70)
    print(f"Total PDFs Found : {len(pdf_files):,}")
    print(f"Total Disk Space : {total_mb:.2f} MB (~{total_mb/1024:.2f} GB)")
    print(f"Dataset Location : {dest_dir.resolve()}")
    print("=" * 70)
    print("\nYou can now index the corpus by running:")
    print(f"python index_corpus.py --data-dir \"{dest_dir.resolve()}\"")


if __name__ == "__main__":
    main()

