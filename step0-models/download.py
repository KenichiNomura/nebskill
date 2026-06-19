"""Download NequIP/Allegro model ZIPs from Zenodo and TACE checkpoint.

NequIP/Allegro models:
  Zenodo 10.5281/zenodo.18775904 → NequIP-OAM-{XL,L,M,S}, NequIP-MP-L
  Zenodo 10.5281/zenodo.16980200 → Allegro-OAM-L, Allegro-MP-L

Allegro-FM (NequIP 0.6.1 / Allegro 0.3.0, deployed .pt files):
  Zenodo 10.5281/zenodo.14915165 → afm256_01_HL.pt, afm512_01_HL.pt
  Elements: C, O, H. No compilation needed — use directly with NequIPCalculator.

After download, run compile.py to produce .nequip.pt2 files for NequIP 2.x models.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

NEQUIP_RECORD  = "18775904"
ALLEGRO_RECORD = "16980200"
ALLEGRO_FM_RECORD = "14915165"

NEQUIP_MODELS = [
    "NequIP-OAM-XL",
    "NequIP-OAM-L",
    "NequIP-OAM-M",
    "NequIP-OAM-S",
    "NequIP-MP-L",
]

ALLEGRO_MODELS = [
    "Allegro-OAM-L",
    "Allegro-MP-L",
]

# Allegro-FM: plain .pt files, no compilation needed
ALLEGRO_FM_FILES = [
    "afm512_01_HL.pt",
    "afm256_01_HL.pt",
]

# TACE-OAM-L checkpoint — update URL when released publicly
TACE_URL  = None   # set when official download link is available
TACE_DEST = MODELS_DIR / "TACE-OAM-L.pt"


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"done ({size_kb:,} KB)")
    except Exception as exc:
        print(f"FAILED: {exc}")
        raise


def zenodo_url(record_id: str, filename: str) -> str:
    return (
        f"https://zenodo.org/api/records/{record_id}"
        f"/files/{filename}/content"
    )


def main():
    parser = argparse.ArgumentParser(description="Download MLIP model files")
    parser.add_argument("--model", default=None,
                        help="Download only this model key (e.g. allegro-fm-512). "
                             "Default: download all.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    only = args.model

    # ── Allegro-FM (Zenodo 14915165) — no compilation needed ─────────────────
    if only in (None, "allegro-fm-512", "allegro-fm-256", "allegro-fm"):
        print("Allegro-FM models (Zenodo 14915165):")
        targets = ALLEGRO_FM_FILES if only in (None, "allegro-fm") else \
                  (["afm512_01_HL.pt"] if only == "allegro-fm-512" else ["afm256_01_HL.pt"])
        for fname in targets:
            dest = MODELS_DIR / fname
            if dest.exists():
                print(f"  {fname} already present, skipping")
                continue
            try:
                download_file(zenodo_url(ALLEGRO_FM_RECORD, fname), dest)
            except Exception:
                failed.append(fname)

    # ── NequIP models ─────────────────────────────────────────────────────────
    if only is None:
        print("\nNequIP models (Zenodo 18775904):")
        for name in NEQUIP_MODELS:
            fname    = f"{name}-0.1.nequip.zip"
            dest     = MODELS_DIR / fname
            if dest.exists():
                print(f"  {fname} already present, skipping")
                continue
            try:
                download_file(zenodo_url(NEQUIP_RECORD, fname), dest)
            except Exception:
                failed.append(fname)

        # ── Allegro OAM/MP models ─────────────────────────────────────────────
        print("\nAllegro OAM/MP models (Zenodo 16980200):")
        for name in ALLEGRO_MODELS:
            fname = f"{name}-0.1.nequip.zip"
            dest  = MODELS_DIR / fname
            if dest.exists():
                print(f"  {fname} already present, skipping")
                continue
            try:
                download_file(zenodo_url(ALLEGRO_RECORD, fname), dest)
            except Exception:
                failed.append(fname)

        # ── TACE checkpoint ───────────────────────────────────────────────────
        print("\nTACE-OAM-L checkpoint:")
        if TACE_URL is None:
            print("  No public download URL yet — set TACE_URL in this script when available.")
        elif TACE_DEST.exists():
            print(f"  {TACE_DEST.name} already present, skipping")
        else:
            try:
                download_file(TACE_URL, TACE_DEST)
            except Exception:
                failed.append("TACE-OAM-L.pt")

    print(f"\nModels directory: {MODELS_DIR}")
    if failed:
        print(f"\n{len(failed)} download(s) failed: {failed}")
        sys.exit(1)
    if only is None:
        print("Run step0-models/compile.py next to compile NequIP 2.x models.")


if __name__ == "__main__":
    main()
