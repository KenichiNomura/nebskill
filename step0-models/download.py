"""Download NequIP/Allegro model ZIPs from Zenodo and TACE checkpoint.

NequIP/Allegro models:
  Zenodo 10.5281/zenodo.18775904 → NequIP-OAM-{XL,L,M,S}, NequIP-MP-L
  Zenodo 10.5281/zenodo.16980200 → Allegro-OAM-L, Allegro-MP-L

After download, run compile.py to produce .nequip.pt2 files for your GPU.
"""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

NEQUIP_RECORD  = "18775904"
ALLEGRO_RECORD = "16980200"

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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    failed = []

    # ── NequIP models ─────────────────────────────────────────────────────────
    print("NequIP models (Zenodo 18775904):")
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

    # ── Allegro models ────────────────────────────────────────────────────────
    print("\nAllegro models (Zenodo 16980200):")
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

    # ── TACE checkpoint ───────────────────────────────────────────────────────
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
    print("Run step0-models/compile.py next to compile NequIP/Allegro models.")


if __name__ == "__main__":
    main()
