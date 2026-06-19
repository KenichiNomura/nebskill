"""Compile downloaded NequIP/Allegro models for the current GPU hardware.

Must be run on the same GPU node type where NEB calculations will execute.
Produces .nequip.pt2 files (PyTorch 2.0 AOT-compiled) in step0-models/models/.

Usage (interactive GPU node or inside a SLURM job):
    python step0-models/compile.py

Or submit as a short SLURM job:
    sbatch step0-models/compile_job.sh
"""
import os
import subprocess
import sys
from pathlib import Path

# Ensure nvcc is discoverable for AOT-Inductor on systems without CUDA_HOME set.
# Must happen before any subprocess or torch import so subprocesses inherit it.
if "CUDA_HOME" not in os.environ:
    for _candidate in [
        "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda",  # CUDA 12.9
        "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda",  # CUDA 12.4
        "/usr/local/cuda",
    ]:
        if os.path.isfile(f"{_candidate}/bin/nvcc"):
            os.environ["CUDA_HOME"] = _candidate
            os.environ["PATH"] = f"{_candidate}/bin:{os.environ.get('PATH', '')}"
            print(f"[compile.py] CUDA_HOME set to {_candidate}")
            break

MODELS_DIR = Path(__file__).resolve().parent / "models"

# (model_name, zenodo_source_identifier)
MODELS = [
    ("NequIP-OAM-XL",  "nequip.net:mir-group/NequIP-OAM-XL:0.1"),
    ("NequIP-OAM-L",   "nequip.net:mir-group/NequIP-OAM-L:0.1"),
    ("NequIP-OAM-M",   "nequip.net:mir-group/NequIP-OAM-M:0.1"),
    ("NequIP-OAM-S",   "nequip.net:mir-group/NequIP-OAM-S:0.1"),
    ("NequIP-MP-L",    "nequip.net:mir-group/NequIP-MP-L:0.1"),
    ("Allegro-OAM-L",  "nequip.net:mir-group/Allegro-OAM-L:0.1"),
    ("Allegro-MP-L",   "nequip.net:mir-group/Allegro-MP-L:0.1"),
]


def compile_model(name: str, source: str) -> bool:
    output = MODELS_DIR / f"{name}.nequip.pt2"
    if output.exists():
        print(f"  {output.name} already compiled — skipping")
        return True
    print(f"  Compiling {name} ...", end=" ", flush=True)
    cmd = [
        "nequip-compile", source, str(output),
        "--mode",   "aotinductor",
        "--device", "cuda",
        "--target", "ase",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown"
        print(f"FAILED\n    {err}")
        return False
    print(f"done → {output.name}")
    return True


def main():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. Compilation must run on a GPU node.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    print(f"Compiling on: {gpu}\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, source in MODELS:
        if not compile_model(name, source):
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} model(s) failed to compile: {failed}")
        sys.exit(1)
    print(f"\nAll models compiled to {MODELS_DIR}")


if __name__ == "__main__":
    main()
