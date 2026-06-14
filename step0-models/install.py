"""pip-install all MLIP packages into the current Python environment.

Packages with hard-pinned torch requirements (TACE, EquFlash) are flagged
but skipped by default to avoid breaking the base environment.
"""
import subprocess
import sys

# (pip_spec, description, conflict_note or None)
PACKAGES = [
    ("mace-torch",           "MACE-MP / MACE-OFF",              None),
    ("chgnet",               "CHGNet",                           None),
    ("matgl",                "M3GNet via MatGL",                 None),
    ("sevenn",               "SevenNet (7net-0, 7net-mf-ompa)", None),
    ("upet==0.1.0",          "PET-OAM-XL",                      None),
    ("fairchem-core",        "eSEN-30M-OAM / EquiformerV3",     None),
    ("orb-models",           "Orb-v2",                          None),
    ("mattersim",            "MatterSim-v1.0.0-5M",             None),
    ("nequip",               "NequIP / Allegro (OAM)",          None),
    # --- packages with conflicting torch pins — skipped by default ---
    ("tace>=0.2.0",
     "TACE-OAM-L",
     "requires torch-geometric; pin may conflict with base torch"),
    ("flashTP_e3nn==0.1.0",
     "EquFlash",
     "requires torch==2.8.0+cu126 and CUDA 12.6 strictly"),
]


def pip_install(spec: str) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", spec],
        capture_output=True, text=True,
    )
    return result.returncode == 0, result.stderr.strip()


def main():
    print(f"Installing MLIP packages into: {sys.prefix}\n")
    ok_list: list[str]              = []
    warn_list: list[tuple]          = []
    fail_list: list[tuple[str,str]] = []

    for spec, desc, note in PACKAGES:
        if note:
            print(f"  {desc} ({spec})\n    ⚠  {note}\n    Skipping.")
            warn_list.append((spec, desc, note))
            continue

        print(f"  {desc} ({spec}) ...", end=" ", flush=True)
        ok, err = pip_install(spec)
        if ok:
            print("OK")
            ok_list.append(spec)
        else:
            first_err = err.splitlines()[-1] if err else "unknown error"
            print(f"FAILED\n    {first_err}")
            fail_list.append((spec, desc))

    print(f"\n{'─'*60}")
    print(f"Installed : {len(ok_list)}")
    print(f"Skipped   : {len(warn_list)}  (conflicting torch pins)")
    print(f"Failed    : {len(fail_list)}")

    if fail_list:
        print("\nFailed packages:")
        for spec, desc in fail_list:
            print(f"  {spec}  ({desc})")
        sys.exit(1)

    if warn_list:
        print("\nTo install skipped packages (use a separate venv):")
        for spec, desc, _ in warn_list:
            print(f"  pip install {spec}  # {desc}")


if __name__ == "__main__":
    main()
