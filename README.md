# nebskill

Runs **Nudged Elastic Band (NEB)** calculations to find minimum energy paths
and reaction barriers between a user-supplied initial and final structure,
and compares the result across multiple machine-learned interatomic
potentials (MLIPs) — by default one model each from the **MACE**, **NequIP**,
and **Allegro** families.

An LLM agent (Qwen3-32B via ALCF Sophia) monitors NEB convergence and
adaptively retries on failure, independently for each MLIP run.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.12 | managed via `uv` |
| MACE | `pip install mace-torch` (auto-downloads model weights) |
| NequIP / Allegro | `pip install nequip`; foundation-model checkpoints must be downloaded (`step0-models/download.py`) and compiled on a GPU node (`step0-models/compile.py`) before first use |
| ASE, openai, globus-sdk | core dependencies |
| ALCF allocation + Globus auth | required for the adaptive-retry LLM calls |
| GPU | recommended for NequIP/Allegro; MACE supports CPU via `device: auto` |
| SLURM | optional — lets the per-MLIP runs submit as independent jobs instead of running sequentially |

**First-time setup — Globus authentication:**
```bash
python agent/auth.py login
```

**Install MLIP packages:**
```bash
python step0-models/install.py
```

**Download and compile NequIP/Allegro checkpoints** (once per machine, on a
GPU node of the type used for NEB runs):
```bash
python step0-models/download.py
python step0-models/compile.py
```

---

## Quick Start

```bash
python step1-load/load_xyz.py --initial reactant.xyz --final product.xyz
python step0-multi/runner.py --endpoints-dir outputs/reactant-product
```

This loads the two structures, then for each MLIP in
`assets/neb_defaults.yaml`'s `mlips:` list (default: `mace-off`,
`nequip-oam-l`, `allegro-oam-l`) relaxes the endpoints, runs two-phase NEB
(standard NEB → CI-NEB), retries adaptively on convergence failure, and
writes analysis artifacts — then aggregates barrier heights and convergence
across all MLIPs.

Override the MLIP list or run mode directly:
```bash
python step0-multi/runner.py --endpoints-dir outputs/reactant-product \
    --mlips mace-off nequip-oam-l allegro-oam-l \
    --mode interactive   # or: slurm
```

`--initial`/`--final` accept any ASE-readable format (xyz, extxyz,
POSCAR/CONTCAR, cif, ...). Periodicity (bulk/surface vs isolated/molecular)
is auto-detected from the input structures.

---

## Available MLIPs

The full registry of supported MLIPs — including MACE, NequIP, Allegro, and
several others (CHGNet, M3GNet, SevenNet, PET-MAD, eSEN/fairchem, TACE, Orb,
MatterSim) — is in
[assets/mlip_registry.yaml](assets/mlip_registry.yaml). Any registry key can
be passed via `--mlips`; entries needing a local checkpoint note the
expected file path.

---

## Configuration

All parameters live in `assets/neb_defaults.yaml`.

Key sections:

```yaml
relaxation:
  fmax: 0.01           # eV/Å, endpoint relaxation convergence

neb:
  n_images: auto       # max(9, round(path_length / 1.0))
  spring_constant: 0.1 # eV/Å
  phase2_fmax: 0.05    # eV/Å — CI-NEB convergence target

retry:
  max_attempts: 3      # adaptive retries per MLIP before giving up

mlips:
  - mace-off
  - nequip-oam-l
  - allegro-oam-l

execution:
  mode: interactive     # interactive | slurm
```

---

## Output Artifacts

Written to `outputs/{run_id}/` (run_id defaults to
`{initial-stem}-{final-stem}`):

| File | Contents |
|---|---|
| `endpoints.json` | Loaded initial/final structures |
| `{mlip}/relaxed_endpoints.json` | Relaxed endpoints for that MLIP |
| `{mlip}/neb_result.json` | Convergence data, phase 1 + 2 |
| `{mlip}/neb_trajectory.xyz` | Full NEB path (all images and steps) |
| `{mlip}/report.json` | Forward/reverse barriers, TS geometry |
| `{mlip}/energy_profile.png` | Energy vs image with barrier annotation |
| `{mlip}/convergence.log` | Per-step fmax history |
| `summary.json` | Barrier heights and convergence across all MLIPs |
| `comparison.png` | Overlaid forward/reverse barrier comparison |

---

## References

- [NEB method](references/neb_method.md)
- [MACE usage](references/mace_off_usage.md)
- [NequIP/Allegro usage](references/nequip_allegro_usage.md)
- [ALCF API](references/alcf_api.md)
