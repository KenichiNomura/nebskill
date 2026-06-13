# nebskill

Runs **Nudged Elastic Band (NEB)** calculations to find minimum energy paths and
reaction barriers for organic molecules. Supports three modes:

| Mode | What it does |
|---|---|
| `single` | One NEB job for one reaction using MACE-OFF23 |
| `batch` | Up to 20 parallel SLURM jobs, one reaction each |
| `benchmark` | One reaction × top-10 universal MLIPs from Matbench Discovery |

Reaction endpoints are sourced from the
[Transition1x dataset](https://doi.org/10.1038/s41597-022-01870-w)
(~20,000 organic reactions, ωB97x/6-31G\* DFT reference).
An LLM agent (Qwen3-32B via ALCF Sophia) selects NEB parameters, monitors
convergence, retries on failure, and interprets results.

---

## Architecture

See [`architecture.svg`](architecture.svg) for a full component diagram.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.12 | managed via `uv` |
| MACE-OFF23 | `pip install mace-torch` |
| ASE, h5py, openai, globus-sdk | core dependencies |
| ALCF allocation + Globus auth | required for LLM agent calls |
| GPU | recommended; CPU fallback via `device: auto` |
| SLURM | required for `batch` mode only |

**First-time setup — Globus authentication:**
```bash
python agent/inference_auth_token.py auth
```

**Transition1x dataset** (~6.2 GB) and **MACE-OFF model weights** are
auto-downloaded on first use.

---

## Quick Start

### Single mode

```bash
python agent/llm_agent.py --reaction-id 42 --defaults
```

Runs one NEB job for reaction 42. Outputs go to `outputs/reaction_0042/`.

### Batch mode

```bash
python step0-batch/queue.py init --n-reactions 20
python step0-batch/submit.py --n-jobs 10
python step0-batch/queue.py status        # monitor
python step0-batch/aggregate.py           # collect results
```

### Benchmark mode

```bash
python benchmark_agent.py --reaction-id 42
```

Runs the full NEB pipeline for reaction 42 using the top-10 available universal
MLIPs ranked by [Matbench Discovery CPS](https://matbench-discovery.materialsproject.org).
Produces a comparison report, overlaid energy profiles, and an LLM narrative summary.

---

## MLIP Benchmark Catalog

The benchmark selects MLIPs in **Matbench Discovery CPS rank order**, skipping any
that are not installed or lack a required checkpoint. The target set of 10 is drawn
from `assets/mlip_catalog.yaml`.

Current catalog (top candidates):

| Priority | Model | CPS | Package |
|---|---|---|---|
| 1 | PET-OAM-XL | 0.898 | `upet` + checkpoint |
| 2 | eSEN-30M-OAM | 0.888 | `fairchem-core` |
| 3 | NequIP-OAM-XL | 0.886 | `nequip` + checkpoint |
| 4 | SevenNet-Omni | 0.873 | `sevenn` |
| 5 | ORB v3 | 0.860 | `orb-models` |
| 6 | NequIP-OAM-L | 0.870 | `nequip` + checkpoint |
| 7 | Allegro-OAM-L | 0.840 | `allegro` + checkpoint |
| 8 | DPA-4.0-Pro | 0.830 | `deepmd-kit` + checkpoint |
| 9 | MACE-MPA-0 | 0.795 | `mace-torch` |
| 10 | MatterSim v1 | 0.767 | `mattersim` |

Fallbacks continue down the leaderboard (NequIP-OAM-M/S, MACE-MP-0, ORB v2,
CHGNet, M3GNet) until 10 working models are found.

**Install optional MLIP packages:**
```bash
pip install fairchem-core sevenn orb-models mattersim
pip install deepmd-kit       # DPA-4.0-Pro
pip install upet             # PET-OAM-XL
pip install nequip allegro   # NequIP / Allegro (also need checkpoints)
```

**Set checkpoint paths** for NequIP, Allegro, PET, and DPA in
`assets/neb_defaults.yaml` under `benchmark.checkpoints`.

**Refresh the catalog** when the leaderboard updates:
> Ask Claude to fetch `https://matbench-discovery.materialsproject.org` and
> `https://www.nequip.net`, then update `assets/mlip_catalog.yaml` and
> `references/universal_mlips.md` to match.

---

## Configuration

All parameters live in `assets/neb_defaults.yaml`.

Key sections:

```yaml
calculator:
  model_size: medium   # single/batch: small | medium | large (MACE-OFF23)
  device: auto         # auto | cpu | cuda

neb:
  n_images: auto       # max(9, round(path_length / 1.0))
  spring_constant: 0.1 # eV/Å
  phase2_fmax: 0.05    # eV/Å — CI-NEB convergence target

benchmark:
  max_mlips: 10
  endpoint_strategy: independent   # each MLIP relaxes its own endpoints
  skip_on_failure: true
  checkpoints:
    nequip_oam_xl: null  # /path/to/deployed_model.pth
    nequip_oam_l:  null
    allegro_oam_l: null
    pet:           null
    dpa:           null
```

---

## Output Artifacts

### Single / batch mode — `outputs/reaction_{id:04d}/`

| File | Contents |
|---|---|
| `endpoints.json` | DFT reference endpoints from Transition1x |
| `relaxed_endpoints.json` | MACE-OFF relaxed structures |
| `neb_result.json` | Convergence data, phase 1 + 2 |
| `neb_trajectory.xyz` | Full NEB path (all images and steps) |
| `report.json` | Forward/reverse barriers, MACE vs DFT error |
| `energy_profile.png` | Energy vs image with barrier annotation |
| `convergence.log` | Per-step fmax history |

Batch aggregation also writes `outputs/summary.json` and `outputs/summary.png`.

### Benchmark mode — `outputs/reaction_{id:04d}/benchmark/`

| File | Contents |
|---|---|
| `{mlip_key}/` | Full single-mode output set for each MLIP |
| `benchmark_manifest.json` | Status, wall time, Matbench rank per MLIP |
| `benchmark_comparison.json` | Structured comparison table |
| `benchmark_summary_table.txt` | ASCII table sorted by \|error vs DFT\| |
| `benchmark_energy_profiles.png` | Overlaid NEB profiles, all MLIPs |
| `benchmark_error_chart.png` | Error bar chart vs DFT reference |
| `interpretation.txt` | LLM narrative comparison |

---

## References

- [NEB method](references/neb_method.md)
- [MACE-OFF usage](references/mace_off_usage.md)
- [Transition1x schema](references/transition1x_schema.md)
- [Universal MLIPs catalog](references/universal_mlips.md)
- [ALCF API](references/alcf_api.md)
