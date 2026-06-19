---
name: nebskill
description: >
  Runs Nudged Elastic Band (NEB) calculations to find minimum energy paths
  and reaction barriers between a user-supplied initial and final structure.
  Runs the same NEB calculation across multiple machine-learned interatomic
  potentials (MACE, NequIP, Allegro, and others in the MLIP registry) and
  produces a side-by-side comparison of barrier heights and convergence.
  Activates when the user asks about transition states, reaction barriers,
  minimum energy paths, activation energies, NEB calculations, or comparing
  MLIPs/force fields on a reaction.
license: MIT
compatibility: >
  Requires Python 3.12+, uv, ASE, mace-torch, nequip, openai, globus-sdk.
  Needs an active ALCF allocation with Globus authentication (run
  agent/auth.py / the auth helper on first use) for the adaptive NEB retry
  LLM calls. GPU recommended for NequIP/Allegro; MACE supports CPU via
  auto-detection. Internet access required for MACE auto-download and for
  fetching NequIP/Allegro checkpoints via step0-models/download.py.
metadata:
  author: knomura
  version: "0.2.0"
  potentials: mace, nequip, allegro (extensible via assets/mlip_registry.yaml)
  llm-endpoint: alcf-sophia
  llm-model: Qwen/Qwen3-32B
allowed-tools: Bash Read Write
---

## Overview

This skill finds the minimum energy path (MEP) and reaction barrier between
a reactant and product structure supplied by the user, using the Nudged
Elastic Band (NEB) method. The same NEB calculation is run once per MLIP in
a configurable list — by default one representative model from each of the
MACE, NequIP, and Allegro families — so barrier heights and convergence
behavior can be compared side by side. An LLM agent (Qwen3-32B via ALCF
Sophia) monitors convergence and adaptively retries on failure for each
MLIP run.

For SLURM-equipped systems, the per-MLIP runs can be submitted as
independent jobs instead of running sequentially.

## Prerequisites

1. **Globus token**: must be valid before any LLM calls (used for the
   adaptive retry step). If expired, direct the user to run:
   `python agent/auth.py login`
2. **Initial and final structures**: the user provides two structure files
   in any ASE-readable format (xyz, extxyz, POSCAR/CONTCAR, cif, ...) with
   matching atom count and species order.
3. **MLIP model files**:
   - MACE models auto-download and cache to `~/.cache/mace/` on first use.
   - NequIP/Allegro foundation-model checkpoints must be downloaded and
     compiled once per machine via `step0-models/download.py` then
     `step0-models/compile.py` (compilation must run on a GPU node of the
     same type used for NEB). See `assets/mlip_registry.yaml` for which
     entries need this.

## Clarifying questions (always ask before running)

Before launching, ask the user:

1. **Initial and final structure files** — paths to the two endpoint
   structures.
2. **Which MLIPs to run** — default is one model per family:
   `mace-off`, `nequip-oam-l`, `allegro-oam-l`. The full set of available
   MLIPs is in [assets/mlip_registry.yaml](assets/mlip_registry.yaml).
3. **Use all other default parameters, or customize?**
   > - **[1] Use all defaults** — proceeds immediately with the values below
   > - **[2] Customize** — review and override individual parameters

Default values:

| Parameter | Default |
|---|---|
| MLIPs | `mace-off`, `nequip-oam-l`, `allegro-oam-l` |
| Number of NEB images | auto (`max(9, round(path_length/1.0))`) |
| Spring constant k | `0.1 eV/Å` |
| Final convergence fmax | `0.05 eV/Å` |
| Max retry attempts (per MLIP) | `3` |
| Execution mode | `interactive` (or `slurm` if available) |

If the user chooses **[2] Customize**, ask about each parameter one at a
time, showing the default and accepted options.

## Workflow

Execute steps in order. Read each step's INSTRUCTIONS.md before executing.

1. **Load endpoints** → run
   `python step1-load/load_xyz.py --initial <path> --final <path>`
   (see [step1-load/INSTRUCTIONS.md](step1-load/INSTRUCTIONS.md)). Writes
   `outputs/{run_id}/endpoints.json`.
2. **Run the multi-MLIP campaign** → run
   `python step0-multi/runner.py --endpoints-dir outputs/{run_id} [--mlips ...] [--mode interactive|slurm]`.
   For each MLIP this internally runs relax → NEB → adaptive retry on
   failure → analyze → plot → write artifacts
   ([step2-relax](step2-relax/INSTRUCTIONS.md),
   [step3-neb](step3-neb/INSTRUCTIONS.md),
   [step4-monitor](step4-monitor/INSTRUCTIONS.md),
   [step5-analyze](step5-analyze/INSTRUCTIONS.md)), then aggregates results
   across all MLIPs.
3. **Report to the user** — read `outputs/{run_id}/summary.json` and
   `comparison.png` and summarize forward/reverse barriers, convergence,
   and any disagreement across MLIPs.

## Output artifacts

All outputs written to `outputs/{run_id}/`:
- `endpoints.json` — the loaded initial/final structures
- `{mlip}/relaxed_endpoints.json`, `{mlip}/neb_result.json` — per-MLIP
  intermediate results
- `{mlip}/neb_trajectory.xyz` — full NEB path for that MLIP
- `{mlip}/energy_profile.png` — energy vs image index with barrier
  annotation
- `{mlip}/report.json` — barrier height, TS geometry for that MLIP
- `{mlip}/convergence.log` — per-step force history for both NEB phases
- `summary.json` — barrier heights and convergence across all MLIPs
- `comparison.png` — overlaid forward/reverse barrier comparison across
  MLIPs

## References

- [NEB method](references/neb_method.md)
- [MACE usage](references/mace_off_usage.md)
- [NequIP/Allegro usage](references/nequip_allegro_usage.md)
- [ALCF API](references/alcf_api.md)

## Default parameters

See [assets/neb_defaults.yaml](assets/neb_defaults.yaml) and
[assets/mlip_registry.yaml](assets/mlip_registry.yaml).
