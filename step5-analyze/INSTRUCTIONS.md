# Step 5 — Analyze and Report

Produces all output artifacts from a converged NEB calculation for one
MLIP, then (after all MLIPs finish) aggregates a cross-MLIP comparison.

## Per-MLIP scripts

```bash
python step5-analyze/analyze.py --output-dir outputs/{run_id}/{mlip} \
    --config assets/neb_defaults.yaml
python step5-analyze/plot.py    --output-dir outputs/{run_id}/{mlip}
python step5-analyze/writer.py  --output-dir outputs/{run_id}/{mlip}
```

All three are called in sequence by `step0-multi/runner.py` after each
MLIP's NEB converges.

## analyze.py — compute results

Reads `neb_result.json`, `relaxed_endpoints.json`, and `endpoints.json`.
Computes:

- **Forward barrier** (eV): `E_TS - E_reactant`
- **Reverse barrier** (eV): `E_TS - E_product`
- Both also reported in kcal/mol (×23.0609)
- **TS image index**: image with maximum energy in the converged NEB
- **MLIP vs DFT comparison** (only if `endpoints.json` carries a DFT
  reference — `null` for user-supplied structures, in which case this is
  skipped, not an error)

Writes `report.json` in `--output-dir`:
```json
{
  "run_id": "reactant-product",
  "formula": "C4H8O",
  "n_atoms": 13,
  "mlip": "nequip-oam-l",
  "forward_barrier_ev": 1.31,
  "forward_barrier_kcal": 30.2,
  "reverse_barrier_ev": 0.87,
  "ts_image_idx": 5,
  "dft_forward_barrier_ev": null,
  "n_images": 9,
  "neb_method": "improvedtangent",
  "neb_converged": true
}
```

## plot.py — energy profile

Produces `energy_profile.png` in `--output-dir`:
- X axis: image index (0 = reactant, N-1 = product)
- Y axis: energy relative to reactant (eV)
- Annotations: forward barrier, reverse barrier, TS image marker
- DFT reference barrier shown as a dashed line only if present

## writer.py — convergence log

`convergence.log` in `--output-dir`: tab-separated, one row per NEB phase
(`phase | steps | fmax_target | fmax_final | converged | wall_time_s`).
Also confirms `neb_trajectory.xyz` exists and reports its frame count.

## Cross-MLIP aggregation (step5-analyze/aggregate.py)

After all MLIPs finish, `step0-multi/runner.py` calls:
```bash
python step5-analyze/aggregate.py --endpoints-dir outputs/{run_id}
```

Reads `report.json` from each `outputs/{run_id}/{mlip}/` subdirectory and
writes:

- `outputs/{run_id}/summary.json` — forward/reverse barriers and
  convergence status per MLIP
- `outputs/{run_id}/comparison.png` — bar chart of forward/reverse barriers
  across MLIPs (non-converged MLIPs shown with reduced opacity)

## Final report to the user

After artifacts are written, summarize for the user:

1. Forward and reverse barriers (eV and kcal/mol) for each MLIP
2. Where MLIPs agree or disagree, and by how much
3. Location and character of the transition state
4. Any convergence issues encountered per MLIP and how retry resolved them
   (or didn't)
