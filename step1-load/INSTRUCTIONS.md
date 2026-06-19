# Step 1 — Load Endpoints

Loads the user-supplied initial and final structures, validates that they
match (same atom count and species order), detects periodicity, and writes
`endpoints.json` for the rest of the pipeline.

## Script

```bash
python step1-load/load_xyz.py --initial <path> --final <path> \
    [--run-id <label>] [--output-dir <path>]
```

Accepts any ASE-readable format (xyz, extxyz, POSCAR/CONTCAR, cif, ...).
`--run-id` defaults to `{initial-stem}-{final-stem}`; output defaults to
`outputs/{run_id}/`.

## Validation

- **Atom count**: `--initial` and `--final` must have the same number of
  atoms — error otherwise.
- **Species order**: atomic numbers must appear in the same order in both
  files — error otherwise (this is what lets NEB interpolate directly
  between the two position arrays).

## Periodicity detection

A system is treated as periodic if `pbc.any()` is true AND the cell is
non-zero. This sets `is_periodic` in `endpoints.json`, which downstream
steps use to decide whether to disable
`remove_rotation_and_translation` and to use the minimum-image convention
during NEB interpolation.

## Output: endpoints.json

```json
{
  "run_id": "reactant-product",
  "formula": "C4H8O",
  "rxn_key": "custom",
  "n_atoms": 13,
  "initial_xyz": "/abs/path/reactant.xyz",
  "final_xyz": "/abs/path/product.xyz",
  "is_periodic": false,
  "dft_forward_barrier_ev": null,
  "dft_reverse_barrier_ev": null,
  "ts_reference": null,
  "reactant": { "positions": [...], "atomic_numbers": [...], "pbc": [...], "cell": [...] },
  "product":  { "..." }
}
```

`dft_*_barrier_ev` and `ts_reference` are always `null` for user-supplied
structures (no DFT ground truth) — downstream steps treat this as "skip the
MLIP-vs-DFT comparison," not as an error.

## Next step

`step0-multi/runner.py --endpoints-dir outputs/{run_id}` runs the rest of
the pipeline (relax → NEB → monitor/retry → analyze) once per MLIP.
