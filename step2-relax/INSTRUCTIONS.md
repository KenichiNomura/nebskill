# Step 2 — Relax Endpoints

Geometrically relaxes both endpoint structures (reactant and product) with
the chosen MLIP before NEB interpolation. This step is mandatory — raw
user-supplied endpoints are rarely exact local minima and will cause poor
NEB convergence if used directly.

## Script

```bash
python step2-relax/relax_endpoints.py --output-dir outputs/{run_id}/{mlip} \
    --mlip <mlip_name> --config assets/neb_defaults.yaml \
    --registry assets/mlip_registry.yaml
```

`--fmax` overrides the config's `relaxation.fmax` (used by the adaptive
retry loop's `tighten_endpoint_relaxation` tool in step 4).

Reads `outputs/{run_id}/endpoints.json` (one level up from `--output-dir`
is not assumed — `step0-multi/runner.py` copies `endpoints.json` into each
per-MLIP output directory before calling this script).
Writes `relaxed_endpoints.json` with the same structure as endpoints.json
but with relaxed positions and MLIP energies.

## Relaxation protocol

For each endpoint (reactant, then product):

1. **FIRE optimizer**, max `optimizer_1_max_steps` steps, target
   `fmax = 0.05 eV/Å` (default; configurable via `relaxation.fmax`)
2. If FIRE does not converge:
   - Switch to **BFGS optimizer**, same step cap and `fmax`
3. If BFGS also fails to converge:
   - **Hard stop**: write failure report to `relax_failure.json`
     (exit code 3)
   - Do not proceed to NEB (this failure does NOT consume a NEB retry)

## Calculator

Uses the MLIP dispatcher in [lib/calculator.py](../lib/calculator.py):

```python
from lib.calculator import make_calculator
calc = make_calculator(mlip_name, registry)  # auto-detects GPU, loads model
```

See [references/mace_usage.md](../references/mace_usage.md) and
[references/nequip_allegro_usage.md](../references/nequip_allegro_usage.md).

## Output: relaxed_endpoints.json

```json
{
  "run_id": "reactant-product",
  "formula": "C4H8O",
  "mlip": "nequip-oam-l",
  "reactant": {
    "positions": [[...], ...],
    "atomic_numbers": [...],
    "energy_ev": -123.45,
    "fmax_ev_per_ang": 0.008,
    "converged": true,
    "optimizer_used": "FIRE"
  },
  "product": { "..." },
  "dft_forward_barrier_ev": null
}
```

`dft_forward_barrier_ev`/`dft_reverse_barrier_ev` are only non-null if the
loaded endpoints came with a DFT reference; for arbitrary user structures
they are `null` and downstream steps simply skip the MLIP-vs-DFT
comparison.

## Notes

- `remove_rotation_and_translation` is NOT applied during relaxation
  (only during NEB)
- Both endpoints use the same calculator instance to avoid redundant model
  loading
