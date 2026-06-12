# Step 2 — Relax Endpoints

Geometrically relaxes both endpoint structures (reactant and product) using
the MACE-OFF calculator before NEB interpolation. This step is mandatory —
MD snapshots from Transition1x are not local minima and will cause poor NEB
convergence if used directly.

## Script

```bash
python step2-relax/relax_endpoints.py --reaction-id INT --config assets/neb_defaults.yaml
```

Reads `outputs/reaction_{id:04d}/endpoints.json`.
Writes `outputs/reaction_{id:04d}/relaxed_endpoints.json` with the same
structure as endpoints.json but with relaxed positions and MACE-OFF energies.

## Relaxation protocol

For each endpoint (reactant, then product):

1. **FIRE optimizer**, max 500 steps, target `fmax = 0.01 eV/Å`
2. If FIRE does not converge within 500 steps:
   - Switch to **BFGS optimizer**, max 500 steps, same `fmax`
3. If BFGS also fails to converge:
   - **Hard stop**: write failure report to `outputs/reaction_{id:04d}/relax_failure.json`
   - Mark queue.json status as `failed` with reason `endpoint_relaxation_failed`
   - Do not proceed to NEB (this failure does NOT consume a NEB retry)

## Calculator

Uses the MACE-OFF calculator factory from `step3-neb/calculator.py`:

```python
from step3_neb.calculator import make_calculator
calc = make_calculator(config)  # auto-detects GPU, loads model
```

See [references/mace_off_usage.md](../references/mace_off_usage.md).

## Output: relaxed_endpoints.json

```json
{
  "reactant": {
    "positions": [[...], ...],
    "atomic_numbers": [...],
    "energy_mace_ev": -123.45,
    "fmax_ev_per_ang": 0.008,
    "converged": true,
    "optimizer_used": "FIRE"
  },
  "product": { "..." },
  "dft_barrier_ev": 1.24
}
```

## Notes

- `remove_rotation_and_translation` is NOT applied during relaxation
  (only during NEB)
- Both endpoints use the same calculator instance to avoid redundant model loading
- Relaxation energies with MACE-OFF may differ slightly from Transition1x DFT
  energies — this is expected and acceptable
