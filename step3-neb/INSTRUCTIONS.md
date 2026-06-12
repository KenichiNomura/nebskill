# Step 3 — Run NEB

Interpolates NEB images between the relaxed endpoints and runs the two-phase
NEB calculation: standard NEB followed by Climbing Image NEB (CI-NEB).

## Script

```bash
python step3-neb/neb_runner.py --reaction-id INT --config assets/neb_defaults.yaml
```

Reads `outputs/reaction_{id:04d}/relaxed_endpoints.json`.
Writes `outputs/reaction_{id:04d}/neb_result.json` and
`outputs/reaction_{id:04d}/neb_trajectory.xyz` (updated after each phase).

## n_images calculation

Unless overridden by the user or agent:
```
path_length = sum of distances between consecutive interpolated images (Å)
n_images = max(9, round(path_length / 1.0))
```
If the edge-check flag is set in endpoints.json, double `n_images`.

## Phase 1 — Standard NEB

1. Create `n_images` copies of the reactant Atoms object
2. Attach MACE-OFF calculator to each image (from `step3-neb/calculator.py`)
3. Set endpoint positions (first and last images fixed)
4. Run IDPP interpolation: `neb.interpolate('idpp')`
5. Create NEB object:
   ```python
   neb = NEB(images,
             k=spring_constant,
             method='improvedtangent',
             remove_rotation_and_translation=True)
   ```
6. Run FIRE optimizer, max `phase1_max_steps` steps, until `fmax < phase1_fmax`
7. Save trajectory to `neb_trajectory.xyz`

If phase 1 exceeds max steps without converging → trigger step 4 (monitor/retry).

## Phase 2 — Climbing Image NEB (CI-NEB)

Only runs after phase 1 converges. Continues from phase 1 final positions.

1. Recreate NEB with `climb=True` (same images, same calculator)
2. Run FIRE optimizer, max `phase2_max_steps` steps, until `fmax < phase2_fmax`
3. Append to `neb_trajectory.xyz`

If phase 2 exceeds max steps → trigger step 4 (monitor/retry).

## neb_result.json

Written after each phase attempt:
```json
{
  "phase": 1,
  "converged": false,
  "steps_taken": 200,
  "fmax_final": 0.45,
  "n_images": 9,
  "method": "improvedtangent",
  "spring_constant": 0.1,
  "energies": [-123.1, -122.8, ...],
  "forces_per_image": [0.45, 0.32, ...]
}
```

## Calculator factory (calculator.py)

```python
from mace.calculators import mace_off
import torch

def make_calculator(config):
    device = config.calculator.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return mace_off(model=config.calculator.model_size, device=device,
                    default_dtype=config.calculator.dtype)
```

Each image requires its own calculator instance (do not share unless
`allow_shared_calculator=True` in NEB — only safe for non-parallel runs).

## Method fallback

If the agent selects `string` method on retry:
```python
neb = NEB(images, k=k, method='string', precon='Exp',
          remove_rotation_and_translation=True)
```
The `string` method with `precon='Exp'` performs better for gas-phase
organics with highly varying bond stiffness.

See [references/neb_method.md](../references/neb_method.md) for full details.
