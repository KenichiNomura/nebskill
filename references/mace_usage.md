# MACE Usage Reference

## What is MACE-MP

MACE-MP(A) is a foundation model trained on inorganic/materials data
(broad element coverage across the periodic table) — see
`assets/mlip_registry.yaml`'s `mace-mp`/`mace-omat` entries.

MACE-OMOL (trained on the OMol25 dataset of isolated/non-periodic molecules)
was removed from the registry after repeatedly failing NEB on periodic
slabs — see `outputs/scaling-4node-54753443/mace-omol` and the
`scaling-2node-54765882` / `scaling-4node-54765850` runs (image_collapse /
kinking, fmax in the thousands). It wasn't trained for periodic/bulk
systems, so don't re-add it for this kind of work.

License: Academic Software License (ASL) — free for academic use only.

## Model sizes

| Size | File size | Speed | Accuracy |
|---|---|---|---|
| small | ~5 MB | fastest | lower |
| medium | ~17.5 MB | balanced | recommended |
| large | ~50 MB | slowest | highest |

Model files are cached at `~/.cache/mace/` after first download.

## Instantiation

```python
from mace.calculators import mace_mp
import torch

def make_calculator(model_size='medium', device='auto', dtype='float64'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return mace_mp(model=model_size, device=device, default_dtype=dtype)
```

Use `float64` for geometry optimization (more accurate forces).
Use `float32` for speed if only screening barriers approximately.

## Attaching to ASE Atoms

```python
atoms.calc = make_calculator()
energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/Å, shape (n_atoms, 3)
```

## NEB image setup

Each NEB image needs its own calculator instance:
```python
images = [reactant.copy() for _ in range(n_images)]
for image in images[1:-1]:             # skip fixed endpoints
    image.calc = make_calculator()
```

Do not share one calculator instance across images unless `parallel=False`
and `allow_shared_calculator=True` in the NEB object.

## Known warnings

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`: safe to ignore, related to torch.load
- `cuequivariance not available`: means no GPU kernel acceleration for
  equivariant operations; MACE still runs correctly via standard torch ops

## References

- MACE GitHub: https://github.com/ACEsuit/mace
