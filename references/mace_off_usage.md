# MACE Usage Reference

## What is MACE-OFF / MACE-MP

MACE-OFF23 is a pre-trained transferable machine learning force field for
organic molecules, based on the MACE equivariant architecture. It covers
elements H, C, N, O, F, S, Cl. MACE-MP(A) is the companion foundation model
trained on inorganic/materials data (broader element coverage, see
`assets/mlip_registry.yaml`'s `mace-mp`/`mace-omat` entries). MACE-OMOL
(`mace-omol` entry) is trained on the OMol25 dataset of isolated/non-periodic
molecules and complexes — treat results on periodic/bulk systems as
exploratory, since it wasn't trained for that regime.

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
from mace.calculators import mace_off
import torch

def make_calculator(model_size='medium', device='auto', dtype='float64'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return mace_off(model=model_size, device=device, default_dtype=dtype)
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

## Supported elements

MACE-OFF: H, C, N, O, F, S, Cl. For systems with other elements, use
`mace-mp`/`mace-omat` or a different MLIP family (NequIP/Allegro OAM models
cover the full periodic table — see
[references/nequip_allegro_usage.md](nequip_allegro_usage.md)).

## Known warnings

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`: safe to ignore, related to torch.load
- `cuequivariance not available`: means no GPU kernel acceleration for
  equivariant operations; MACE still runs correctly via standard torch ops

## References

- MACE-OFF paper: Kovács et al., JACS 2024, https://pubs.acs.org/doi/10.1021/jacs.4c07099
- MACE GitHub: https://github.com/ACEsuit/mace
- MACE-OFF models: https://github.com/ACEsuit/mace-off
