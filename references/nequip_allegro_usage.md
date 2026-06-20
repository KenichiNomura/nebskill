# NequIP / Allegro Usage Reference

## What they are

NequIP and Allegro are equivariant message-passing (NequIP) and strictly
local (Allegro) interatomic potentials. The OAM (Open Architecture Models)
and MP foundation models cover the full periodic table and are distributed
as compiled artifacts rather than raw checkpoints.

## Deployment format in this skill

`package: nequip` (current NequIP 2.x) — models are AOT-compiled
`.nequip.pt2` files, produced once per GPU architecture:
```bash
python step0-models/download.py   # fetch raw weights from Zenodo
python step0-models/compile.py    # compile on a GPU node → *.nequip.pt2
```
`lib/calculator.py` loads these via
`NequIPCalculator.from_compiled_model(compiled, device=device)`.
Compilation must happen on the same GPU type used for NEB — a `.pt2`
compiled for one architecture will not load on another.

See `assets/mlip_registry.yaml` for the registered entries and their
checkpoint paths (`step0-models/models/`).

## Adding it to a NEB run

No code changes needed — just add the registry key to `--mlips` or
`assets/neb_defaults.yaml`'s `mlips:` list, e.g. `nequip-oam-l`,
`allegro-oam-l`, `nequip-oam-xl`, `allegro-mp-l`.

## Known gotchas

- `device: cuda` is hard-required for the compiled (`package: nequip`)
  entries in the registry — these checkpoints are AOT-compiled for GPU and
  will not run on CPU.
- e3nn 0.5.x's `constants.pt` loads via `torch.load()` without
  `weights_only=False`; PyTorch ≥ 2.6 defaults to `weights_only=True` and
  will reject it. `lib/calculator.py` works around this with
  `torch.serialization.add_safe_globals([slice])` before importing
  `nequip.ase.nequip_calculator`.
