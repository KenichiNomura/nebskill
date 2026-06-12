# Step 1 — Load Reaction from Transition1x

Loads one reaction from the Transition1x HDF5 dataset, identifies the
transition state, validates the reaction, and extracts the two endpoint
structures for NEB.

## Script

```bash
python step1-load/load_dataset.py --reaction-id INT --config assets/neb_defaults.yaml
```

Output: writes `outputs/reaction_{id:04d}/endpoints.json` with:
- `reactant`: ASE Atoms dict (positions, numbers, cell, pbc)
- `product`: ASE Atoms dict
- `ts_frame_idx`: index of the TS frame in the trajectory
- `dft_barrier_ev`: DFT barrier height (forward, eV)
- `dft_energies`: full energy array along trajectory
- `n_atoms`: number of atoms
- `formula`: chemical formula

## Auto-download

If `data/Transition1x.h5` is missing, `download.py` is called automatically:

```bash
python step1-load/download.py
```

Downloads from `https://ndownloader.figshare.com/files/36035789` (~6.2 GB)
with a progress bar. Resumes interrupted downloads via HTTP Range requests.

## HDF5 structure

The Transition1x file is organized as:
```
/                          root
  rxn_{id}/                one group per reaction
    positions/             (n_frames, n_atoms, 3) Å
    atomic_numbers/        (n_atoms,)
    energies/              (n_frames,) eV  — ωB97x/6-31G*
    forces/                (n_frames, n_atoms, 3) eV/Å
```

See [references/transition1x_schema.md](../references/transition1x_schema.md)
for the full schema.

## TS detection and endpoint selection logic

1. Load energy array for the reaction
2. Find TS: `ts_idx = argmax(energies)`
3. **Edge check**: if `ts_idx < N` or `ts_idx > n_frames - N - 1` (N=2),
   double the planned `n_images` via interpolation (flag in endpoints.json)
4. **Multi-peak check**: detect secondary peaks; use global maximum only
5. **Barrier check**: compute `barrier = E[ts_idx] - max(E[0:ts_idx].min(), E[ts_idx+1:].min())`
   If `barrier < 0.1 eV`: skip reaction, log to queue.json as `skipped`, advance to next
6. Select endpoints:
   - `reactant` = frame with lowest energy in `[0 : ts_idx]`
   - `product`  = frame with lowest energy in `[ts_idx+1 : n_frames]`
7. Write `outputs/reaction_{id:04d}/endpoints.json`

## Error handling

- Missing HDF5 file → trigger auto-download
- Reaction group not found → log error, mark queue as `failed`
- Barrier too low → mark queue as `skipped`, move to next reaction
- Endpoint relaxation will handle non-minimum structures (mandatory in step 2)
