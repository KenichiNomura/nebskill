# Transition1x Dataset Schema

## Overview

Transition1x contains DFT calculations (ωB97x/6-31G*) for 9.6 million
molecular configurations along and around minimum energy paths for ~20,000
organic reactions involving H, C, N, O, F.

File: `data/Transition1x.h5` (~6.2 GB)
Download: `https://ndownloader.figshare.com/files/36035789`
Total reactions: 20,146 (across data/train/val/test splits)

Reference: Schreiner et al., Scientific Data 2022,
https://www.nature.com/articles/s41597-022-01870-w

## HDF5 structure

```
/
  data/                         split (also: train, val, test)
    {formula}/                  molecular formula, e.g. C2H2N2O
      {rxn_id}/                 reaction ID, e.g. rxn2091
        atomic_numbers          (N,)         int32  — atom types
        positions               (F, N, 3)    float64 — trajectory frames, Å
        wB97x_6-31G(d).energy   (F,)         float64 — per-frame energies, eV
        wB97x_6-31G(d).forces   (F, N, 3)    float64 — per-frame forces, eV/Å
        reactant/               pre-optimized DFT reactant
          atomic_numbers        (N,)
          positions             (1, N, 3)    — squeeze to (N, 3)
          wB97x_6-31G(d).energy scalar       — eV
          wB97x_6-31G(d).forces (1, N, 3)
        product/                pre-optimized DFT product
          (same as reactant/)
        transition_state/       pre-optimized DFT transition state
          (same as reactant/)
```

## Key design note

Pre-optimized `reactant/`, `product/`, and `transition_state/` subgroups
provide DFT-level minimum structures directly. Use these as NEB endpoints
rather than scanning the trajectory for lowest-energy frames.
The trajectory is useful for context (energy profile, path character) and
for providing the LLM agent with convergence diagnostics.

## Accessing with h5py

```python
import h5py
import numpy as np

with h5py.File('data/Transition1x.h5', 'r') as f:
    rxn = f['data']['C2H2N2O']['rxn2091']

    # pre-optimized endpoints (use these for NEB)
    reactant_pos = rxn['reactant']['positions'][0]       # (N, 3)
    product_pos  = rxn['product']['positions'][0]        # (N, 3)
    ts_pos       = rxn['transition_state']['positions'][0]

    e_react = float(np.array(rxn['reactant']['wB97x_6-31G(d).energy']).flat[0])
    e_prod  = float(np.array(rxn['product']['wB97x_6-31G(d).energy']).flat[0])
    e_ts    = float(np.array(rxn['transition_state']['wB97x_6-31G(d).energy']).flat[0])

    forward_barrier = e_ts - e_react   # eV
    reverse_barrier = e_ts - e_prod    # eV

    atomic_numbers = rxn['atomic_numbers'][:]    # (N,)
```

## Enumerating reactions (flat integer index)

Reactions are identified by `(split, formula, rxn_key)` triples. For
sequential access, build a flat index at startup:

```python
def build_reaction_index(h5_path, split='data'):
    index = []
    with h5py.File(h5_path, 'r') as f:
        for formula in sorted(f[split].keys()):
            for rxn_key in sorted(f[split][formula].keys()):
                index.append((split, formula, rxn_key))
    return index  # index[i] → (split, formula, rxn_key) for reaction i
```

## Energy units

All energies are in eV. Forces are in eV/Å. Positions are in Å.
These match ASE conventions directly — no unit conversion needed.

## Splits

- `data`: full dataset (all reactions, 20,146 total)
- `train` / `val` / `test`: standard ML splits of the same reactions
- For NEB screening, use `data` (or `train` to stay within the training set)
