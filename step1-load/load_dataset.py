"""Load one reaction from Transition1x, validate, and extract NEB endpoints."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from ase import Atoms


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_reaction_index(h5_path: Path, split: str = "data") -> list[tuple]:
    """Build a flat list of (split, formula, rxn_key) for sequential access."""
    index = []
    with h5py.File(h5_path, "r") as f:
        for formula in sorted(f[split].keys()):
            for rxn_key in sorted(f[split][formula].keys()):
                index.append((split, formula, rxn_key))
    return index


def _scalar_energy(dataset) -> float:
    return float(np.array(dataset).flat[0])


def load_reaction(h5_path: Path, split: str, formula: str, rxn_key: str) -> dict:
    """Read one reaction group from HDF5, using pre-optimized endpoint structures."""
    with h5py.File(h5_path, "r") as f:
        if split not in f or formula not in f[split] or rxn_key not in f[split][formula]:
            raise KeyError(f"Reaction {split}/{formula}/{rxn_key} not found in {h5_path}")
        rxn = f[split][formula][rxn_key]

        atomic_numbers = rxn["atomic_numbers"][:].astype(int)
        traj_positions = rxn["positions"][:]                        # (F, N, 3)
        traj_energies  = rxn["wB97x_6-31G(d).energy"][:]           # (F,)
        traj_forces    = rxn["wB97x_6-31G(d).forces"][:]           # (F, N, 3)

        reactant_pos = rxn["reactant"]["positions"][0]              # (N, 3)
        product_pos  = rxn["product"]["positions"][0]               # (N, 3)
        ts_pos       = rxn["transition_state"]["positions"][0]      # (N, 3)

        e_react = _scalar_energy(rxn["reactant"]["wB97x_6-31G(d).energy"])
        e_prod  = _scalar_energy(rxn["product"]["wB97x_6-31G(d).energy"])
        e_ts    = _scalar_energy(rxn["transition_state"]["wB97x_6-31G(d).energy"])

    return {
        "atomic_numbers": atomic_numbers,
        "traj_positions":  traj_positions,
        "traj_energies":   traj_energies,
        "traj_forces":     traj_forces,
        "reactant_pos":    reactant_pos,
        "product_pos":     product_pos,
        "ts_pos":          ts_pos,
        "e_react":         e_react,
        "e_prod":          e_prod,
        "e_ts":            e_ts,
    }


def atoms_dict(positions: np.ndarray, atomic_numbers: np.ndarray) -> dict:
    return {
        "positions":      positions.tolist(),
        "atomic_numbers": atomic_numbers.tolist(),
        "pbc":            False,
        "cell":           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    }


def validate_and_build_result(data: dict, cfg: dict,
                               reaction_id: int,
                               split: str, formula: str, rxn_key: str) -> dict:
    """Validate barrier and assemble endpoints.json payload."""
    min_barrier = cfg["filter"]["min_barrier_ev"]

    forward_barrier = data["e_ts"] - data["e_react"]
    reverse_barrier = data["e_ts"] - data["e_prod"]

    if forward_barrier < min_barrier and reverse_barrier < min_barrier:
        raise ValueError(
            f"Both barriers below {min_barrier} eV threshold "
            f"(fwd={forward_barrier:.3f}, rev={reverse_barrier:.3f}) — skipping"
        )

    atoms = Atoms(numbers=data["atomic_numbers"])
    formula_str = atoms.get_chemical_formula()

    return {
        "reaction_id":        reaction_id,
        "split":              split,
        "formula":            formula_str,
        "rxn_key":            rxn_key,
        "n_atoms":            int(len(data["atomic_numbers"])),
        "n_traj_frames":      int(len(data["traj_energies"])),
        "dft_forward_barrier_ev": round(float(forward_barrier), 6),
        "dft_reverse_barrier_ev": round(float(reverse_barrier), 6),
        "dft_e_reactant_ev":  round(float(data["e_react"]), 6),
        "dft_e_product_ev":   round(float(data["e_prod"]),  6),
        "dft_e_ts_ev":        round(float(data["e_ts"]),    6),
        "dft_traj_energies":  data["traj_energies"].tolist(),
        "reactant":           atoms_dict(data["reactant_pos"], data["atomic_numbers"]),
        "product":            atoms_dict(data["product_pos"],  data["atomic_numbers"]),
        "ts_reference":       atoms_dict(data["ts_pos"],       data["atomic_numbers"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Load reaction from Transition1x")
    parser.add_argument("--reaction-id", type=int, required=True,
                        help="Sequential integer reaction index (0-based)")
    parser.add_argument("--split", default="data",
                        help="Dataset split: data | train | val | test")
    parser.add_argument("--config", default="assets/neb_defaults.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    h5_path = Path(cfg["dataset"]["path"])

    if not h5_path.exists():
        print(f"Dataset not found at {h5_path}, downloading...")
        from step1_load.download import download
        download(h5_path)

    print(f"Building reaction index for split='{args.split}'...")
    index = build_reaction_index(h5_path, split=args.split)
    n_total = len(index)
    print(f"  {n_total} reactions found")

    if args.reaction_id >= n_total:
        print(f"ERROR: reaction_id {args.reaction_id} >= total {n_total}", file=sys.stderr)
        sys.exit(1)

    split, formula, rxn_key = index[args.reaction_id]
    print(f"Loading reaction {args.reaction_id}: {split}/{formula}/{rxn_key}")

    data = load_reaction(h5_path, split, formula, rxn_key)
    n_atoms  = len(data["atomic_numbers"])
    n_frames = len(data["traj_energies"])
    print(f"  Atoms: {n_atoms}, Trajectory frames: {n_frames}")
    print(f"  DFT energies — reactant: {data['e_react']:.4f}, "
          f"TS: {data['e_ts']:.4f}, product: {data['e_prod']:.4f} eV")

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(f"outputs/reaction_{args.reaction_id:04d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = validate_and_build_result(data, cfg, args.reaction_id,
                                           split, formula, rxn_key)
    except ValueError as e:
        print(f"SKIP: {e}", file=sys.stderr)
        skip_path = out_dir / "skip.json"
        skip_path.write_text(json.dumps(
            {"reaction_id": args.reaction_id, "reason": str(e)}, indent=2))
        sys.exit(2)

    out_path = out_dir / "endpoints.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"  Formula: {result['formula']}")
    print(f"  Forward barrier: {result['dft_forward_barrier_ev']:.3f} eV  |  "
          f"Reverse: {result['dft_reverse_barrier_ev']:.3f} eV")
    print(f"Endpoints written to {out_path}")


if __name__ == "__main__":
    main()
