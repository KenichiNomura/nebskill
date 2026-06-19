"""Load NEB endpoints from two user-provided structure files.

Handles both periodic (bulk/surface, pbc=T T T) and isolated/molecular
(pbc=F F F) systems.

Usage:
    python step1-load/load_xyz.py \
        --initial path/to/Config000.xyz \
        --final   path/to/Config014.xyz

Output:
    outputs/{run_id}/endpoints.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read


def atoms_to_dict(atoms: Atoms) -> dict:
    return {
        "positions":      atoms.get_positions().tolist(),
        "atomic_numbers": atoms.get_atomic_numbers().tolist(),
        "pbc":            atoms.pbc.tolist(),
        "cell":           atoms.get_cell().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build endpoints.json from two XYZ files"
    )
    parser.add_argument("--initial",    required=True, help="Initial config XYZ")
    parser.add_argument("--final",      required=True, help="Final config XYZ")
    parser.add_argument("--run-id",     default=None,
                        help="Run label for output dir (default: derived from filenames)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    initial = ase_read(args.initial)
    final   = ase_read(args.final)

    # ── validation ────────────────────────────────────────────────────────────
    if len(initial) != len(final):
        raise ValueError(
            f"Atom count mismatch: {args.initial} has {len(initial)} atoms, "
            f"{args.final} has {len(final)}"
        )
    if not np.array_equal(initial.get_atomic_numbers(), final.get_atomic_numbers()):
        raise ValueError(
            "Species order mismatch between initial and final structures. "
            "Atoms must appear in the same order in both files."
        )

    # ── periodicity detection ─────────────────────────────────────────────────
    # Periodic if any pbc component is True AND the cell is non-zero.
    # Applies remove_rotation_and_translation=False during NEB for bulk/surface.
    is_periodic = bool(initial.pbc.any()) and not np.allclose(initial.get_cell(), 0)

    formula = Atoms(numbers=initial.get_atomic_numbers()).get_chemical_formula()

    run_id = args.run_id or (
        f"{Path(args.initial).stem}-{Path(args.final).stem}"
    )
    out_dir = (
        Path(args.output_dir) if args.output_dir else Path(f"outputs/{run_id}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id":                 run_id,
        "formula":                formula,
        "rxn_key":                "custom",
        "n_atoms":                len(initial),
        "initial_xyz":            str(Path(args.initial).resolve()),
        "final_xyz":              str(Path(args.final).resolve()),
        "is_periodic":            is_periodic,
        # No DFT reference — skips MACE-vs-DFT comparison in analyze/plot
        "dft_forward_barrier_ev": None,
        "dft_reverse_barrier_ev": None,
        "ts_reference":           None,
        "reactant":               atoms_to_dict(initial),
        "product":                atoms_to_dict(final),
    }

    out_path = out_dir / "endpoints.json"
    out_path.write_text(json.dumps(result, indent=2))

    system_type = "periodic (bulk/surface)" if is_periodic else "isolated/molecular"
    rrt = "False" if is_periodic else "True"
    print(f"System:  {formula}, {len(initial)} atoms, {system_type}")
    print(f"  remove_rotation_and_translation during NEB: {rrt}")
    print(f"Endpoints written to {out_path}")


if __name__ == "__main__":
    main()
