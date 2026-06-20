"""Run NEB starting from a pre-existing set of image XYZ files.

Skips interpolation entirely — uses the provided files as the initial path.
Endpoints (first and last image) are fixed; all others are relaxed.

Usage:
    python step3-neb/neb_from_images.py \
        --images path/to/Config*.xyz \
        --output-dir outputs/run_name \
        --mlip mace-mp \
        --config assets/neb_defaults.yaml \
        --registry assets/mlip_registry.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from ase.io import read as ase_read
from ase.mep import NEB
from ase.optimize import FIRE2, MDMin

from lib.calculator import make_calculator

OPTIMIZERS = {"FIRE2": FIRE2, "MDMin": MDMin}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def neb_fmax_per_image(neb, images):
    n_internal = len(images) - 2
    n_atoms = len(images[0])
    forces = neb.get_forces().reshape(n_internal, n_atoms, 3)
    return [float(np.max(np.linalg.norm(f, axis=1))) for f in forces]


def write_trajectory(images, traj_path, append=False):
    from ase.io import write as ase_write
    mode = "a" if append else "w"
    for img in images:
        ase_write(str(traj_path), img, format="extxyz", append=(mode == "a"))
        mode = "a"


def run_phase(neb, images, fmax, max_steps, phase, traj_path, append_traj,
              optimizer_name="FIRE2"):
    t0 = time.monotonic()
    opt = OPTIMIZERS[optimizer_name](neb, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    steps_taken = opt.get_number_of_steps()
    elapsed = time.monotonic() - t0

    energies = ([float(images[0].get_potential_energy())]
                + [float(img.get_potential_energy()) for img in images[1:-1]]
                + [float(images[-1].get_potential_energy())])
    img_fmax = neb_fmax_per_image(neb, images)
    fmax_final = max(img_fmax) if img_fmax else 0.0

    write_trajectory(images, traj_path, append=append_traj)

    status = "converged" if converged else "NOT converged"
    print(f"  Phase {phase}: {status} — NEB fmax={fmax_final:.4f} eV/Å, "
          f"steps={steps_taken}/{max_steps}, time={elapsed:.1f}s")
    return converged, fmax_final, steps_taken, energies, img_fmax


def main():
    parser = argparse.ArgumentParser(
        description="NEB from pre-existing image files (no interpolation)"
    )
    parser.add_argument("--images",     nargs="+", required=True,
                        help="Ordered list of XYZ image files (first=reactant, last=product)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mlip",       default="mace-mp")
    parser.add_argument("--config",     default="assets/neb_defaults.yaml")
    parser.add_argument("--registry",   default="assets/mlip_registry.yaml")
    parser.add_argument("--spring-constant", type=float, default=None)
    parser.add_argument("--method",     default=None)
    parser.add_argument("--optimizer",  default=None, choices=list(OPTIMIZERS))
    parser.add_argument("--skip-cineb", action="store_true",
                        help="Run Phase 1 only; skip CI-NEB Phase 2.")
    parser.add_argument("--phase1-fmax", type=float, default=None,
                        help="Override Phase 1 convergence target (eV/Å).")
    parser.add_argument("--relaxed-endpoints", default=None,
                        help="Path to relaxed_endpoints.json (step 2 output). "
                             "If provided, replaces first and last image with "
                             "relaxed structures so endpoints are true PES minima.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    neb_cfg = cfg["neb"]
    with open(args.registry) as rf:
        registry = yaml.safe_load(rf)

    k      = args.spring_constant or float(neb_cfg["spring_constant"])
    method = args.method or neb_cfg["method"]
    optimizer_name = args.optimizer or neb_cfg.get("optimizer", "FIRE2")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load images in order
    image_paths = sorted(args.images)
    images = [ase_read(p) for p in image_paths]
    n_images = len(images)
    is_periodic = bool(images[0].pbc.any())
    rrt = False if is_periodic else bool(neb_cfg["remove_rotation_translation"])

    print(f"Loaded {n_images} images from {Path(image_paths[0]).name} "
          f"… {Path(image_paths[-1]).name}")
    print(f"System: {images[0].get_chemical_formula()}, periodic={is_periodic}")

    # Replace endpoints with relaxed structures if provided
    if args.relaxed_endpoints:
        from ase import Atoms as _Atoms
        rel = json.loads(Path(args.relaxed_endpoints).read_text())
        def _dict_to_atoms(d):
            return _Atoms(numbers=d["atomic_numbers"], positions=d["positions"],
                          pbc=d["pbc"], cell=d["cell"])
        images[0]  = _dict_to_atoms(rel["reactant"])
        images[-1] = _dict_to_atoms(rel["product"])
        print(f"Endpoints replaced with relaxed structures from {args.relaxed_endpoints}")

    # Attach calculator to all images
    calc = make_calculator(args.mlip, registry)
    for img in images:
        img.calc = calc

    print(f"NEB: method={method}, k={k} eV/Å, optimizer={optimizer_name}, rrt={rrt}")

    neb = NEB(images, k=k, method=method, climb=False,
              allow_shared_calculator=True,
              remove_rotation_and_translation=rrt)

    traj_path = out_dir / "neb_trajectory.xyz"

    # Phase 1
    p1_fmax = args.phase1_fmax or float(neb_cfg["phase1_fmax"])
    print(f"  Phase 1: standard NEB → fmax < {p1_fmax} eV/Å")
    conv1, fmax1, steps1, energies1, imgf1 = run_phase(
        neb, images,
        fmax=p1_fmax,
        max_steps=int(neb_cfg["phase1_max_steps"]),
        phase=1, traj_path=traj_path, append_traj=False,
        optimizer_name=optimizer_name)

    phase1_record = {"converged": bool(conv1), "steps_taken": int(steps1),
                     "fmax_final": float(fmax1), "energies": energies1,
                     "forces_per_image": imgf1}
    result = {"n_images": n_images, "method": method, "spring_constant": k,
              "optimizer": optimizer_name,
              "dft_barrier_ev": None, "phase1": phase1_record,
              "latest": {"phase": 1, **phase1_record}}
    (out_dir / "neb_result.json").write_text(json.dumps(result, indent=2))

    if not conv1:
        print("Phase 1 did not converge.")
        sys.exit(4)

    e0 = energies1[0]
    rel = [e - e0 for e in energies1]
    ts_img = rel.index(max(rel))
    print(f"  Phase 1 barrier estimate: {max(rel):.4f} eV at image {ts_img}")

    if args.skip_cineb:
        print("Skipping CI-NEB (--skip-cineb). Phase 1 result saved.")
        sys.exit(0)

    # Phase 2: CI-NEB
    print(f"  Phase 2: CI-NEB (climb=True) → fmax < {neb_cfg['phase2_fmax']} eV/Å")
    neb.climb = True
    conv2, fmax2, steps2, energies2, imgf2 = run_phase(
        neb, images,
        fmax=float(neb_cfg["phase2_fmax"]),
        max_steps=int(neb_cfg["phase2_max_steps"]),
        phase=2, traj_path=traj_path, append_traj=True,
        optimizer_name=optimizer_name)

    result["latest"] = {"phase": 2, "converged": bool(conv2), "steps_taken": int(steps2),
                        "fmax_final": float(fmax2), "energies": energies2,
                        "forces_per_image": imgf2}
    (out_dir / "neb_result.json").write_text(json.dumps(result, indent=2))

    if not conv2:
        print("Phase 2 (CI-NEB) did not converge.")
        sys.exit(4)

    print(f"NEB converged. Trajectory → {traj_path}")


if __name__ == "__main__":
    main()
