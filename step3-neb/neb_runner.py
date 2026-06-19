"""Two-phase NEB: standard NEB (phase 1) → CI-NEB (phase 2)."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from ase import Atoms
from ase.io import write as ase_write
from ase.mep import NEB
from ase.optimize import FIRE
from lib.calculator import make_calculator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dict_to_atoms(d: dict) -> Atoms:
    return Atoms(
        numbers=d["atomic_numbers"],
        positions=d["positions"],
        pbc=d["pbc"],
        cell=d["cell"],
    )


def compute_n_images(reactant: Atoms, product: Atoms, cfg: dict,
                     edge_flag: bool = False) -> int:
    """Estimate n_images from reactant→product displacement.

    For periodic systems uses minimum-image convention so atoms that crossed
    a cell boundary don't inflate the apparent path length.
    """
    neb_cfg = cfg["neb"]
    if neb_cfg["n_images"] != "auto":
        n = int(neb_cfg["n_images"])
    else:
        if reactant.pbc.any() and not np.allclose(reactant.get_cell(), 0):
            # minimum-image displacement: wrap into [-L/2, L/2] per axis
            cell = reactant.get_cell()
            disp = product.positions - reactant.positions
            # fractional coords → wrap → Cartesian
            frac = np.linalg.solve(cell.T, disp.T).T
            frac -= np.round(frac)
            disp = frac @ cell
        else:
            disp = product.positions - reactant.positions
        path_length = float(np.sum(np.linalg.norm(disp, axis=1)))
        n = max(int(neb_cfg["n_images_min"]),
                round(path_length / float(neb_cfg["images_per_angstrom"])))
    if edge_flag:
        n = n * 2
        print(f"  Edge flag set — doubling n_images to {n}")
    return n


def build_images(reactant: Atoms, product: Atoms, n_images: int, calc) -> list:
    """Create image list; all images share one calculator (sequential NEB)."""
    images = [reactant.copy() for _ in range(n_images)]
    images[-1].positions = product.get_positions()
    for img in images:
        img.calc = calc
    return images


def neb_fmax_per_image(neb: NEB, images: list) -> list[float]:
    """Max NEB force magnitude per internal image (nudged forces, not raw)."""
    n_internal = len(images) - 2
    n_atoms = len(images[0])
    forces = neb.get_forces().reshape(n_internal, n_atoms, 3)
    return [float(np.max(np.linalg.norm(f, axis=1))) for f in forces]


def inter_image_rmsd(images: list) -> list[float]:
    """RMSD between consecutive images."""
    rmsds = []
    for a, b in zip(images[:-1], images[1:]):
        diff = a.get_positions() - b.get_positions()
        rmsds.append(float(np.sqrt(np.mean(diff ** 2))))
    return rmsds


def write_trajectory(images: list, traj_path: Path, append: bool = False) -> None:
    mode = "a" if append else "w"
    for img in images:
        ase_write(str(traj_path), img, format="extxyz", append=(mode == "a"))
        mode = "a"


def run_phase(neb: NEB, images: list, fmax: float, max_steps: int,
              phase: int, traj_path: Path, append_traj: bool) -> dict:
    """Run one FIRE optimization phase. Returns result dict."""
    t0 = time.monotonic()
    opt = FIRE(neb, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    steps_taken = opt.get_number_of_steps()
    elapsed = time.monotonic() - t0

    energies = ([float(images[0].get_potential_energy())]
                + [float(img.get_potential_energy()) for img in images[1:-1]]
                + [float(images[-1].get_potential_energy())])
    img_fmax = neb_fmax_per_image(neb, images)  # nudged NEB forces
    rmsds    = inter_image_rmsd(images)
    fmax_final = max(img_fmax) if img_fmax else 0.0

    write_trajectory(images, traj_path, append=append_traj)

    print(f"  Phase {phase}: {'converged' if converged else 'NOT converged'} — "
          f"NEB fmax={fmax_final:.4f} eV/Å, steps={steps_taken}/{max_steps}, "
          f"time={elapsed:.1f}s")

    return {
        "phase":            phase,
        "converged":        bool(converged),
        "steps_taken":      steps_taken,
        "max_steps":        max_steps,
        "fmax_target":      fmax,
        "fmax_final":       fmax_final,
        "wall_time_s":      round(elapsed, 2),
        "energies":         energies,
        "forces_per_image": img_fmax,
        "inter_image_rmsd": rmsds,
    }


def main():
    parser = argparse.ArgumentParser(description="Run two-phase NEB with chosen MLIP")
    parser.add_argument("--config",   default="assets/neb_defaults.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mlip",     default=None,
                        help="MLIP name from registry")
    parser.add_argument("--registry", default="assets/mlip_registry.yaml")
    parser.add_argument("--n-images", type=int, default=None,
                        help="Override n_images (agent retry use)")
    parser.add_argument("--method", default=None,
                        help="Override NEB method (agent retry use)")
    parser.add_argument("--spring-constant", type=float, default=None,
                        help="Override spring constant k (agent retry use)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    neb_cfg = cfg["neb"]

    # allow agent overrides
    if args.method:
        neb_cfg["method"] = args.method
    if args.spring_constant:
        neb_cfg["spring_constant"] = args.spring_constant

    out_dir = Path(args.output_dir)
    relaxed_path = out_dir / "relaxed_endpoints.json"

    if not relaxed_path.exists():
        print(f"ERROR: {relaxed_path} not found — run step2-relax first", file=sys.stderr)
        sys.exit(1)

    relaxed = json.loads(relaxed_path.read_text())
    endpoints_path = out_dir / "endpoints.json"
    endpoints = json.loads(endpoints_path.read_text())
    edge_flag   = endpoints.get("edge_flag", False)   # legacy
    is_periodic = endpoints.get("is_periodic", False)

    reactant = dict_to_atoms(relaxed["reactant"])
    product  = dict_to_atoms(relaxed["product"])

    with open(args.registry) as rf:
        registry = yaml.safe_load(rf)
    mlip_name = args.mlip or "mace-off"
    calc = make_calculator(mlip_name, registry)

    n_images = args.n_images if args.n_images else \
               compute_n_images(reactant, product, cfg, edge_flag)
    method   = neb_cfg["method"]
    k        = float(neb_cfg["spring_constant"])

    # periodic systems: disable remove_rotation_and_translation
    rrt = False if is_periodic else bool(neb_cfg["remove_rotation_translation"])
    print(f"NEB for {relaxed.get('formula', '?')} [{mlip_name}]"
          f"{' (periodic)' if is_periodic else ''}")
    print(f"  n_images={n_images}, method={method}, k={k} eV/Å, rrt={rrt}")

    images = build_images(reactant, product, n_images, calc)

    neb = NEB(images,
              k=k,
              method=method,
              climb=False,
              allow_shared_calculator=True,
              remove_rotation_and_translation=rrt)

    interp_method = neb_cfg.get("interpolation", "idpp")
    mic = is_periodic  # use minimum-image convention for periodic cells
    print(f"  Running {interp_method} interpolation (mic={mic})...")
    neb.interpolate(interp_method, mic=mic)

    traj_path = out_dir / "neb_trajectory.xyz"

    # --- Phase 1: standard NEB ---
    print(f"  Phase 1: standard NEB → fmax < {neb_cfg['phase1_fmax']} eV/Å")
    result1 = run_phase(neb, images,
                        fmax=float(neb_cfg["phase1_fmax"]),
                        max_steps=int(neb_cfg["phase1_max_steps"]),
                        phase=1, traj_path=traj_path, append_traj=False)

    if not result1["converged"]:
        _write_neb_result(out_dir, result1, n_images, method, k,
                          relaxed["dft_forward_barrier_ev"])
        print("Phase 1 did not converge — triggering retry (step4-monitor)")
        sys.exit(4)  # exit code 4 = NEB convergence failure

    # --- Phase 2: CI-NEB ---
    print(f"  Phase 2: CI-NEB (climb=True) → fmax < {neb_cfg['phase2_fmax']} eV/Å")
    neb.climb = True
    result2 = run_phase(neb, images,
                        fmax=float(neb_cfg["phase2_fmax"]),
                        max_steps=int(neb_cfg["phase2_max_steps"]),
                        phase=2, traj_path=traj_path, append_traj=True)

    _write_neb_result(out_dir, result2, n_images, method, k,
                      relaxed["dft_forward_barrier_ev"],
                      phase1_result=result1)

    if not result2["converged"]:
        print("Phase 2 did not converge — triggering retry (step4-monitor)")
        sys.exit(4)

    print(f"NEB converged. Trajectory written to {traj_path}")
    print(f"Results written to {out_dir / 'neb_result.json'}")


def _write_neb_result(out_dir: Path, result: dict, n_images: int,
                      method: str, k: float, dft_barrier: float,
                      phase1_result: dict | None = None) -> None:
    payload = {
        "n_images":         n_images,
        "method":           method,
        "spring_constant":  k,
        "dft_barrier_ev":   dft_barrier,
        "phase1":           phase1_result,
        "latest":           result,
    }
    path = out_dir / "neb_result.json"
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
