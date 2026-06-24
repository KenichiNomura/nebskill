"""Two-phase NEB: standard NEB (phase 1) → CI-NEB (phase 2)."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import write as ase_write
from ase.mep import NEB
from ase.optimize import FIRE2, MDMin
from lib.calculator import make_calculator

MAX_NEB_GPUS = 4  # this system's GPUs/node (1 process/GPU memory budget)

# BFGS/LBFGS deliberately excluded: they assume the optimized force is a
# true gradient of one scalar function, which breaks down for CI-NEB phase 2
# (climbing image inverts the parallel force component). FIRE2 is the
# default; MDMin is the fallback when FIRE2 stalls (e.g. kinking).
OPTIMIZERS = {
    "FIRE2": FIRE2,
    "MDMin": MDMin,
}


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


def build_images(reactant: Atoms, product: Atoms, n_images: int, calc=None,
                 mlip_name: str | None = None, registry: dict | None = None,
                 n_gpus: int = 1) -> list:
    """Create the image list.

    n_gpus == 1: all images share one calculator instance (`calc`),
    sequential NEB on a single device — today's default behavior.

    n_gpus > 1: every image gets its own calculator instance, round-robin
    pinned to cuda:0..n_gpus-1, for NEB(parallel=True)'s threaded multi-GPU
    fallback (see ase/mep/neb.py — one thread per image when world.size==1).
    """
    images = [reactant.copy() for _ in range(n_images)]
    images[-1].positions = product.get_positions()
    if n_gpus == 1:
        for img in images:
            img.calc = calc
    else:
        for i, img in enumerate(images):
            device = f"cuda:{i % n_gpus}"
            img.calc = make_calculator(mlip_name, registry, device_override=device)
            print(f"    image {i} -> {device}")
    return images


def validate_n_gpus(n_gpus: int) -> None:
    if n_gpus < 1:
        raise SystemExit(f"ERROR: --n-gpus must be >= 1, got {n_gpus}")
    if n_gpus == 1:
        return
    if n_gpus > MAX_NEB_GPUS:
        raise SystemExit(
            f"ERROR: --n-gpus={n_gpus} exceeds this system's {MAX_NEB_GPUS}-GPU/node "
            f"budget. Multi-node parallel NEB is not supported."
        )
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: --n-gpus > 1 requires CUDA; no CUDA device available.")
    available = torch.cuda.device_count()
    if available < n_gpus:
        raise SystemExit(
            f"ERROR: --n-gpus={n_gpus} requested but only {available} CUDA "
            f"device(s) visible."
        )


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


def _drop_none_results(atoms: Atoms) -> None:
    """MatRIS always sets calc.results['magmoms'] even when the model didn't
    predict it (None), which crashes ASE's extxyz writer (it copies per-atom
    results into atoms.arrays without a None check). Strip None entries before
    writing. Gated to MatRIS so other calculators' results pass through untouched.
    """
    calc = atoms.calc
    if calc is None or type(calc).__name__ != "MatRISCalculator":
        return
    for key in [k for k, v in calc.results.items() if v is None]:
        del calc.results[key]


def write_trajectory(images: list, traj_path: Path, append: bool = False) -> None:
    mode = "a" if append else "w"
    for img in images:
        _drop_none_results(img)
        ase_write(str(traj_path), img, format="extxyz", append=(mode == "a"))
        mode = "a"


def run_phase(neb: NEB, images: list, fmax: float, max_steps: int,
              phase: int, traj_path: Path, append_traj: bool,
              optimizer_name: str = "FIRE2") -> dict:
    """Run one optimization phase with the chosen optimizer. Returns result dict."""
    t0 = time.monotonic()
    optimizer_cls = OPTIMIZERS[optimizer_name]
    opt = optimizer_cls(neb, logfile=None)
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
          f"optimizer={optimizer_name}, time={elapsed:.1f}s")

    return {
        "phase":            phase,
        "optimizer":        optimizer_name,
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
    parser.add_argument("--optimizer", default=None, choices=list(OPTIMIZERS),
                        help="Override NEB optimizer (agent retry use)")
    parser.add_argument("--n-gpus", type=int, default=None,
                        help="Spread one calculator per image across N GPUs "
                             "(round-robin cuda:0..N-1) via NEB(parallel=True). "
                             "Default 1 = current single-shared-calculator behavior.")
    parser.add_argument("--skip-cineb", dest="skip_cineb", action="store_true",
                        default=None, help="Force-skip Phase 2 (CI-NEB).")
    parser.add_argument("--run-cineb", dest="skip_cineb", action="store_false",
                        help="Force-run Phase 2 (CI-NEB), overriding config default.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    neb_cfg = cfg["neb"]

    # allow agent overrides
    if args.method:
        neb_cfg["method"] = args.method
    if args.spring_constant:
        neb_cfg["spring_constant"] = args.spring_constant
    if args.optimizer:
        neb_cfg["optimizer"] = args.optimizer

    n_gpus = args.n_gpus if args.n_gpus is not None else int(neb_cfg.get("n_gpus", 4))
    validate_n_gpus(n_gpus)

    skip_cineb = args.skip_cineb if args.skip_cineb is not None \
        else bool(neb_cfg.get("skip_cineb", False))

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
    mlip_name = args.mlip or "mace-omat"

    # deepmd-kit's pytorch backend resolves its CUDA device once at module
    # import time into a process-wide global (not per-instance), so per-image
    # device pinning for multi-GPU NEB silently collapses onto one GPU.
    # Cap up front instead of pretending these MLIPs scale across n_gpus.
    max_n_gpus = registry.get(mlip_name, {}).get("max_n_gpus")
    if max_n_gpus is not None and n_gpus > max_n_gpus:
        print(f"  [{mlip_name}] capping n_gpus {n_gpus} -> {max_n_gpus} "
              f"(registry max_n_gpus; see mlip_registry.yaml)")
        n_gpus = max_n_gpus

    calc = make_calculator(mlip_name, registry) if n_gpus == 1 else None

    n_images  = args.n_images if args.n_images else \
               compute_n_images(reactant, product, cfg, edge_flag)
    method    = neb_cfg["method"]
    k         = float(neb_cfg["spring_constant"])
    optimizer_name = neb_cfg.get("optimizer", "FIRE2")

    # periodic systems: disable remove_rotation_and_translation
    rrt = False if is_periodic else bool(neb_cfg["remove_rotation_translation"])
    print(f"NEB for {relaxed.get('formula', '?')} [{mlip_name}]"
          f"{' (periodic)' if is_periodic else ''}")
    print(f"  n_images={n_images}, method={method}, k={k} eV/Å, "
          f"optimizer={optimizer_name}, rrt={rrt}, n_gpus={n_gpus}")

    images = build_images(reactant, product, n_images, calc,
                          mlip_name=mlip_name, registry=registry, n_gpus=n_gpus)

    neb = NEB(images,
              k=k,
              method=method,
              climb=False,
              parallel=(n_gpus > 1),
              allow_shared_calculator=(n_gpus == 1),
              remove_rotation_and_translation=rrt)

    interp_method = neb_cfg.get("interpolation", "idpp")
    mic = is_periodic  # use minimum-image convention for periodic cells
    print(f"  Running {interp_method} interpolation (mic={mic})...")
    neb.interpolate(interp_method, mic=mic)

    traj_path = out_dir / "neb_trajectory.xyz"

    # --- Phase 1: standard NEB ---
    # When CI-NEB (phase 2) is skipped, phase 1 is the only convergence
    # check, so it must target phase2_fmax/phase2_max_steps (the intended
    # single-phase criterion) rather than the tighter phase1_fmax meant to
    # gate entry into phase 2 of the two-phase scheme.
    phase1_fmax = neb_cfg["phase2_fmax"] if skip_cineb else neb_cfg["phase1_fmax"]
    phase1_max_steps = neb_cfg["phase2_max_steps"] if skip_cineb else neb_cfg["phase1_max_steps"]
    print(f"  Phase 1: standard NEB → fmax < {phase1_fmax} eV/Å")
    result1 = run_phase(neb, images,
                        fmax=float(phase1_fmax),
                        max_steps=int(phase1_max_steps),
                        phase=1, traj_path=traj_path, append_traj=False,
                        optimizer_name=optimizer_name)

    if not result1["converged"]:
        _write_neb_result(out_dir, result1, n_images, method, k, optimizer_name,
                          relaxed["dft_forward_barrier_ev"])
        print("Phase 1 did not converge — triggering retry (step4-monitor)")
        sys.exit(4)  # exit code 4 = NEB convergence failure

    if skip_cineb:
        _write_neb_result(out_dir, result1, n_images, method, k, optimizer_name,
                          relaxed["dft_forward_barrier_ev"], phase1_result=result1)
        print("Skipping Phase 2 (CI-NEB) per config — Phase 1 result is final.")
        print(f"NEB converged. Trajectory written to {traj_path}")
        print(f"Results written to {out_dir / 'neb_result.json'}")
        return

    # --- Phase 2: CI-NEB ---
    print(f"  Phase 2: CI-NEB (climb=True) → fmax < {neb_cfg['phase2_fmax']} eV/Å")
    neb.climb = True
    result2 = run_phase(neb, images,
                        fmax=float(neb_cfg["phase2_fmax"]),
                        max_steps=int(neb_cfg["phase2_max_steps"]),
                        phase=2, traj_path=traj_path, append_traj=True,
                        optimizer_name=optimizer_name)

    _write_neb_result(out_dir, result2, n_images, method, k, optimizer_name,
                      relaxed["dft_forward_barrier_ev"],
                      phase1_result=result1)

    if not result2["converged"]:
        print("Phase 2 did not converge — triggering retry (step4-monitor)")
        sys.exit(4)

    print(f"NEB converged. Trajectory written to {traj_path}")
    print(f"Results written to {out_dir / 'neb_result.json'}")


def _write_neb_result(out_dir: Path, result: dict, n_images: int,
                      method: str, k: float, optimizer_name: str, dft_barrier: float,
                      phase1_result: dict | None = None) -> None:
    payload = {
        "n_images":         n_images,
        "method":           method,
        "spring_constant":  k,
        "optimizer":        optimizer_name,
        "dft_barrier_ev":   dft_barrier,
        "phase1":           phase1_result,
        "latest":           result,
    }
    path = out_dir / "neb_result.json"
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
