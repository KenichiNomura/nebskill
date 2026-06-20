"""Compute barriers and MACE-OFF vs DFT comparison. Writes report.json."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EV_TO_KCAL = 23.0609


def main():
    parser = argparse.ArgumentParser(description="Analyze converged NEB results")
    parser.add_argument("--config", default="assets/neb_defaults.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    neb_result   = json.loads((out_dir / "neb_result.json").read_text())
    relaxed      = json.loads((out_dir / "relaxed_endpoints.json").read_text())
    endpoints    = json.loads((out_dir / "endpoints.json").read_text())

    latest = neb_result["latest"]
    energies = latest["energies"]           # list: all images incl. endpoints

    ts_idx          = int(max(range(len(energies)), key=lambda i: energies[i]))
    e_ts            = energies[ts_idx]
    e_reactant      = energies[0]
    e_product       = energies[-1]

    forward_barrier_ev  = e_ts - e_reactant
    reverse_barrier_ev  = e_ts - e_product

    dft_forward = endpoints.get("dft_forward_barrier_ev")
    if dft_forward is not None:
        mace_error = forward_barrier_ev - dft_forward
        rel_error  = (mace_error / dft_forward * 100) if dft_forward != 0 else None
    else:
        mace_error = None
        rel_error  = None

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    report = {
        "run_id":                   endpoints.get("run_id", "custom"),
        "formula":                  endpoints["formula"],
        "rxn_key":                  endpoints["rxn_key"],
        "n_atoms":                  endpoints["n_atoms"],
        "mlip":                     relaxed.get("mlip", relaxed.get("mace_model_size", "unknown")),
        "n_images":                 neb_result["n_images"],
        "neb_method":               neb_result["method"],
        "forward_barrier_ev":       round(forward_barrier_ev,  4),
        "forward_barrier_kcal":     round(forward_barrier_ev * EV_TO_KCAL, 3),
        "reverse_barrier_ev":       round(reverse_barrier_ev,  4),
        "reverse_barrier_kcal":     round(reverse_barrier_ev * EV_TO_KCAL, 3),
        "ts_image_idx":             ts_idx,
        "e_reactant_ev":            round(e_reactant, 6),
        "e_ts_ev":                  round(e_ts,       6),
        "e_product_ev":             round(e_product,  6),
        "dft_forward_barrier_ev":   dft_forward,
        "mlip_vs_dft_error_ev":     round(mace_error, 4) if mace_error is not None else None,
        "mlip_vs_dft_relative_pct": round(rel_error, 2) if rel_error is not None else None,
        "neb_converged":            latest["converged"],
        "phase1_steps":             neb_result.get("phase1", {}).get("steps_taken"),
        "phase2_steps":             latest["steps_taken"],
        "phase2_fmax_final":        latest["fmax_final"],
        "neb_energies":             energies,
    }

    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"{report['run_id']} ({report['formula']}) — NEB analysis [{report['mlip']}]")
    print(f"  Forward barrier:  {forward_barrier_ev:.3f} eV  "
          f"({forward_barrier_ev * EV_TO_KCAL:.1f} kcal/mol)")
    print(f"  Reverse barrier:  {reverse_barrier_ev:.3f} eV  "
          f"({reverse_barrier_ev * EV_TO_KCAL:.1f} kcal/mol)")
    print(f"  TS image index:   {ts_idx}")
    if dft_forward is not None and mace_error is not None:
        print(f"  DFT reference:    {dft_forward:.3f} eV")
        err_str = f"  ({rel_error:+.1f}%)" if rel_error is not None else ""
        print(f"  MLIP error:       {mace_error:+.3f} eV{err_str}")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
