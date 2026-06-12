"""Aggregate results from completed NEB batch jobs.

Reads all outputs/reaction_*/report.json and produces:
  outputs/summary.json  — per-reaction table + aggregate statistics
  outputs/summary.png   — barrier distribution + DFT parity plot

Usage:
  python step0-batch/aggregate.py
  python step0-batch/aggregate.py --output-dir /path/to/outputs
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV_TO_KCAL = 23.0609


def load_reports(outputs_dir: Path) -> list[dict]:
    reports = []
    for p in sorted(outputs_dir.glob("reaction_*/report.json")):
        try:
            reports.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"  WARNING: could not read {p}: {e}")
    return reports


def compute_stats(errors: list[float]) -> dict:
    if not errors:
        return {}
    n = len(errors)
    mae  = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    mean = sum(errors) / n
    return {"n": n, "mae_ev": mae, "rmse_ev": rmse, "mean_bias_ev": mean}


def build_summary(reports: list[dict]) -> dict:
    rows = []
    errors = []
    converged = []

    for r in reports:
        row = {
            "reaction_id":          r.get("reaction_id"),
            "formula":              r.get("formula"),
            "forward_barrier_ev":   r.get("forward_barrier_ev"),
            "forward_barrier_kcal": r.get("forward_barrier_kcal"),
            "reverse_barrier_ev":   r.get("reverse_barrier_ev"),
            "dft_forward_barrier_ev": r.get("dft_forward_barrier_ev"),
            "mace_vs_dft_error_ev": r.get("mace_vs_dft_error_ev"),
            "mace_vs_dft_relative_pct": r.get("mace_vs_dft_relative_pct"),
            "ts_image_idx":         r.get("ts_image_idx"),
            "n_images":             r.get("n_images"),
            "neb_method":           r.get("neb_method"),
            "mace_model_size":      r.get("mace_model_size"),
        }
        rows.append(row)
        err = r.get("mace_vs_dft_error_ev")
        if err is not None:
            errors.append(err)
        converged.append(True)

    return {
        "n_reactions": len(rows),
        "statistics": compute_stats(errors),
        "reactions": rows,
    }


def plot_summary(reports: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dft    = np.array([r["dft_forward_barrier_ev"]  for r in reports
                       if r.get("dft_forward_barrier_ev") is not None])
    mace   = np.array([r["forward_barrier_ev"]       for r in reports
                       if r.get("forward_barrier_ev") is not None])
    errors = mace - dft

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: parity plot ──────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(dft, mace, alpha=0.7, s=40, edgecolors="steelblue",
               facecolors="lightsteelblue")
    lim = (min(dft.min(), mace.min()) - 0.2, max(dft.max(), mace.max()) + 0.2)
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("DFT barrier (eV)", fontsize=12)
    ax.set_ylabel("MACE-OFF barrier (eV)", fontsize=12)
    ax.set_title("MACE-OFF vs DFT parity", fontsize=13)

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ax.text(0.05, 0.92, f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7))
    ax.legend(fontsize=9)

    # ── right: error distribution ──────────────────────────────────────────
    ax = axes[1]
    ax.hist(errors, bins=min(20, max(5, len(errors) // 3)),
            color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.axvline(float(np.mean(errors)), color="red", lw=1.5,
               linestyle="--", label=f"mean = {np.mean(errors):.3f} eV")
    ax.set_xlabel("MACE − DFT error (eV)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Barrier error distribution", fontsize=13)
    ax.legend(fontsize=9)

    fig.suptitle(f"NEB Batch Summary  (n = {len(dft)})", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary plot written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate NEB batch results")
    parser.add_argument("--output-dir", default=None,
                        help="outputs directory (default: <repo>/outputs)")
    args = parser.parse_args()

    outputs_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs"
    if not outputs_dir.exists():
        print(f"No outputs directory found at {outputs_dir}")
        import sys; sys.exit(1)

    print(f"Scanning {outputs_dir} for completed reactions...")
    reports = load_reports(outputs_dir)
    print(f"  Found {len(reports)} report(s).")

    if not reports:
        print("Nothing to aggregate.")
        return

    summary = build_summary(reports)

    summary_json = outputs_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary written to {summary_json}")

    stats = summary["statistics"]
    if stats:
        print(f"\nAggregate statistics ({stats['n']} reactions):")
        print(f"  MAE       = {stats['mae_ev']:.4f} eV  "
              f"({stats['mae_ev'] * EV_TO_KCAL:.2f} kcal/mol)")
        print(f"  RMSE      = {stats['rmse_ev']:.4f} eV  "
              f"({stats['rmse_ev'] * EV_TO_KCAL:.2f} kcal/mol)")
        print(f"  Mean bias = {stats['mean_bias_ev']:+.4f} eV")

    # plot only if matplotlib is available
    try:
        plot_summary(reports, outputs_dir / "summary.png")
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
