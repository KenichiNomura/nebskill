"""Aggregate NEB results across MLIPs into summary.json and comparison.png."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-MLIP NEB results"
    )
    parser.add_argument("--endpoints-dir", required=True,
                        help="Directory containing endpoints.json and per-MLIP subdirs")
    args = parser.parse_args()

    endpoints_dir = Path(args.endpoints_dir)
    endpoints = json.loads((endpoints_dir / "endpoints.json").read_text())

    # collect one report.json per MLIP subdirectory
    results: dict[str, dict] = {}
    for mlip_dir in sorted(endpoints_dir.iterdir()):
        if not mlip_dir.is_dir():
            continue
        report_path = mlip_dir / "report.json"
        if not report_path.exists():
            continue
        r = json.loads(report_path.read_text())
        results[mlip_dir.name] = {
            "forward_barrier_ev":   r.get("forward_barrier_ev"),
            "forward_barrier_kcal": r.get("forward_barrier_kcal"),
            "reverse_barrier_ev":   r.get("reverse_barrier_ev"),
            "reverse_barrier_kcal": r.get("reverse_barrier_kcal"),
            "ts_image_idx":         r.get("ts_image_idx"),
            "neb_converged":        r.get("neb_converged"),
            "phase2_fmax_final":    r.get("phase2_fmax_final"),
        }

    if not results:
        print("No report.json files found — nothing to aggregate.", file=sys.stderr)
        sys.exit(1)

    summary = {
        "run_id":      endpoints.get("run_id", endpoints.get("rxn_key", "unknown")),
        "formula":     endpoints["formula"],
        "n_atoms":     endpoints["n_atoms"],
        "is_periodic": endpoints.get("is_periodic", False),
        "n_mlips":     len(results),
        "mlips":       results,
    }

    summary_path = endpoints_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")

    # ── print table ──────────────────────────────────────────────────────────
    print(f"\n{'MLIP':<20} {'Fwd (eV)':>10} {'Rev (eV)':>10} {'Converged':>10}")
    print("─" * 55)
    for name, r in results.items():
        fwd = f"{r['forward_barrier_ev']:.3f}" if r["forward_barrier_ev"] is not None else "N/A"
        rev = f"{r['reverse_barrier_ev']:.3f}" if r["reverse_barrier_ev"] is not None else "N/A"
        conv = "yes" if r["neb_converged"] else "no"
        print(f"{name:<20} {fwd:>10} {rev:>10} {conv:>10}")

    # ── comparison plot ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        names = list(results.keys())
        fwd = [results[n]["forward_barrier_ev"] or 0.0 for n in names]
        rev = [results[n]["reverse_barrier_ev"] or 0.0 for n in names]
        conv = [results[n]["neb_converged"] for n in names]

        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))

        bars_fwd = ax.bar(x - 0.2, fwd, width=0.38, label="Forward",
                          color="steelblue")
        bars_rev = ax.bar(x + 0.2, rev, width=0.38, label="Reverse",
                          color="salmon")

        # grey out bars for non-converged MLIPs
        for i, c in enumerate(conv):
            if not c:
                bars_fwd[i].set_alpha(0.35)
                bars_rev[i].set_alpha(0.35)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Barrier height (eV)", fontsize=12)
        ax.set_title(
            f"NEB barriers across MLIPs — {endpoints['formula']} "
            f"({summary['run_id']})",
            fontsize=12,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        img_path = endpoints_dir / "comparison.png"
        fig.savefig(str(img_path), dpi=150)
        plt.close(fig)
        print(f"Comparison plot written to {img_path}")

    except ImportError:
        print("matplotlib not available — skipping comparison plot")


if __name__ == "__main__":
    main()
