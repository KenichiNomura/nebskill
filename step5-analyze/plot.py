"""Generate energy_profile.png from converged NEB results."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EV_TO_KCAL = 23.0609


def main():
    parser = argparse.ArgumentParser(description="Plot NEB energy profile")
    parser.add_argument("--reaction-id", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(f"outputs/reaction_{args.reaction_id:04d}")

    report = json.loads((out_dir / "report.json").read_text())
    energies = np.array(report["neb_energies"])
    e_rel    = energies - energies[0]          # relative to reactant
    ts_idx   = report["ts_image_idx"]
    fwd_ev   = report["forward_barrier_ev"]
    rev_ev   = report["reverse_barrier_ev"]
    dft_ev   = report["dft_forward_barrier_ev"]
    n        = len(energies)

    fig, ax = plt.subplots(figsize=(8, 5))

    # main NEB path
    x = np.arange(n)
    ax.plot(x, e_rel, "o-", color="steelblue", lw=2, ms=7, label="MACE-OFF NEB")

    # TS marker
    ax.plot(ts_idx, e_rel[ts_idx], "*", color="red", ms=16, zorder=5,
            label=f"TS (image {ts_idx})")

    # forward barrier annotation
    ax.annotate(
        f"Fwd: {fwd_ev:.3f} eV\n({fwd_ev * EV_TO_KCAL:.1f} kcal/mol)",
        xy=(ts_idx, e_rel[ts_idx]),
        xytext=(ts_idx + 0.5, e_rel[ts_idx] * 0.7),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9, color="red",
    )

    # reverse barrier annotation
    ax.annotate(
        f"Rev: {rev_ev:.3f} eV",
        xy=(ts_idx, e_rel[ts_idx]),
        xytext=(ts_idx - 1.5, e_rel[ts_idx] * 0.6),
        arrowprops=dict(arrowstyle="->", color="darkorange"),
        fontsize=9, color="darkorange",
    )

    # DFT reference barrier as dashed line
    ax.axhline(dft_ev, color="gray", linestyle="--", lw=1.5,
               label=f"DFT ref: {dft_ev:.3f} eV (ωB97x/6-31G*)")

    ax.set_xlabel("Image index", fontsize=12)
    ax.set_ylabel("Energy relative to reactant (eV)", fontsize=12)
    ax.set_title(
        f"NEB energy profile — {report['formula']} (rxn {report['reaction_id']})",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "energy_profile.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Energy profile written to {out_path}")


if __name__ == "__main__":
    main()
