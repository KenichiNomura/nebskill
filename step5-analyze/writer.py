"""Write convergence.log from neb_result.json phase data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Write NEB convergence log")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    neb_result = json.loads((out_dir / "neb_result.json").read_text())

    lines = ["phase\tsteps\tfmax_target\tfmax_final\tconverged\twall_time_s"]

    phase1 = neb_result.get("phase1")
    if phase1:
        lines.append(
            f"1\t{phase1['steps_taken']}\t{phase1['fmax_target']}\t"
            f"{phase1['fmax_final']:.6f}\t{phase1['converged']}\t{phase1['wall_time_s']}"
        )

    latest = neb_result["latest"]
    lines.append(
        f"{latest['phase']}\t{latest['steps_taken']}\t{latest['fmax_target']}\t"
        f"{latest['fmax_final']:.6f}\t{latest['converged']}\t{latest['wall_time_s']}"
    )

    out_path = out_dir / "convergence.log"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Convergence log written to {out_path}")

    # confirm trajectory exists
    traj_path = out_dir / "neb_trajectory.xyz"
    if traj_path.exists():
        n_frames = traj_path.read_text().count("Properties=")
        print(f"NEB trajectory: {traj_path} ({n_frames} frames)")
    else:
        print(f"WARNING: trajectory not found at {traj_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
