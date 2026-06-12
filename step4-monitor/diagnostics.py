"""CLI wrapper — compute NEB diagnostics from neb_result.json."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.neb_diagnostics import diagnose


def main():
    parser = argparse.ArgumentParser(description="Compute NEB diagnostics")
    parser.add_argument("--reaction-id", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(f"outputs/reaction_{args.reaction_id:04d}")

    neb_result_path = out_dir / "neb_result.json"
    if not neb_result_path.exists():
        raise FileNotFoundError(f"{neb_result_path} not found")

    neb_result = json.loads(neb_result_path.read_text())
    payload = diagnose(neb_result)

    out_path = out_dir / "diagnostics.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Failure mode: {payload['failure_mode']}")
    print(f"  fmax={payload['fmax_final']:.4f} (target {payload['fmax_target']}), "
          f"steps={payload['steps_taken']}/{payload['max_steps']}")
    print(f"  Energy kink score: {payload['energy_smoothness']['max_abs_d2']:.4f} eV")
    print(f"  Image spacing CV:  {payload['image_spacing']['cv']:.3f}")
    print(f"Diagnostics written to {out_path}")


if __name__ == "__main__":
    main()
