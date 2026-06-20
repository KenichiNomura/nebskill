"""Orchestrate a multi-MLIP NEB campaign over a fixed endpoint pair.

Reads endpoints.json produced by step1-load/load_xyz.py, then runs steps
2–5 for each MLIP in the configured list.

Usage:
    python step0-multi/runner.py \
        --endpoints-dir outputs/Config000-Config014 \
        [--mlips mace-mp chgnet sevennet] \
        [--mode interactive|slurm]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _ask_mode() -> str:
    print("\nHow would you like to run the NEB campaign?")
    print("  [1] Interactive — run MLIPs sequentially in this session  (default)")
    print("  [2] SLURM       — submit one job per MLIP as a SLURM array")
    choice = input("Select [1/2] (Enter = 1): ").strip()
    return "slurm" if choice == "2" else "interactive"


def _run(script: str, extra: list, allow_exit4: bool = False) -> int:
    cmd = [sys.executable, str(ROOT / script)] + extra
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0 and not (allow_exit4 and rc == 4):
        raise RuntimeError(f"{script} exited {rc}")
    return rc


def _submit_slurm(mlip: str, out_dir: Path, cfg: dict, registry_path: str) -> str:
    slurm = cfg.get("execution", {})
    venv_python = sys.executable
    script = f"""#!/bin/bash
#SBATCH --job-name=neb_{mlip}
#SBATCH --partition={slurm.get('slurm_partition', 'gpu')}
#SBATCH --nodes={slurm.get('slurm_nodes', 1)}
#SBATCH --gpus-per-node={slurm.get('slurm_gpus_per_node', 1)}
#SBATCH --time={slurm.get('slurm_time', '02:00:00')}
#SBATCH --account={slurm.get('slurm_account', '')}
#SBATCH --output={out_dir}/slurm_%j.out
cd {ROOT}
{venv_python} step2-relax/relax_endpoints.py --mlip {mlip} --output-dir {out_dir} --config assets/neb_defaults.yaml --registry {registry_path} || exit $?
{venv_python} step3-neb/neb_runner.py --mlip {mlip} --output-dir {out_dir} --config assets/neb_defaults.yaml --registry {registry_path}
rc=$?
if [ $rc -eq 4 ]; then
    {venv_python} step4-monitor/retry.py --mlip {mlip} --output-dir {out_dir} --config assets/neb_defaults.yaml --registry {registry_path} || exit $?
elif [ $rc -ne 0 ]; then
    exit $rc
fi
{venv_python} step5-analyze/analyze.py --output-dir {out_dir} --config assets/neb_defaults.yaml
{venv_python} step5-analyze/plot.py    --output-dir {out_dir}
{venv_python} step5-analyze/writer.py  --output-dir {out_dir}
"""
    job_sh = out_dir / "job.sh"
    job_sh.write_text(script)
    result = subprocess.run(["sbatch", str(job_sh)], capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else "???"
    print(f"    Submitted SLURM job {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Run NEB for multiple MLIPs over the same endpoint pair"
    )
    parser.add_argument("--endpoints-dir", required=True,
                        help="Directory containing endpoints.json")
    parser.add_argument("--config", default="assets/neb_defaults.yaml")
    parser.add_argument("--registry", default="assets/mlip_registry.yaml")
    parser.add_argument("--mlips", nargs="*", default=None,
                        help="Override MLIP list from config")
    parser.add_argument("--mode", choices=["interactive", "slurm"], default=None,
                        help="Execution mode (asked at runtime if omitted)")
    args = parser.parse_args()

    with open(args.config)   as f: cfg      = yaml.safe_load(f)
    with open(args.registry) as f: registry = yaml.safe_load(f)

    endpoints_dir = Path(args.endpoints_dir)
    endpoints     = json.loads((endpoints_dir / "endpoints.json").read_text())

    mlip_list = args.mlips or cfg.get("mlips", [])
    if not mlip_list:
        print("ERROR: no MLIPs specified in --mlips or config 'mlips:' list",
              file=sys.stderr)
        sys.exit(1)

    # validate all MLIPs against registry before starting
    unknown = [m for m in mlip_list if m not in registry]
    if unknown:
        print(f"ERROR: unknown MLIP(s) in list: {unknown}", file=sys.stderr)
        sys.exit(1)

    mode = args.mode or cfg.get("execution", {}).get("mode") or _ask_mode()
    print(f"\nCampaign: {endpoints['formula']} "
          f"({endpoints_dir.name}), {len(mlip_list)} MLIPs, mode={mode}")

    slurm_jobs: dict[str, str] = {}

    for mlip in mlip_list:
        out_dir = endpoints_dir / mlip
        out_dir.mkdir(parents=True, exist_ok=True)

        # each per-MLIP subdir needs its own endpoints.json copy so that the
        # existing step scripts (which resolve it from --output-dir) find it
        shutil.copy(endpoints_dir / "endpoints.json", out_dir / "endpoints.json")

        print(f"\n{'='*60}")
        print(f"  MLIP: {mlip}  →  {out_dir}")
        print(f"{'='*60}")

        if mode == "slurm":
            slurm_jobs[mlip] = _submit_slurm(mlip, out_dir, cfg, args.registry)
            continue

        # ── interactive sequential ────────────────────────────────────────────
        mlip_args    = ["--mlip", mlip, "--output-dir", str(out_dir),
                        "--config", args.config, "--registry", args.registry]
        analyze_args = ["--output-dir", str(out_dir), "--config", args.config]
        out_dir_args = ["--output-dir", str(out_dir)]
        try:
            _run("step2-relax/relax_endpoints.py", mlip_args)
            rc = _run("step3-neb/neb_runner.py",   mlip_args, allow_exit4=True)
            if rc == 4:
                _run("step4-monitor/retry.py",     mlip_args)
            _run("step5-analyze/analyze.py", analyze_args)
            _run("step5-analyze/plot.py",    out_dir_args)
            _run("step5-analyze/writer.py",  out_dir_args)
        except RuntimeError as exc:
            print(f"  FAILED for {mlip}: {exc}")

    if mode == "slurm":
        print(f"\nSubmitted {len(slurm_jobs)} SLURM jobs:")
        for m, jid in slurm_jobs.items():
            print(f"  {m}: job {jid}")
        print(f"\nRun after completion:")
        print(f"  python step5-analyze/aggregate.py "
              f"--endpoints-dir {endpoints_dir}")
    else:
        print("\nAll MLIPs done. Running aggregation …")
        subprocess.run(
            [sys.executable, str(ROOT / "step5-analyze/aggregate.py"),
             "--endpoints-dir", str(endpoints_dir)],
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
