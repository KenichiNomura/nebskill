"""Submit SLURM jobs for pending NEB reactions.

Usage:
  python step0-batch/submit.py --n-jobs 5
  python step0-batch/submit.py --n-jobs 5 --dry-run
  python step0-batch/submit.py --n-jobs 5 --reaction-ids 10,11,12
"""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_FILE = ROOT / "queue.json"
JOB_TEMPLATE = Path(__file__).resolve().parent / "job_template.sh"
VENV_PYTHON = ROOT.parent.parent / ".venv" / "bin" / "python"
# fall back to venv adjacent to repo root
if not VENV_PYTHON.exists():
    VENV_PYTHON = ROOT.parent / ".venv" / "bin" / "python"


def claim_pending(n: int, specific_ids: list[int] | None) -> list[dict]:
    """
    Atomically claim up to n pending reactions from queue.json.
    Returns the claimed entries (with status still pending — caller sets running).
    """
    claimed = []
    with open(QUEUE_FILE, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        data = json.load(fh)
        for r in data["reactions"]:
            if len(claimed) >= n:
                break
            if r["status"] != "pending":
                continue
            if specific_ids and r["id"] not in specific_ids:
                continue
            r["status"] = "running"
            claimed.append(r)
        fh.seek(0)
        json.dump(data, fh, indent=2)
        fh.write("\n")
        fh.truncate()
        fcntl.flock(fh, fcntl.LOCK_UN)
    return claimed


def build_sbatch_cmd(reaction: dict, cfg: dict, config_path: str,
                     dry_run: bool) -> list[str]:
    """Build the sbatch command for one reaction."""
    batch = cfg.get("batch", {})
    partition   = batch.get("slurm_partition", "gpu")
    time_limit  = batch.get("slurm_time", "02:00:00")
    nodes       = batch.get("slurm_nodes", 1)
    gpus        = batch.get("slurm_gpus_per_node", 1)
    account     = batch.get("slurm_account", "")

    rid = reaction["id"]
    job_name = f"neb_{rid:04d}"
    log_dir  = ROOT / f"outputs/reaction_{rid:04d}"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--time={time_limit}",
        f"--nodes={nodes}",
        f"--gpus-per-node={gpus}",
        f"--output={log_dir}/slurm_%j.out",
        f"--error={log_dir}/slurm_%j.err",
        f"--export=ALL,REACTION_ID={rid},CONFIG={config_path},NEB_ROOT={ROOT}",
    ]
    if account:
        cmd.append(f"--account={account}")

    cmd.append(str(JOB_TEMPLATE))
    return cmd


def update_job_id(reaction_id: int, slurm_job: str) -> None:
    """Write the SLURM job ID back to queue.json (already set running)."""
    with open(QUEUE_FILE, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        data = json.load(fh)
        for r in data["reactions"]:
            if r["id"] == reaction_id:
                r["slurm_job"] = slurm_job
                break
        fh.seek(0)
        json.dump(data, fh, indent=2)
        fh.write("\n")
        fh.truncate()
        fcntl.flock(fh, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description="Submit batch NEB SLURM jobs")
    parser.add_argument("--n-jobs", type=int, default=5,
                        help="Number of jobs to submit (default: 5)")
    parser.add_argument("--config", default="assets/neb_defaults.yaml",
                        help="Path to config yaml (default: assets/neb_defaults.yaml)")
    parser.add_argument("--reaction-ids", default=None,
                        help="Comma-separated list of specific reaction IDs to submit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    if not QUEUE_FILE.exists():
        print("No queue.json. Initialize first:\n"
              "  python step0-batch/queue.py init --n-reactions N")
        sys.exit(1)

    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text())

    max_jobs = cfg.get("batch", {}).get("max_jobs", 20)
    n = min(args.n_jobs, max_jobs)

    specific_ids = None
    if args.reaction_ids:
        specific_ids = [int(x) for x in args.reaction_ids.split(",")]

    claimed = claim_pending(n, specific_ids)
    if not claimed:
        print("No pending reactions found.")
        sys.exit(0)

    print(f"{'[dry-run] ' if args.dry_run else ''}Submitting {len(claimed)} job(s)...\n")

    for reaction in claimed:
        cmd = build_sbatch_cmd(reaction, cfg, args.config, args.dry_run)
        print("  " + " ".join(cmd))

        if not args.dry_run:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}")
                # revert status back to pending
                with open(QUEUE_FILE, "r+") as fh:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                    data = json.load(fh)
                    for r in data["reactions"]:
                        if r["id"] == reaction["id"]:
                            r["status"] = "pending"
                            break
                    fh.seek(0)
                    json.dump(data, fh, indent=2)
                    fh.write("\n")
                    fh.truncate()
                    fcntl.flock(fh, fcntl.LOCK_UN)
            else:
                # "Submitted batch job 123456"
                job_id = result.stdout.strip().split()[-1]
                update_job_id(reaction["id"], job_id)
                print(f"  → reaction {reaction['id']:4d}  SLURM job {job_id}")
        print()

    if args.dry_run:
        print("[dry-run] No jobs submitted. Remove --dry-run to submit.")


if __name__ == "__main__":
    main()
