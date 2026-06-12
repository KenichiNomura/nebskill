"""Queue management for batch NEB jobs.

Usage:
  python step0-batch/queue.py init --n-reactions 20
  python step0-batch/queue.py status
  python step0-batch/queue.py requeue --status failed
  python step0-batch/queue.py sync        # refresh running→done/failed via sacct
"""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_FILE = ROOT / "queue.json"


# --------------------------------------------------------------------------- #
# File-locked queue I/O
# --------------------------------------------------------------------------- #

class LockedQueue:
    """Context manager that holds an exclusive file lock while the queue is open."""

    def __init__(self, path: Path = QUEUE_FILE):
        self.path = path
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "r+")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        self.data = json.load(self._fh)
        return self

    def __exit__(self, *_):
        self._fh.seek(0)
        json.dump(self.data, self._fh, indent=2)
        self._fh.write("\n")
        self._fh.truncate()
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def _init_queue_file(reactions: list[dict]) -> None:
    """Write a fresh queue.json (no locking — called only at init time)."""
    QUEUE_FILE.write_text(json.dumps({"reactions": reactions}, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_init(args):
    """Populate queue.json with n_reactions pending entries."""
    n = args.n_reactions
    start = args.start_id

    if QUEUE_FILE.exists() and not args.force:
        print(f"queue.json already exists. Use --force to overwrite.")
        sys.exit(1)

    reactions = [{"id": start + i, "status": "pending"} for i in range(n)]
    _init_queue_file(reactions)
    print(f"Initialized queue with {n} reactions (IDs {start}–{start + n - 1})")
    print(f"  {QUEUE_FILE}")


def cmd_status(args):
    """Print a summary table of the queue."""
    if not QUEUE_FILE.exists():
        print("No queue.json found. Run: python step0-batch/queue.py init --n-reactions N")
        sys.exit(1)

    data = json.loads(QUEUE_FILE.read_text())
    reactions = data.get("reactions", [])

    from collections import Counter
    counts = Counter(r["status"] for r in reactions)
    total = len(reactions)

    print(f"\nQueue summary  ({total} total)")
    print(f"  {'pending':10s}  {counts.get('pending', 0)}")
    print(f"  {'running':10s}  {counts.get('running', 0)}")
    print(f"  {'done':10s}  {counts.get('done', 0)}")
    print(f"  {'failed':10s}  {counts.get('failed', 0)}")
    print(f"  {'skipped':10s}  {counts.get('skipped', 0)}")
    print()

    if args.verbose:
        width = max(len(str(r["id"])) for r in reactions)
        print(f"  {'ID':>{width}}  {'status':10s}  {'job':12s}  {'barrier_eV':>12}  note")
        print("  " + "-" * 60)
        for r in reactions:
            job = r.get("slurm_job", "—")
            barr = f"{r['barrier_eV']:.4f}" if "barrier_eV" in r else "—"
            note = r.get("reason", "")
            print(f"  {r['id']:>{width}}  {r['status']:10s}  {job:12s}  {barr:>12}  {note}")
        print()


def cmd_requeue(args):
    """Reset reactions matching --status back to pending."""
    if not QUEUE_FILE.exists():
        print("No queue.json found.")
        sys.exit(1)

    target_statuses = set(args.status.split(","))
    with LockedQueue() as q:
        changed = 0
        for r in q.data.get("reactions", []):
            if r["status"] in target_statuses:
                r["status"] = "pending"
                r.pop("slurm_job", None)
                r.pop("reason", None)
                r.pop("barrier_eV", None)
                changed += 1
    print(f"Reset {changed} reaction(s) to pending.")


def cmd_sync(args):
    """
    Query sacct for all running jobs and update their status.
    Marks jobs that have completed as done or failed based on exit code.
    """
    if not QUEUE_FILE.exists():
        print("No queue.json found.")
        sys.exit(1)

    data = json.loads(QUEUE_FILE.read_text())
    running = [r for r in data["reactions"] if r["status"] == "running"
               and "slurm_job" in r]
    if not running:
        print("No running jobs to sync.")
        return

    job_ids = [r["slurm_job"] for r in running]
    result = subprocess.run(
        ["sacct", "-j", ",".join(job_ids),
         "--format=JobID,State,ExitCode", "--noheader", "--parsable2"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"sacct failed: {result.stderr.strip()}")
        sys.exit(1)

    # build a map: job_id -> (state, exit_code)
    job_info: dict[str, tuple[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        jid, state, exitcode = parts[0], parts[1], parts[2]
        # skip sub-steps (e.g. "12345.batch")
        if "." in jid:
            continue
        job_info[jid] = (state, exitcode)

    with LockedQueue() as q:
        changed = 0
        for r in q.data["reactions"]:
            jid = r.get("slurm_job")
            if r["status"] != "running" or jid not in job_info:
                continue
            state, exitcode = job_info[jid]
            if state in ("COMPLETED",):
                # check whether the output report exists
                out = ROOT / f"outputs/reaction_{r['id']:04d}/report.json"
                if out.exists():
                    import json as _json
                    rep = _json.loads(out.read_text())
                    r["status"] = "done"
                    r["barrier_eV"] = rep.get("forward_barrier_ev")
                else:
                    r["status"] = "failed"
                    r["reason"] = "completed_no_report"
                changed += 1
            elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                           "OUT_OF_MEMORY"):
                r["status"] = "failed"
                r["reason"] = state.lower()
                changed += 1
    print(f"Synced {changed} job(s) from sacct.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="NEB queue management")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize queue.json")
    p_init.add_argument("--n-reactions", type=int, required=True)
    p_init.add_argument("--start-id",   type=int, default=0)
    p_init.add_argument("--force",      action="store_true")

    p_status = sub.add_parser("status", help="Print queue summary")
    p_status.add_argument("-v", "--verbose", action="store_true")

    p_req = sub.add_parser("requeue", help="Reset reactions to pending")
    p_req.add_argument("--status", default="failed",
                       help="Comma-separated list of statuses to reset (default: failed)")

    sub.add_parser("sync", help="Sync running jobs via sacct")

    args = parser.parse_args()
    {"init": cmd_init, "status": cmd_status,
     "requeue": cmd_requeue, "sync": cmd_sync}[args.cmd](args)


if __name__ == "__main__":
    main()
