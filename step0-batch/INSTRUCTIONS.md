# Step 0 — Batch Mode

Manages a queue of up to 20 independent NEB jobs submitted as SLURM jobs.
Each job runs the full single-job pipeline (steps 1–5) for one reaction.

## Queue file: queue.json

Located at the project root. Tracks status of all reactions:

```json
{
  "reactions": [
    {"id": 0, "status": "done",    "slurm_job": "123456", "barrier_eV": 1.24},
    {"id": 1, "status": "running", "slurm_job": "123457"},
    {"id": 2, "status": "failed",  "slurm_job": "123458", "reason": "retry_exhausted"},
    {"id": 3, "status": "pending"}
  ]
}
```

Valid statuses: `pending`, `running`, `done`, `failed`.

## Scripts

### submit.py — launch N jobs from the queue

```bash
python step0-batch/submit.py --n-jobs 5 [--dry-run]
```

- Claims the next N `pending` reactions from queue.json (file-locked)
- Submits one SLURM job per reaction via `sbatch`
- Updates status to `running` with the assigned SLURM job ID
- `--dry-run` prints the sbatch commands without submitting

### queue.py — queue management utilities

```bash
python step0-batch/queue.py init --n-reactions 20   # populate queue with IDs 0..19
python step0-batch/queue.py status                   # print summary table
python step0-batch/queue.py requeue --status failed  # reset failed jobs to pending
```

### aggregate.py — collect results after all jobs complete

```bash
python step0-batch/aggregate.py
```

Reads all `outputs/reaction_*/report.json` files and produces:
- `outputs/summary.json` — barrier heights, MACE-OFF vs DFT MAE/RMSE, failure stats
- `outputs/summary.png` — barrier height distribution + parity plot vs DFT reference

## SLURM job template: job_template.sh

Each submitted job runs:
```bash
source .venv/bin/activate
python agent/llm_agent.py --reaction-id $REACTION_ID --config assets/neb_defaults.yaml
```

SLURM settings come from `assets/neb_defaults.yaml` (batch section).

## Workflow

1. Initialize queue: `python step0-batch/queue.py init --n-reactions N`
2. Submit jobs: `python step0-batch/submit.py --n-jobs N`
3. Monitor: `python step0-batch/queue.py status` (run periodically)
4. Requeue any failures: `python step0-batch/queue.py requeue --status failed`
5. Aggregate: `python step0-batch/aggregate.py`

## Notes

- Jobs are independent — no inter-job communication
- File locking in queue.py prevents race conditions when jobs mark themselves done
- Output directories are isolated: `outputs/reaction_{id:04d}/`
- Failed jobs can be requeued and resubmitted without affecting completed ones
