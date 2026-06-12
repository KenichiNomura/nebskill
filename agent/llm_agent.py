"""
NEB Agent — main orchestrator.
Asks clarifying questions, runs the full pipeline (steps 1-5),
drives adaptive retry, and generates an LLM interpretation of results.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from openai import OpenAI

PYTHON = sys.executable
EV_TO_KCAL = 23.0609

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Clarifying questions
# --------------------------------------------------------------------------- #

DEFAULTS_TABLE = """\
┌─────────────────────────────┬─────────────────────────────────────────┐
│ Parameter                   │ Default                                 │
├─────────────────────────────┼─────────────────────────────────────────┤
│ Mode                        │ single                                  │
│ Reaction index              │ next pending in queue                   │
│ MACE-OFF model size         │ medium                                  │
│ Number of NEB images        │ auto (max(9, round(path_length / 1.0))) │
│ Spring constant k           │ 0.1 eV/Å                               │
│ Final convergence fmax      │ 0.05 eV/Å                              │
│ Max retry attempts          │ 3                                       │
└─────────────────────────────┴─────────────────────────────────────────┘"""


def ask_clarifying_questions(cfg: dict) -> dict:
    """
    Interactive parameter confirmation.
    Returns a dict of user-confirmed overrides (may be empty = use all defaults).
    """
    print("\n" + "=" * 60)
    print("  NEB Calculation — Parameter Confirmation")
    print("=" * 60)
    print(DEFAULTS_TABLE)
    print()

    choice = input("Use all defaults? [Y/n]: ").strip().lower()
    if choice in ("", "y", "yes"):
        print("Using all defaults.\n")
        return {}

    overrides = {}

    # mode
    mode = input("Mode [single/batch] (default: single): ").strip().lower()
    if mode in ("batch",):
        overrides["mode"] = "batch"
        n_jobs = input("Number of batch jobs [1-20] (default: 5): ").strip()
        if n_jobs.isdigit():
            overrides["n_jobs"] = int(n_jobs)
    else:
        overrides["mode"] = "single"
        rxn = input("Reaction index (default: next pending): ").strip()
        if rxn.isdigit():
            overrides["reaction_id"] = int(rxn)

    # model size
    model = input("MACE-OFF model size [small/medium/large] (default: medium): ").strip().lower()
    if model in ("small", "large"):
        overrides["model_size"] = model

    # n_images
    n_img = input("Number of NEB images (default: auto): ").strip()
    if n_img.isdigit() and int(n_img) >= 7:
        overrides["n_images"] = int(n_img)

    # spring constant
    k = input("Spring constant k in eV/Å (default: 0.1): ").strip()
    try:
        overrides["spring_constant"] = float(k)
    except ValueError:
        pass

    # fmax
    fmax = input("Final fmax in eV/Å (default: 0.05): ").strip()
    try:
        overrides["phase2_fmax"] = float(fmax)
    except ValueError:
        pass

    # retries
    retries = input("Max retry attempts (default: 3): ").strip()
    if retries.isdigit():
        overrides["max_attempts"] = int(retries)

    print()
    return overrides


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Merge user overrides into config."""
    if "model_size" in overrides:
        cfg["calculator"]["model_size"] = overrides["model_size"]
    if "n_images" in overrides:
        cfg["neb"]["n_images"] = overrides["n_images"]
    if "spring_constant" in overrides:
        cfg["neb"]["spring_constant"] = overrides["spring_constant"]
    if "phase2_fmax" in overrides:
        cfg["neb"]["phase2_fmax"] = overrides["phase2_fmax"]
    if "max_attempts" in overrides:
        cfg["retry"]["max_attempts"] = overrides["max_attempts"]
    return cfg


# --------------------------------------------------------------------------- #
# Pipeline steps (subprocess calls)
# --------------------------------------------------------------------------- #

def run_step(cmd: list[str], label: str) -> int:
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def step1_load(reaction_id: int, config: str) -> int:
    return run_step(
        [PYTHON, "step1-load/load_dataset.py",
         "--reaction-id", str(reaction_id), "--config", config],
        "Step 1 — Loading reaction from Transition1x"
    )


def step2_relax(reaction_id: int, config: str) -> int:
    return run_step(
        [PYTHON, "step2-relax/relax_endpoints.py",
         "--reaction-id", str(reaction_id), "--config", config],
        "Step 2 — Relaxing endpoints with MACE-OFF"
    )


def step3_neb(reaction_id: int, config: str, extra_args: list[str] | None = None) -> int:
    cmd = [PYTHON, "step3-neb/neb_runner.py",
           "--reaction-id", str(reaction_id), "--config", config]
    if extra_args:
        cmd += extra_args
    return run_step(cmd, "Step 3 — Running NEB")


def step4_retry(reaction_id: int, config: str) -> int:
    return run_step(
        [PYTHON, "step4-monitor/retry.py",
         "--reaction-id", str(reaction_id), "--config", config],
        "Step 4 — Adaptive retry"
    )


def step5_analyze(reaction_id: int, config: str) -> int:
    out_dir = f"outputs/reaction_{reaction_id:04d}"
    rc  = run_step([PYTHON, "step5-analyze/analyze.py",
                    "--reaction-id", str(reaction_id), "--config", config],
                   "Step 5a — Computing barriers")
    rc |= run_step([PYTHON, "step5-analyze/plot.py",
                    "--reaction-id", str(reaction_id)],
                   "Step 5b — Generating energy profile plot")
    rc |= run_step([PYTHON, "step5-analyze/writer.py",
                    "--reaction-id", str(reaction_id)],
                   "Step 5c — Writing convergence log")
    return rc


# --------------------------------------------------------------------------- #
# LLM result interpretation
# --------------------------------------------------------------------------- #

INTERPRET_SYSTEM = """You are an expert computational chemist. You have just
completed a Nudged Elastic Band (NEB) calculation for an organic reaction.
Summarize the results clearly and concisely for a researcher. Cover:
1. Forward and reverse barriers (eV and kcal/mol)
2. Accuracy: how well MACE-OFF matches the DFT reference
3. Character of the transition state (image index, energy)
4. Any convergence issues and how they were resolved
5. Overall assessment: is this barrier reliable?
Be concise — no more than 200 words."""


def interpret_results(report: dict, retry_log: dict | None, cfg: dict) -> str:
    """Call LLM to generate a natural language interpretation of the NEB results."""
    try:
        from agent.auth import get_access_token
        token = get_access_token()
    except RuntimeError as e:
        return f"[LLM interpretation unavailable — {e}]"

    client = OpenAI(base_url=cfg["alcf"]["base_url"], api_key=token)

    summary_data = {
        "formula":             report["formula"],
        "forward_barrier_ev":  report["forward_barrier_ev"],
        "forward_barrier_kcal":report["forward_barrier_kcal"],
        "reverse_barrier_ev":  report["reverse_barrier_ev"],
        "reverse_barrier_kcal":report["reverse_barrier_kcal"],
        "ts_image_idx":        report["ts_image_idx"],
        "n_images":            report["n_images"],
        "dft_forward_barrier": report["dft_forward_barrier_ev"],
        "mace_vs_dft_error_ev":report["mace_vs_dft_error_ev"],
        "mace_vs_dft_pct":     report["mace_vs_dft_relative_pct"],
        "neb_method":          report["neb_method"],
        "mace_model":          report["mace_model_size"],
        "retry_attempts":      len(retry_log["attempts"]) if retry_log else 0,
    }

    response = client.chat.completions.create(
        model=cfg["alcf"]["model"],
        messages=[
            {"role": "system", "content": INTERPRET_SYSTEM},
            {"role": "user",
             "content": f"NEB results:\n```json\n{json.dumps(summary_data, indent=2)}\n```"},
        ],
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Queue helpers
# --------------------------------------------------------------------------- #

def next_pending_reaction(queue_file: Path) -> int:
    """Return the next pending reaction ID from queue.json, or 0 if no queue."""
    if not queue_file.exists():
        return 0
    data = json.loads(queue_file.read_text())
    for rxn in data.get("reactions", []):
        if rxn.get("status") == "pending":
            return rxn["id"]
    return 0


def update_queue_status(queue_file: Path, reaction_id: int,
                        status: str, **kwargs) -> None:
    if not queue_file.exists():
        return
    data = json.loads(queue_file.read_text())
    for rxn in data.get("reactions", []):
        if rxn["id"] == reaction_id:
            rxn["status"] = status
            rxn.update(kwargs)
            break
    queue_file.write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="NEB agentic pipeline")
    parser.add_argument("--reaction-id", type=int, default=None)
    parser.add_argument("--config", default="assets/neb_defaults.yaml")
    parser.add_argument("--defaults", action="store_true",
                        help="Skip clarifying questions and use all defaults")
    args = parser.parse_args()

    cfg = load_config(args.config)
    queue_file = ROOT / cfg["dataset"]["queue_file"]

    # --- clarifying questions ---
    if args.defaults:
        overrides = {}
    else:
        overrides = ask_clarifying_questions(cfg)

    cfg = apply_overrides(cfg, overrides)

    mode        = overrides.get("mode", "single")
    reaction_id = args.reaction_id \
                  or overrides.get("reaction_id") \
                  or next_pending_reaction(queue_file)

    print(f"\nRunning NEB for reaction {reaction_id} "
          f"(MACE-OFF {cfg['calculator']['model_size']})\n")

    update_queue_status(queue_file, reaction_id, "running")
    out_dir = ROOT / f"outputs/reaction_{reaction_id:04d}"

    # --- Step 1: load ---
    rc = step1_load(reaction_id, args.config)
    if rc == 2:
        print("Reaction skipped (low barrier or edge case).")
        update_queue_status(queue_file, reaction_id, "skipped")
        sys.exit(0)
    if rc != 0:
        update_queue_status(queue_file, reaction_id, "failed",
                            reason="load_failed")
        sys.exit(rc)

    # --- Step 2: relax ---
    rc = step2_relax(reaction_id, args.config)
    if rc != 0:
        update_queue_status(queue_file, reaction_id, "failed",
                            reason="endpoint_relaxation_failed")
        sys.exit(rc)

    # --- Step 3: NEB (with inline retry via step 4) ---
    rc = step3_neb(reaction_id, args.config)
    if rc == 4:
        print("\nNEB did not converge — starting adaptive retry loop.")
        rc = step4_retry(reaction_id, args.config)

    if rc != 0:
        update_queue_status(queue_file, reaction_id, "failed",
                            reason="neb_not_converged")
        fail_path = out_dir / "failure_report.json"
        if fail_path.exists():
            print(f"\nFailure report: {fail_path}")
        sys.exit(rc)

    # --- Step 5: analyze ---
    rc = step5_analyze(reaction_id, args.config)
    if rc != 0:
        print("Analysis step failed.", file=sys.stderr)
        sys.exit(rc)

    # --- LLM interpretation ---
    print(f"\n{'=' * 60}")
    print("  LLM Interpretation")
    print(f"{'=' * 60}")
    report_path = out_dir / "report.json"
    report      = json.loads(report_path.read_text())
    retry_path  = out_dir / "retry_log.json"
    retry_log   = json.loads(retry_path.read_text()) if retry_path.exists() else None

    interpretation = interpret_results(report, retry_log, cfg)
    print(interpretation)

    # save interpretation alongside report
    (out_dir / "interpretation.txt").write_text(interpretation)

    update_queue_status(queue_file, reaction_id, "done",
                        barrier_eV=report["forward_barrier_ev"])

    print(f"\n{'=' * 60}")
    print(f"  Done — outputs in outputs/reaction_{reaction_id:04d}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
