"""
Adaptive NEB retry loop.
Calls the LLM agent to diagnose failures and choose interventions,
then re-runs step3-neb up to max_attempts times.
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

from agent.auth import get_access_token
from lib.neb_diagnostics import diagnose


# --------------------------------------------------------------------------- #
# Tool definitions for LLM function calling
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_n_images",
            "description": (
                "Increase the number of NEB images. Use when inter-image RMSD is "
                "high or uneven (bunching), or when images are too far apart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "New number of images (must be >= current + 2)"
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["n", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_spring_constant",
            "description": (
                "Change the NEB spring constant k (eV/Å). Increase to fix image "
                "collapse (images too close). Decrease if springs dominate forces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "k": {
                        "type": "number",
                        "description": "New spring constant in eV/Å (typical range 0.05–1.0)"
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["k", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_method",
            "description": (
                "Switch the NEB band method. Use 'string' when kinking or energy "
                "discontinuities are detected (high second-derivative score)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["string", "improvedtangent", "spline"],
                        "description": "NEB method to use"
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["method", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tighten_endpoint_relaxation",
            "description": (
                "Re-relax endpoints with a tighter fmax. Use when endpoint images "
                "(index 0 or last internal image) have the highest forces, "
                "suggesting the endpoints are not at true minima."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fmax": {
                        "type": "number",
                        "description": "Tighter fmax for endpoint relaxation (e.g. 0.005)"
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["fmax", "reasoning"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# LLM agent call
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an expert computational chemist specializing in NEB
(Nudged Elastic Band) calculations. You will receive diagnostic information
about a failed NEB convergence and must choose exactly one intervention to
fix the problem. Base your decision on the failure_mode, energy smoothness
(kinking), inter-image spacing (collapse/bunching), and force distribution.
Be concise and precise in your reasoning."""


def call_llm_for_intervention(diagnostics: dict, cfg: dict) -> tuple[str, dict]:
    """
    Ask the LLM agent to choose one intervention.
    Returns (tool_name, tool_args).
    """
    token = get_access_token()
    client = OpenAI(
        base_url=cfg["alcf"]["base_url"],
        api_key=token,
    )

    user_message = (
        f"NEB convergence failure diagnostics:\n"
        f"```json\n{json.dumps(diagnostics, indent=2)}\n```\n\n"
        f"Current parameters: n_images={diagnostics['n_images']}, "
        f"method={diagnostics['method']}, "
        f"spring_constant={diagnostics['spring_constant']} eV/Å.\n\n"
        f"Choose exactly one intervention tool to fix this failure."
    )

    response = client.chat.completions.create(
        model=cfg["alcf"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        tools=TOOLS,
        tool_choice="required",
    )

    tool_call = response.choices[0].message.tool_calls[0]
    fn_name = tool_call.function.name
    fn_args = json.loads(tool_call.function.arguments)
    return fn_name, fn_args


# --------------------------------------------------------------------------- #
# Intervention application
# --------------------------------------------------------------------------- #

def apply_intervention(fn_name: str, fn_args: dict,
                       neb_args: dict, relax_args: dict) -> tuple[dict, dict, bool]:
    """
    Update neb_args / relax_args based on the chosen intervention.
    Returns (neb_args, relax_args, needs_rerelax).
    """
    needs_rerelax = False

    if fn_name == "set_n_images":
        neb_args["n_images"] = fn_args["n"]

    elif fn_name == "adjust_spring_constant":
        neb_args["spring_constant"] = fn_args["k"]

    elif fn_name == "switch_method":
        neb_args["method"] = fn_args["method"]

    elif fn_name == "tighten_endpoint_relaxation":
        relax_args["fmax"] = fn_args["fmax"]
        needs_rerelax = True

    return neb_args, relax_args, needs_rerelax


# --------------------------------------------------------------------------- #
# Subprocess helpers
# --------------------------------------------------------------------------- #

def run_relax(out_dir: Path, config: str, mlip: str, registry: str) -> int:
    cmd = [sys.executable, "step2-relax/relax_endpoints.py",
           "--output-dir", str(out_dir),
           "--config", config,
           "--mlip", mlip,
           "--registry", registry]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_neb(out_dir: Path, config: str, mlip: str, registry: str,
            neb_args: dict) -> int:
    cmd = [sys.executable, "step3-neb/neb_runner.py",
           "--output-dir", str(out_dir),
           "--config", config,
           "--mlip", mlip,
           "--registry", registry]
    if "n_images" in neb_args:
        cmd += ["--n-images", str(neb_args["n_images"])]
    if "method" in neb_args:
        cmd += ["--method", neb_args["method"]]
    if "spring_constant" in neb_args:
        cmd += ["--spring-constant", str(neb_args["spring_constant"])]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


# --------------------------------------------------------------------------- #
# Main retry loop
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Adaptive NEB retry loop")
    parser.add_argument("--reaction-id", type=int, default=None)
    parser.add_argument("--config",      default="assets/neb_defaults.yaml")
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--mlip",        required=True,
                        help="MLIP name from registry")
    parser.add_argument("--registry",    default="assets/mlip_registry.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_attempts = int(cfg["retry"]["max_attempts"])
    out_dir = Path(args.output_dir)

    neb_args   = {}
    relax_args = {}
    retry_log  = []

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Retry attempt {attempt}/{max_attempts} ---")

        neb_result_path = out_dir / "neb_result.json"
        if not neb_result_path.exists():
            print("No neb_result.json found — cannot diagnose", file=sys.stderr)
            sys.exit(1)

        neb_result   = json.loads(neb_result_path.read_text())
        diagnostics  = diagnose(neb_result)
        diag_path    = out_dir / "diagnostics.json"
        diag_path.write_text(json.dumps(diagnostics, indent=2))

        print(f"  Failure mode: {diagnostics['failure_mode']}")
        print(f"  Calling LLM agent for intervention...")

        try:
            fn_name, fn_args = call_llm_for_intervention(diagnostics, cfg)
        except Exception as e:
            print(f"  LLM call failed: {e}", file=sys.stderr)
            fn_name, fn_args = _fallback_intervention(diagnostics, attempt)
            print(f"  Using fallback intervention: {fn_name}")

        print(f"  Intervention: {fn_name}({fn_args})")
        print(f"  Reasoning: {fn_args.get('reasoning', '')}")

        retry_log.append({
            "attempt":   attempt,
            "tool":      fn_name,
            "args":      fn_args,
            "reasoning": fn_args.get("reasoning", ""),
            "diagnostics_snapshot": {
                "failure_mode":  diagnostics["failure_mode"],
                "fmax_final":    diagnostics["fmax_final"],
                "n_images":      diagnostics["n_images"],
                "method":        diagnostics["method"],
            },
        })

        neb_args, relax_args, needs_rerelax = apply_intervention(
            fn_name, fn_args, neb_args, relax_args
        )

        if needs_rerelax:
            print("  Re-relaxing endpoints with tighter fmax...")
            rc = run_relax(out_dir, args.config, args.mlip, args.registry)
            if rc != 0:
                print("  Endpoint re-relaxation failed — aborting retry", file=sys.stderr)
                break

        rc = run_neb(out_dir, args.config, args.mlip, args.registry, neb_args)
        if rc == 0:
            print(f"\nNEB converged after {attempt} retry attempt(s).")
            _write_retry_log(out_dir, retry_log, success=True)
            sys.exit(0)

        print(f"  NEB still not converged (exit code {rc})")

    # all retries exhausted
    print(f"\nAll {max_attempts} retry attempts exhausted.")
    _write_failure_report(out_dir, args.mlip, retry_log,
                          json.loads((out_dir / "neb_result.json").read_text()),
                          json.loads((out_dir / "diagnostics.json").read_text()))
    _write_retry_log(out_dir, retry_log, success=False)
    sys.exit(5)


def _fallback_intervention(diagnostics: dict, attempt: int) -> tuple[str, dict]:
    """Fixed escalation used when LLM call fails."""
    mode = diagnostics["failure_mode"]
    if mode == "image_collapse" or attempt == 2:
        return "adjust_spring_constant", {
            "k": diagnostics["spring_constant"] * 2,
            "reasoning": "fallback: doubling spring constant for image collapse",
        }
    elif mode == "kinking" or attempt == 3:
        return "switch_method", {
            "method": "string",
            "reasoning": "fallback: switching to string method for kinking",
        }
    else:
        n = diagnostics["n_images"]
        return "set_n_images", {
            "n": n + 4,
            "reasoning": "fallback: adding 4 images",
        }


def _write_retry_log(out_dir: Path, log: list, success: bool) -> None:
    path = out_dir / "retry_log.json"
    path.write_text(json.dumps({"success": success, "attempts": log}, indent=2))


def _write_failure_report(out_dir: Path, mlip: str,
                          retry_log: list, last_neb: dict, last_diag: dict) -> None:
    report = {
        "mlip":             mlip,
        "status":           "failed",
        "reason":           "retry_exhausted",
        "n_attempts":       len(retry_log),
        "interventions":    retry_log,
        "last_diagnostics": last_diag,
        "last_neb_result":  last_neb["latest"],
    }
    path = out_dir / "failure_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Failure report written to {path}")


if __name__ == "__main__":
    main()
