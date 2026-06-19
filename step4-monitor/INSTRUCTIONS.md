# Step 4 — Monitor and Retry

Diagnoses NEB convergence failures and uses the LLM agent to choose an
intervention. Up to `retry.max_attempts` retries are made before issuing a
structured failure report.

## Scripts

```bash
python step4-monitor/diagnostics.py --output-dir outputs/{run_id}/{mlip}
python step4-monitor/retry.py --output-dir outputs/{run_id}/{mlip} \
    --mlip <mlip_name> --config assets/neb_defaults.yaml \
    --registry assets/mlip_registry.yaml
```

`retry.py` orchestrates the full retry loop: computes diagnostics, sends
the payload to the ALCF LLM agent for an intervention, applies it, and
re-runs step 3 (and step 2 if the intervention requires tighter endpoint
relaxation).

## Diagnostic payload (diagnostics.py / lib/neb_diagnostics.py)

Computed from the last `neb_result.json`:

| Metric | Description | Failure signal |
|---|---|---|
| `forces_per_image` | max force per image (eV/Å) | high forces concentrated at specific images |
| `inter_image_rmsd` | RMSD between consecutive images (Å) | low = collapse; highly uneven = bunching |
| `energy_smoothness` | second derivative of energy profile | large values = kinking or discontinuity |
| `steps_taken` | steps used vs cap | near cap = almost converged vs stuck |
| `phase` | which phase failed (1 or 2) | phase 2 failures need different fixes |

Written to `diagnostics.json` in the same `--output-dir`.

## LLM intervention selection

The diagnostic payload is sent to the LLM agent via OpenAI function calling
(`step4-monitor/retry.py:call_llm_for_intervention`). The agent selects
exactly one intervention per retry attempt:

```
set_n_images(n: int)
  → use when inter_image_rmsd shows bunching or images are too far apart

adjust_spring_constant(k: float)
  → use when inter_image_rmsd shows collapse (increase k) or over-tension

switch_method(method: "string" | "improvedtangent" | "spline")
  → use when energy_smoothness shows kinking or energy discontinuities

tighten_endpoint_relaxation(fmax: float)
  → use when forces_per_image is high at endpoint images, suggesting
    endpoints are not true minima; re-runs step 2 with tighter fmax
```

The agent must also provide a brief `reasoning` string. If the LLM call
itself fails (token expired, endpoint down), a fixed escalation fallback
(`_fallback_intervention`) is used instead so the retry loop still makes
progress.

This is logged to `retry_log.json` in `--output-dir`.

## Retry loop (retry.py)

```
for attempt in 1..max_attempts:
    read neb_result.json, compute diagnostics
    call LLM (or fallback) → get intervention
    apply intervention; re-relax endpoints if needed
    re-run step 3 (neb_runner.py)
    if converged: write retry_log.json (success=true), exit 0

if not converged after max_attempts:
    write failure_report.json
    write retry_log.json (success=false)
    exit 5
```

## Structured failure report

Written to `failure_report.json` in `--output-dir`:

```json
{
  "mlip": "nequip-oam-l",
  "status": "failed",
  "reason": "retry_exhausted",
  "n_attempts": 3,
  "interventions": [
    {"attempt": 1, "tool": "switch_method", "args": {"method": "string", "reasoning": "..."}},
    {"attempt": 2, "tool": "set_n_images", "args": {"n": 13, "reasoning": "..."}},
    {"attempt": 3, "tool": "adjust_spring_constant", "args": {"k": 0.2, "reasoning": "..."}}
  ],
  "last_diagnostics": { "..." },
  "last_neb_result":  { "..." }
}
```

## Notes

- Endpoint relaxation failures (step 2) are NOT counted as retry attempts
- The LLM call uses `agent/auth.py:get_access_token()` — the Globus token
  must be valid (`python agent/auth.py login`)
- Retries reuse the last NEB geometry by re-running step 3 with updated
  parameters, except `tighten_endpoint_relaxation`, which re-runs from
  step 2
