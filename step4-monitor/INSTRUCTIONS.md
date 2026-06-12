# Step 4 — Monitor and Retry

Diagnoses NEB convergence failures and uses the LLM agent to choose an
intervention. Up to 3 retry attempts are made before issuing a structured
failure report.

## Scripts

```bash
python step4-monitor/diagnostics.py --reaction-id INT   # compute diagnostic payload
python step4-monitor/retry.py --reaction-id INT --config assets/neb_defaults.yaml
```

`retry.py` orchestrates the full retry loop: calls diagnostics, sends payload
to LLM agent, applies the chosen intervention, re-runs step 3.

## Diagnostic payload (diagnostics.py)

Computed from the last `neb_result.json`:

| Metric | Description | Failure signal |
|---|---|---|
| `per_image_fmax` | max force per image (eV/Å) | high forces concentrated at specific images |
| `inter_image_rmsd` | RMSD between consecutive images (Å) | < 0.05 Å = collapse; highly uneven = bunching |
| `energy_smoothness` | second derivative of energy profile | large values = kinking or discontinuity |
| `steps_taken` | steps used vs cap | near cap = almost converged vs stuck |
| `phase` | which phase failed (1 or 2) | phase 2 failures need different fixes |

Written to `outputs/reaction_{id:04d}/diagnostics.json`.

## LLM intervention selection

The diagnostic payload is sent to the LLM agent via OpenAI function calling.
The agent selects exactly one intervention per retry using one of these tools:

```
set_n_images(n: int)
  → use when inter_image_rmsd shows bunching or images are too far apart

adjust_spring_constant(k: float)
  → use when inter_image_rmsd shows collapse (increase k) or over-tension

switch_method(method: "string")
  → use when energy_smoothness shows kinking or energy discontinuities

tighten_endpoint_relaxation(fmax: float)
  → use when per_image_fmax is high at endpoint images (0 or n-1),
    suggesting endpoints are not true minima; re-runs step 2 with tighter fmax
```

The agent must also provide a brief `reasoning` string explaining the choice.
This is logged to `outputs/reaction_{id:04d}/retry_log.json`.

## Retry loop (retry.py)

```
attempt = 0
while attempt < max_attempts:
    run step 3 (neb_runner.py)
    if converged: break
    compute diagnostics
    send to LLM agent → get intervention
    apply intervention to config
    attempt += 1

if not converged after max_attempts:
    write failure report (see below)
    mark queue.json as failed
```

## Structured failure report

Written to `outputs/reaction_{id:04d}/failure_report.json`:

```json
{
  "reaction_id": 42,
  "attempts": 3,
  "final_fmax": 0.38,
  "final_phase": 2,
  "interventions": [
    {"attempt": 1, "tool": "switch_method",    "value": "string",  "reasoning": "..."},
    {"attempt": 2, "tool": "set_n_images",     "value": 13,        "reasoning": "..."},
    {"attempt": 3, "tool": "adjust_spring_constant", "value": 0.2, "reasoning": "..."}
  ],
  "last_diagnostics": { "..." },
  "last_neb_result":  { "..." }
}
```

## Notes

- Endpoint relaxation failures (step 2) are NOT counted as retry attempts
- The LLM agent is called via `agent/llm_agent.py` — Globus token must be valid
- Each retry starts from the last converged NEB geometry, not from scratch,
  unless `tighten_endpoint_relaxation` is selected (which re-runs from step 2)
