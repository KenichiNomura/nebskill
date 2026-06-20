# ALCF Inference Endpoint Reference

## Overview

The ALCF inference endpoint provides OpenAI-compatible LLM inference on
the Sophia cluster at Argonne Leadership Computing Facility, powered by
vLLM and the FIRST (Federated Inference Resource Scheduling Toolkit).

## Endpoint

```
Base URL: https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1
Model:    meta-llama/Meta-Llama-3.1-70B-Instruct
```

The model is set in `assets/neb_defaults.yaml`'s `alcf.model` key and read
from there by `step4-monitor/retry.py:call_llm_for_intervention` — change it
there (not in this doc) if you want to switch models.

## Authentication

Uses Globus Auth, implemented in `agent/auth.py` (not a separate
`inference_auth_token.py` — that module doesn't exist in this skill).

```python
from agent.auth import get_access_token
token = get_access_token()   # returns a valid Globus bearer token, auto-refreshing
```

Token is cached at `~/.globus/nebskill/tokens.json`.

First-time setup or after expiry, run:
```bash
python agent/auth.py login    # interactive Globus OAuth flow, caches the token
python agent/auth.py check    # verify a cached token is still valid
```

### Globus client IDs

- Auth client ID: `58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944`
- Gateway client ID: `681c10cc-f684-4540-bcd7-0b4df3bc26ef`

## OpenAI client setup

```python
from openai import OpenAI
from agent.auth import get_access_token

client = OpenAI(
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    api_key=get_access_token()
)
```

## Chat completion with function calling

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=messages,
    tools=tools,           # list of tool definitions (JSON schema)
    tool_choice="required" # force the model to call a tool
)

# Extract tool call
tool_call = response.choices[0].message.tool_calls[0]
fn_name = tool_call.function.name
fn_args = json.loads(tool_call.function.arguments)

# Token usage (see step4-monitor/retry.py — logged into retry_log.json)
usage = response.usage
print(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
```

## Tool definition format (OpenAI schema)

Example — one of the four tools actually defined in
`step4-monitor/retry.py:TOOLS`:

```python
tools = [
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
]
```

The full set is `set_n_images`, `adjust_spring_constant`, `switch_optimizer`
(`FIRE2`/`MDMin` only — see `references/neb_method.md` for why BFGS/LBFGS
are excluded), and `tighten_endpoint_relaxation`.

## Available models

To list all models available on the endpoint at runtime:

```python
models = client.models.list()
for m in models.data:
    print(m.id)
```

## Notes

- The endpoint is OpenAI API-compatible: `openai` Python package works directly
- `meta-llama/Meta-Llama-3.1-70B-Instruct` supports native function/tool calling
- Rate limits and quotas depend on your ALCF allocation
- For LangChain integration: use `langchain_openai.ChatOpenAI` with the same
  `base_url` and `api_key` parameters
