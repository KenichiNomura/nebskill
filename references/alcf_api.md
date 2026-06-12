# ALCF Inference Endpoint Reference

## Overview

The ALCF inference endpoint provides OpenAI-compatible LLM inference on
the Sophia cluster at Argonne Leadership Computing Facility, powered by
vLLM and the FIRST (Federated Inference Resource Scheduling Toolkit).

## Endpoint

```
Base URL: https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1
Model:    Qwen/Qwen3-32B
```

## Authentication

Uses Globus Auth. The access token is obtained via `inference_auth_token.py`
from the ALCF inference-endpoints repository and passed as the API key.

```python
from agent.inference_auth_token import get_access_token
token = get_access_token()   # returns a valid Globus bearer token
```

Token is cached at `~/.globus/app/{client_id}/{app_name}/tokens.json`.
On expiry, direct the user to run: `python agent/inference_auth_token.py auth`

### Globus client IDs

- Auth client ID: `58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944`
- Gateway client ID: `681c10cc-f684-4540-bcd7-0b4df3bc26ef`

## OpenAI client setup

```python
from openai import OpenAI
from agent.inference_auth_token import get_access_token

client = OpenAI(
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    api_key=get_access_token()
)
```

## Chat completion with function calling

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=messages,
    tools=tools,           # list of tool definitions (JSON schema)
    tool_choice="required" # force the model to call a tool
)

# Extract tool call
tool_call = response.choices[0].message.tool_calls[0]
fn_name = tool_call.function.name
fn_args = json.loads(tool_call.function.arguments)
```

## Tool definition format (OpenAI schema)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "set_n_images",
            "description": "Increase the number of NEB images to fix bunching or large inter-image gaps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "New number of images (>= 7)"},
                    "reasoning": {"type": "string", "description": "Why this intervention was chosen"}
                },
                "required": ["n", "reasoning"]
            }
        }
    }
]
```

## Available models

`Qwen/Qwen3-32B` is confirmed available on Sophia. To list all available
models at runtime:

```python
models = client.models.list()
for m in models.data:
    print(m.id)
```

## Notes

- The endpoint is OpenAI API-compatible: `openai` Python package works directly
- `Qwen3-32B` supports native function/tool calling
- Rate limits and quotas depend on your ALCF allocation
- For LangChain integration: use `langchain_openai.ChatOpenAI` with the same
  `base_url` and `api_key` parameters
