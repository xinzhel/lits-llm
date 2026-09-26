# Private llama.cpp Server over Tailscale

Use the prompt below with a coding agent running on an Apple Silicon host. It sets up a
reusable OpenAI-compatible endpoint, verifies LiTS integration, and exposes the endpoint only
within a Tailscale network.

```text
Set up and verify a private, reusable, OpenAI-compatible LLM server on this Mac.

## Objective

Run a capable open-weight model through llama.cpp with Metal acceleration. Expose its
OpenAI-compatible API locally and privately over Tailscale. Do not run a full agent or search
experiment.

## 1. Inspect the Host

Report the Apple Silicon chip, unified memory, macOS version, available storage, existing
llama.cpp installation, Tailscale status, and `uv tool list`. Do not overwrite customized
installations or unrelated Git changes.

## 2. Select a Current Model

Search authoritative sources and compare two or three recent open-weight instruction models.
Choose a model that:

- is practical for the detected hardware, normally around 30--40B parameters or a comparable
  mixture-of-experts model;
- performs strongly in reasoning, coding, tool use, instruction following, and structured JSON;
- supports at least a 32K context;
- has a suitable license, verified GGUF release, and llama.cpp Jinja chat template.

Explain the choice before downloading. Start with Q4_K_M unless another quantization has a
clear quality, memory, or performance advantage.

## 3. Start llama-server

Install or update llama.cpp with Metal support, then run a localhost-only server. Adapt flags
only when the installed version requires it and explain material changes.

llama-server \
  -m /absolute/path/to/model.gguf \
  --alias local-instruct \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 32768 \
  --jinja

Record the exact final command.

## 4. Verify localhost

Using both `curl` and an OpenAI Python client, verify:

- GET `http://127.0.0.1:8080/v1/models`;
- POST `http://127.0.0.1:8080/v1/chat/completions`;
- system and user messages, deterministic decoding, and JSON-only output;
- standard prompt, completion, and total token usage fields;
- `chat_template_kwargs={"enable_thinking": false}`;
- model alias `local-instruct`.

Do not configure Tailscale until localhost passes.

## 5. Verify the Editable LiTS Installation

If `lits-llm` appears in `uv tool list`, identify its Python environment and confirm that
`import lits` resolves to the expected editable Git checkout. Report its path, branch, commit,
and `git status`.

Through that environment, run one minimal call:

from lits import get_lm
from lits.lm import setup_inference_logging

model = get_lm(
    "openai/local-instruct",
    base_url="http://127.0.0.1:8080/v1",
    api_key="no-key",
)
setup_inference_logging(model, root_dir="/tmp/lits-local-model-check")
output = model(
    [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": "Select one ID from node-a and node-b."},
    ],
    role="evaluator_selector",
    temperature=1e-6,
    max_new_tokens=64,
    enable_thinking=False,
)

Verify the response, JSON, model identifier, token counts, running time, and logged role. Reuse
the same model for a second call under a task-policy role and confirm that the roles remain
distinguishable in one inference log.

## 6. Diagnose Before Editing

Classify failures as model/GGUF, chat template, llama-server, OpenAI API, Tailscale, or LiTS.
Use direct `curl` results to isolate server failures. Do not modify LiTS to compensate for a
server or network configuration error.

If a LiTS defect is confirmed:

1. preserve unrelated working-tree changes;
2. make the smallest fix and preserve documentation;
3. add only a minimal sequential check under `unit_test/` without mocks, test frameworks, or
   assertions;
4. update the existing `CHANGELOG.md` before committing;
5. verify, commit, and push the fix;
6. report the commit and explicitly remind the user to update LiTS on every client machine.

## 7. Expose It through Tailscale

After localhost passes, expose only the local port to the tailnet:

tailscale serve --bg 8080

Do not use Tailscale Funnel or bind llama-server to `0.0.0.0`. Record the MagicDNS HTTPS URL,
then repeat `/v1/models` and `/v1/chat/completions` from another tailnet device. Run
`tailscale ping <host>` and report whether the path is direct or DERP-relayed.

The remote client configuration is:

model = get_lm(
    "openai/local-instruct",
    base_url="https://<magicdns-host>/v1",
    api_key="no-key",
)

## 8. Report

Report the model candidates, selected model and license, quantization, file size, llama.cpp
version, context length, exact server command, localhost and LiTS checks, inference-log fields,
MagicDNS URL, working remote curl request, Tailscale path, prompt-processing speed, generation
speed, limitations, and any LiTS fix commit that client machines must pull.
```

## Remote Client Examples

The verified tailnet-only endpoint is:

```text
https://xinzhes-macbook-pro-2.tailde6fe4.ts.net
```

The current deployment uses `ggml-org/Qwen3.8-27B-GGUF` with the
`Qwen3.8-27B-Q4_K_M.gguf` quantization. It is served with a 32K context window while keeping
the stable `local-instruct` alias used by LiTS clients:

```bash
/opt/homebrew/bin/llama-server \
  -m /Users/xinzheli/models/llama.cpp/Qwen3.8-27B-Q4_K_M.gguf \
  --alias local-instruct \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 32768 \
  --jinja
```

The client Mac must be connected to the same Tailscale tailnet. The server Mac must keep
`llama-server` and Tailscale running. Tailscale Serve proxies HTTPS to
`http://127.0.0.1:8080`; Tailscale Funnel is not required and must remain disabled.

### Check the served model

```bash
curl -sS \
  https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1/models
```

### Send a normal chat request

```bash
curl -sS \
  https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "local-instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Write a short Python function that performs binary search."
      }
    ],
    "temperature": 0,
    "max_tokens": 512,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'
```

### Require a schema-constrained JSON response

Use a JSON Schema when a field must contain one of a fixed set of values. A plain
`response_format: {"type": "json_object"}` guarantees valid JSON but does not constrain the
value itself.

```bash
curl -sS \
  https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "local-instruct",
    "messages": [
      {
        "role": "system",
        "content": "Return only valid JSON matching the supplied schema."
      },
      {
        "role": "user",
        "content": "Select exactly one ID from node-a and node-b."
      }
    ],
    "temperature": 0,
    "seed": 42,
    "max_tokens": 64,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "node_selection",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "selected_id": {
              "type": "string",
              "enum": ["node-a", "node-b"]
            }
          },
          "required": ["selected_id"],
          "additionalProperties": false
        }
      }
    },
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'
```

Verified response:

```json
{
  "selected_id": "node-a"
}
```

### Call the remote model through LiTS

Run this from an editable or installed `lits-llm` environment on another tailnet Mac:

```python
import json

from lits import get_lm
from lits.lm import setup_inference_logging

model = get_lm(
    "openai/local-instruct",
    base_url=(
        "https://xinzhes-macbook-pro-2."
        "tailde6fe4.ts.net/v1"
    ),
    api_key="no-key",
)

setup_inference_logging(
    model,
    root_dir="/tmp/lits-remote-model-check",
)

output = model(
    [
        {
            "role": "system",
            "content": (
                "Return only valid JSON. The selected_id must be "
                "exactly node-a or node-b."
            ),
        },
        {
            "role": "user",
            "content": "Select one ID from node-a and node-b.",
        },
    ],
    role="evaluator_selector",
    temperature=1e-6,
    max_new_tokens=64,
    enable_thinking=False,
)

result = json.loads(output.text)

if result.get("selected_id") not in {"node-a", "node-b"}:
    raise ValueError(f"Unexpected model output: {result}")

print(result)
```

The root-level `from lits import get_lm` import requires LiTS commit `bb375cc` or later. On an
older client installation, use `from lits.lm import get_lm` until LiTS has been updated.
