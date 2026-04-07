# az-llm-proxy

LiteLLM proxy for Azure AI Foundry models. Exposes a single OpenAI-compatible
endpoint on port 4003 that fans out to Azure-hosted models.

## Models

| Proxy alias | Azure Foundry model |
|---|---|
| `deepseek-v3.2-speciale` | DeepSeek-V3.2-Speciale |
| `phi-4-reasoning` | Phi-4-reasoning |
| `phi-4` | Phi-4 |
| `phi-4-multimodal-instruct` | Phi-4-multimodal-instruct |

## Running

```bash
cp .env.example .env   # fill AZURE_AI_KEY, LITELLM_MASTER_KEY, POSTGRES_PASSWORD
docker compose up -d
```

Proxy is available at `http://localhost:4003/v1`.

## Tool calling for Phi models

Azure AI Foundry hosts Phi models on vLLM **without** `--enable-auto-tool-choice`
and `--tool-call-parser` flags, so native OpenAI-format tool calling is rejected.

`custom_hooks.py` implements a transparent translation layer (`PhiTextToolCalling`):

### Request (client → Azure)

When a request for a Phi model contains a `tools` array:

1. Tool definitions are serialised to JSON and injected into the system prompt
   with instructions to respond using `<tool_call>` XML blocks.
2. `tools` and `tool_choice` are removed from the payload before it reaches Azure.

### Response (Azure → client)

The hook inspects the text response for `<tool_call>` blocks:

```
<tool_call>
{"name": "get_weather", "arguments": {"location": "Prague"}}
</tool_call>
```

If found, the hook:
- Removes the blocks from `content`
- Populates `message.tool_calls` in standard OpenAI format
- Sets `finish_reason` to `"tool_calls"`

The client (e.g. OpenClaw) sees a perfectly normal OpenAI tool-call response and
never knows the translation happened.

### Limitations

- **phi-4-reasoning** does not follow the injected tool-call format — it answers
  questions directly with extensive CoT reasoning instead of emitting `<tool_call>`
  blocks. Tool calling is not usable with this model.
- Streaming responses are not translated (tool calls appear as plain text).
- If the model produces malformed JSON inside a `<tool_call>` block the entire
  response is returned as plain text.

## Testing

```bash
# Basic model availability
python3 tools/test.py

# Tool calling test (via proxy)
python3 tools/test.py tools

# Tool calling test (directly against Azure, bypasses proxy)
python3 tools/test.py tools direct

# Both
python3 tools/test.py tools both
```

## Files

| File | Purpose |
|---|---|
| `litellm_config.yaml` | Model list and router settings |
| `custom_hooks.py` | Phi tool-call translation hook |
| `docker-compose.yml` | Proxy + Postgres services |
| `tools/test.py` | Model and tool-call smoke tests |
| `tools/find_azure_models.py` | Discover available Azure Foundry models |
