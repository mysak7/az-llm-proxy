import json
import re
import uuid
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger

# ── System prompt snippet injected when tools are present ──────────────────────

_TOOL_SYSTEM_PROMPT = """\
You have access to the following tools. When you want to call a tool, output ONLY \
one or more <tool_call> blocks — no prose before or after. \
For multiple parallel calls include multiple blocks.

Format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

Available tools:
{tools_json}"""

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _build_tool_snippet(tools: list) -> str:
    definitions = [t["function"] for t in tools if t.get("type") == "function"]
    return _TOOL_SYSTEM_PROMPT.format(tools_json=json.dumps(definitions, indent=2))


def _parse_tool_calls(text: str) -> Optional[list]:
    """Extract <tool_call> blocks and return OpenAI tool_calls list, or None."""
    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        return None
    calls = []
    for raw in matches:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None  # malformed block — treat whole response as plain text
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": json.dumps(obj.get("arguments", {})),
            },
        })
    return calls


# ── Hook ───────────────────────────────────────────────────────────────────────

class PhiTextToolCalling(CustomLogger):
    """Text-based tool calling bridge for Phi models on Azure AI Foundry.

    Azure's vLLM backend for Phi does not support native tool calling
    (requires --enable-auto-tool-choice / --tool-call-parser flags not set).

    Request  (client → Azure):
      - Converts tools definitions to a system prompt snippet
      - Removes tools and tool_choice from the payload

    Response (Azure → client):
      - Detects <tool_call>…</tool_call> blocks in the text
      - Rewrites the response into standard OpenAI tool_calls format
        with finish_reason "tool_calls"
    """

    def __init__(self):
        super().__init__()
        self._pending: dict[str, list] = {}  # call_id → original tools

    @staticmethod
    def _is_phi(model: str) -> bool:
        return "phi" in model.lower()

    # ── pre-call: inject tools into system prompt ──────────────────────────────

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        model = data.get("model", "")
        tools = data.get("tools")
        if not self._is_phi(model) or not tools:
            return data

        call_id = data.get("litellm_call_id", "")
        self._pending[call_id] = tools

        snippet = _build_tool_snippet(tools)
        messages = data.get("messages", [])
        sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if sys_idx is not None:
            messages[sys_idx] = {
                **messages[sys_idx],
                "content": messages[sys_idx]["content"] + "\n\n" + snippet,
            }
        else:
            messages.insert(0, {"role": "system", "content": snippet})

        data.pop("tools", None)
        data.pop("tool_choice", None)
        return data

    # ── post-call: parse <tool_call> blocks and rewrite response ──────────────

    async def async_post_call_success_hook(self, data, user_api_key_dict, response) -> Any:
        call_id = data.get("litellm_call_id", "")
        tools = self._pending.pop(call_id, None)
        if tools is None:
            return response

        choices = getattr(response, "choices", None)
        if not choices:
            return response

        choice = choices[0]
        msg = choice.message
        content = getattr(msg, "content", "") or ""
        tool_calls = _parse_tool_calls(content)
        if not tool_calls:
            return response

        # Strip <tool_call> blocks from content; None signals no text content
        clean = _TOOL_CALL_RE.sub("", content).strip() or None
        msg.content = clean
        msg.tool_calls = tool_calls
        choice.finish_reason = "tool_calls"
        return response

    # ── cleanup on failure so _pending doesn't grow ───────────────────────────

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        call_id = kwargs.get("litellm_call_id", "")
        self._pending.pop(call_id, None)


# Module-level instance — name must match litellm_config.yaml callbacks entry
phi_tool_choice_fix = PhiTextToolCalling()
