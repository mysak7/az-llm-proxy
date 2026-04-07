#!/usr/bin/env python3
"""
Azure proxy tester — calls models via local LiteLLM proxy (port 4003)
and directly against Azure AI Foundry, then compares results.

Usage:
  python tools/test.py                   # test proxy only
  python tools/test.py all               # same
  python tools/test.py direct            # Azure direct only
  python tools/test.py both              # proxy + direct side by side
  python tools/test.py tools             # tool calling test via proxy
  python tools/test.py tools direct      # tool calling test via Azure direct
  python tools/test.py tools both        # tool calling test proxy + direct
"""

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# ── Config ─────────────────────────────────────────────────────────────────────

PROXY_URL  = "http://localhost:4003/v1/chat/completions"
DIRECT_URL = "https://llm-development-artifacts.services.ai.azure.com/models/chat/completions"
PROMPT     = "What AI model are you? Answer in one sentence."
MAX_TOKENS = 200
TIMEOUT    = 60

MODELS: dict[str, str] = {
    # proxy alias (lowercase) : Azure Foundry model name
    "deepseek-v3.2-speciale":      "DeepSeek-V3.2-Speciale",
    "phi-4-reasoning":             "Phi-4-reasoning",
    "phi-4":                       "Phi-4",
    "phi-4-multimodal-instruct":   "Phi-4-multimodal-instruct",
}

# ── .env loader ────────────────────────────────────────────────────────────────

def load_env(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env

_root     = os.path.join(os.path.dirname(__file__), "..")
_env      = load_env(os.path.join(_root, ".env"))
PROXY_KEY = _env.get("LITELLM_MASTER_KEY", "sk-test")
AZURE_KEY = _env.get("AZURE_AI_KEY", "")

# ── Result ──────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    alias:             str
    via:               str
    ok:                bool
    latency_ms:        float
    response:          str = ""
    error:             str = ""
    prompt_tokens:     int = 0
    completion_tokens: int = 0


# ── HTTP ────────────────────────────────────────────────────────────────────────

def _post(url: str, headers: dict, body: dict) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def call(alias: str, via: str) -> Result:
    t0 = time.monotonic()
    try:
        if via == "proxy":
            url     = PROXY_URL
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {PROXY_KEY}"}
            model   = alias
        else:
            url     = DIRECT_URL
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {AZURE_KEY}"}
            model   = MODELS[alias]

        data    = _post(url, headers, {"model": model,
                                       "messages": [{"role": "user", "content": PROMPT}],
                                       "max_tokens": MAX_TOKENS})
        latency = (time.monotonic() - t0) * 1000
        content = data["choices"][0]["message"]["content"].strip()
        usage   = data.get("usage", {})
        return Result(alias=alias, via=via, ok=True, latency_ms=latency,
                      response=content[:120],
                      prompt_tokens=usage.get("prompt_tokens", 0),
                      completion_tokens=usage.get("completion_tokens", 0))

    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        latency = (time.monotonic() - t0) * 1000
        return Result(alias=alias, via=via, ok=False, latency_ms=latency,
                      error=f"HTTP {e.code}: {body}")
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return Result(alias=alias, via=via, ok=False, latency_ms=latency,
                      error=str(e)[:200])


# ── Async runner ────────────────────────────────────────────────────────────────

async def run_all(aliases: list[str], via: str) -> list[Result]:
    loop  = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, call, a, via) for a in aliases]
    return list(await asyncio.gather(*tasks))


# ── Output ──────────────────────────────────────────────────────────────────────

def print_results(results: list[Result]):
    ok  = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]

    print(f"\n{'═'*100}")
    print(f"  OK: {len(ok)}  Failed: {len(err)}  |  Prompt: {PROMPT!r}")
    print(f"{'═'*100}")
    print(f"  {'Via':<8} {'Model alias':<20}  {'Latency':>9}  {'In':>6}  {'Out':>5}  Response")
    print(f"  {'─'*96}")
    for r in sorted(ok, key=lambda x: x.latency_ms):
        print(f"  {r.via:<8} {r.alias:<20}  {r.latency_ms:>8.0f}ms  "
              f"{r.prompt_tokens:>6}  {r.completion_tokens:>5}  {r.response}")
    if err:
        print(f"\n  {'─'*96}")
        for r in err:
            print(f"  {r.via:<8} {r.alias:<20}  {r.latency_ms:>8.0f}ms  ERROR: {r.error}")
    print(f"{'═'*100}\n")


# ── Tool calling test ───────────────────────────────────────────────────────────

TOOL_PROMPT = "What is the weather in Prague and in Tokyo?"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }
]

@dataclass
class ToolResult:
    alias:      str
    via:        str
    ok:         bool
    latency_ms: float
    called:     bool   = False   # did model make a tool call?
    calls:      list   = None    # list of (name, args) tuples
    error:      str    = ""
    finish:     str    = ""

    def __post_init__(self):
        if self.calls is None:
            self.calls = []


def call_tool(alias: str, via: str) -> ToolResult:
    t0 = time.monotonic()
    try:
        if via == "proxy":
            url     = PROXY_URL
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {PROXY_KEY}"}
            model   = alias
        else:
            url     = DIRECT_URL
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {AZURE_KEY}"}
            model   = MODELS[alias]

        body = {
            "model": model,
            "messages": [{"role": "user", "content": TOOL_PROMPT}],
            "tools": TOOLS,
            "max_tokens": 300,
        }
        data    = _post(url, headers, body)
        latency = (time.monotonic() - t0) * 1000
        msg     = data["choices"][0]["message"]
        finish  = data["choices"][0].get("finish_reason", "?")
        tcs     = msg.get("tool_calls") or []
        calls   = []
        for tc in tcs:
            fn   = tc.get("function", {})
            name = fn.get("name", "?")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = fn.get("arguments", "")
            calls.append((name, args))

        return ToolResult(alias=alias, via=via, ok=True, latency_ms=latency,
                          called=bool(calls), calls=calls, finish=finish)

    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(alias=alias, via=via, ok=False, latency_ms=latency,
                          error=f"HTTP {e.code}: {body}")
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return ToolResult(alias=alias, via=via, ok=False, latency_ms=latency,
                          error=str(e)[:200])


async def run_tools_all(aliases: list[str], via: str) -> list[ToolResult]:
    loop  = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, call_tool, a, via) for a in aliases]
    return list(await asyncio.gather(*tasks))


def print_tool_results(results: list[ToolResult]):
    ok  = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]

    print(f"\n{'═'*110}")
    print(f"  TOOL CALLING TEST  |  Prompt: {TOOL_PROMPT!r}")
    print(f"  Tool: get_weather(location)  —  expecting 2 calls (Prague + Tokyo)")
    print(f"{'═'*110}")
    print(f"  {'Via':<8} {'Model alias':<28}  {'Latency':>9}  {'Called':>6}  {'finish':>10}  Tool calls")
    print(f"  {'─'*106}")
    for r in sorted(ok, key=lambda x: x.latency_ms):
        called_str = "✓ YES" if r.called else "✗ NO"
        if r.calls:
            calls_str = "  ".join(f"{n}({a})" for n, a in r.calls)
        else:
            calls_str = "—"
        print(f"  {r.via:<8} {r.alias:<28}  {r.latency_ms:>8.0f}ms  {called_str:>6}  {r.finish:>10}  {calls_str}")
    if err:
        print(f"\n  {'─'*106}")
        for r in err:
            print(f"  {r.via:<8} {r.alias:<28}  {r.latency_ms:>8.0f}ms  ERROR: {r.error}")
    print(f"{'═'*110}\n")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    mode = args[0] if args else "all"

    if mode == "tools":
        sub = args[1] if len(args) > 1 else "proxy"
        if sub in ("proxy", "all"):
            via_list = ["proxy"]
        elif sub == "direct":
            if not AZURE_KEY:
                print("AZURE_AI_KEY not set in .env")
                sys.exit(1)
            via_list = ["direct"]
        elif sub == "both":
            if not AZURE_KEY:
                print("AZURE_AI_KEY not set in .env")
                sys.exit(1)
            via_list = ["proxy", "direct"]
        else:
            via_list = ["proxy"]

        aliases = list(MODELS.keys())
        all_results: list[ToolResult] = []
        for via in via_list:
            print(f"Tool calling test — {len(aliases)} model(s) via [{via}]…")
            results = asyncio.run(run_tools_all(aliases, via))
            all_results.extend(results)
        print_tool_results(all_results)
        return

    if mode in ("all", "proxy"):
        via_list = ["proxy"]
    elif mode == "direct":
        if not AZURE_KEY:
            print("AZURE_AI_KEY not set in .env")
            sys.exit(1)
        via_list = ["direct"]
    elif mode == "both":
        if not AZURE_KEY:
            print("AZURE_AI_KEY not set in .env")
            sys.exit(1)
        via_list = ["proxy", "direct"]
    else:
        print(f"Unknown mode '{mode}'. Use: all | proxy | direct | both | tools [proxy|direct|both]")
        sys.exit(1)

    aliases = list(MODELS.keys())
    all_results: list[Result] = []

    for via in via_list:
        print(f"Testing {len(aliases)} model(s) via [{via}]…")
        results = asyncio.run(run_all(aliases, via))
        all_results.extend(results)

    print_results(all_results)


if __name__ == "__main__":
    main()
