#!/usr/bin/env python3
"""
Azure / GitHub Models parallel tester.
Uses GitHub Models API (backed by Azure AI) — no quota approvals needed.
"""

import asyncio
import time
import os
import subprocess
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q",
                           "--break-system-packages"])
    import aiohttp

# ── Config ─────────────────────────────────────────────────────────────────────

ENDPOINT = "https://models.inference.ai.azure.com/chat/completions"
PROMPT    = "In one sentence, what is the capital of France?"
MAX_TOKENS = 80
TIMEOUT    = 60  # seconds per request

# Known-working models on GitHub Models (backed by Azure AI)
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "Phi-4",
    "Llama-3.3-70B-Instruct",
    "Llama-3.2-90B-Vision-Instruct",
    "Llama-3.2-11B-Vision-Instruct",
    "Llama-4-Scout-17B-16E-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Meta-Llama-3.1-405B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Codestral-2501",
    "mistral-medium-2505",
    "DeepSeek-R1",
    "DeepSeek-R1-0528",
    "DeepSeek-V3-0324",
    "grok-3",
    "grok-3-mini",
]

# ── Pricing table (Azure AI Foundry pay-as-you-go, USD per 1M tokens) ─────────
# Source: Azure AI Foundry model catalog pricing (April 2025)
PRICING = {
    # model_id                                : (input_$/1M, output_$/1M)
    "gpt-4o":                                   (2.50,   10.00),
    "gpt-4o-mini":                              (0.15,    0.60),
    "Phi-4":                                    (0.07,    0.28),
    "Llama-3.3-70B-Instruct":                  (0.77,    0.77),
    "Llama-3.2-90B-Vision-Instruct":           (2.48,    2.48),
    "Llama-3.2-11B-Vision-Instruct":           (0.37,    0.37),
    "Llama-4-Scout-17B-16E-Instruct":          (0.17,    0.17),
    "Llama-4-Maverick-17B-128E-Instruct-FP8":  (0.35,    1.10),
    "Meta-Llama-3.1-405B-Instruct":            (5.33,   16.00),
    "Meta-Llama-3.1-8B-Instruct":              (0.30,    0.61),
    "Codestral-2501":                           (0.27,    0.27),
    "mistral-medium-2505":                      (0.40,    2.00),
    "DeepSeek-R1":                              (3.00,    7.00),
    "DeepSeek-R1-0528":                         (3.00,    7.00),
    "DeepSeek-V3-0324":                         (0.27,    1.10),
    "grok-3":                                   (3.00,   15.00),
    "grok-3-mini":                              (0.30,    0.50),
}

# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    model: str
    ok: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response: str = ""
    error: str = ""
    cost_per_1k_input: float = field(init=False)
    cost_per_1k_output: float = field(init=False)

    def __post_init__(self):
        p = PRICING.get(self.model, (0.0, 0.0))
        self.cost_per_1k_input  = p[0] / 1000
        self.cost_per_1k_output = p[1] / 1000

# ── Core request ───────────────────────────────────────────────────────────────

async def call_model(session: aiohttp.ClientSession, token: str, model: str) -> Result:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    t0 = time.monotonic()
    try:
        async with session.post(
            ENDPOINT, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        ) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            data = await resp.json(content_type=None)

        if "choices" in data:
            msg   = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            # Strip DeepSeek <think>...</think> blocks from display
            import re
            clean = re.sub(r"<think>.*?</think>\s*", "", msg, flags=re.DOTALL).strip()
            return Result(
                model=model, ok=True, latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                response=clean[:120],
            )
        else:
            err = data.get("error", {})
            return Result(
                model=model, ok=False, latency_ms=latency_ms,
                error=f"{err.get('code','?')}: {err.get('message','')[:80]}",
            )
    except asyncio.TimeoutError:
        return Result(model=model, ok=False, latency_ms=TIMEOUT * 1000,
                      error="timeout")
    except Exception as e:
        return Result(model=model, ok=False, latency_ms=(time.monotonic() - t0) * 1000,
                      error=str(e)[:80])

# ── Formatting ─────────────────────────────────────────────────────────────────

def print_results(results: list[Result]):
    ok  = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]

    # ── Working models ─────────────────────────────────────────────────────
    print("\n" + "═" * 130)
    print(f"{'WORKING MODELS':^130}")
    print("═" * 130)

    # header
    print(f"{'Model':<48} {'Latency':>9}  {'Prompt':>7}  {'Compl':>6}  "
          f"{'$/1k-in':>8}  {'$/1k-out':>9}  Response")
    print("─" * 130)

    for r in sorted(ok, key=lambda x: x.latency_ms):
        price_in  = f"${r.cost_per_1k_input:.4f}"  if r.cost_per_1k_input  else "—"
        price_out = f"${r.cost_per_1k_output:.4f}" if r.cost_per_1k_output else "—"
        resp_short = r.response[:55] + "…" if len(r.response) > 55 else r.response
        print(f"{r.model:<48} {r.latency_ms:>8.0f}ms  {r.prompt_tokens:>7}  "
              f"{r.completion_tokens:>6}  {price_in:>8}  {price_out:>9}  {resp_short}")

    # ── Full pricing table ─────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print(f"{'PRICING TABLE  (Azure AI Foundry, pay-as-you-go)':^90}")
    print("═" * 90)
    print(f"{'Model':<48}  {'Input $/1M':>12}  {'Output $/1M':>12}  {'Cost/1k ctx':>12}")
    print("─" * 90)
    for r in sorted(ok, key=lambda x: PRICING.get(x.model, (0, 0))[0]):
        p = PRICING.get(r.model, (None, None))
        if p[0] is None:
            pi, po, pc = "—", "—", "—"
        else:
            pi = f"${p[0]:.2f}"
            po = f"${p[1]:.2f}"
            # approx cost for a 1k-token context + 200-token response
            pc = f"${p[0]/1000 + p[1]/1000*0.2:.5f}"
        print(f"{r.model:<48}  {pi:>12}  {po:>12}  {pc:>12}")

    # ── Failed models ──────────────────────────────────────────────────────
    if err:
        print("\n" + "═" * 90)
        print(f"{'FAILED MODELS':^90}")
        print("═" * 90)
        print(f"{'Model':<48}  {'Latency':>9}  Error")
        print("─" * 90)
        for r in sorted(err, key=lambda x: x.model):
            print(f"{r.model:<48}  {r.latency_ms:>8.0f}ms  {r.error}")

    print("\n" + "─" * 90)
    print(f"  Tested: {len(results)}  ✓ OK: {len(ok)}  ✗ Failed: {len(err)}")
    print("─" * 90)

# ── Entry point ────────────────────────────────────────────────────────────────

def get_github_token() -> str:
    # 1. env var
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if t:
        return t
    # 2. gh CLI
    try:
        t = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        if t:
            return t
    except Exception:
        pass
    raise RuntimeError(
        "No GitHub token found. Set GITHUB_TOKEN env var or run `gh auth login`."
    )

async def main():
    token = get_github_token()
    print(f"Endpoint : {ENDPOINT}")
    print(f"Prompt   : {PROMPT!r}")
    print(f"Models   : {len(MODELS)}")
    print(f"Running all {len(MODELS)} models in parallel…\n")

    connector = aiohttp.TCPConnector(limit=len(MODELS))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [call_model(session, token, m) for m in MODELS]
        results = await asyncio.gather(*tasks)

    print_results(list(results))

if __name__ == "__main__":
    asyncio.run(main())
