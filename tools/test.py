#!/usr/bin/env python3
"""
Azure proxy tester — volá modely pres local LiteLLM proxy (port 4003)
a taky primo na GitHub Models API, pak porovna vysledky.

Pouziti:
  python tools/test.py                  # interaktivni vyber modelu
  python tools/test.py all              # vsechny modely paralelne
  python tools/test.py gpt-4o           # konkretni proxy alias
  python tools/test.py gpt-4o direct   # jenom prime volani (bez proxy)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# ── Config ─────────────────────────────────────────────────────────────────────

PROXY_URL  = "http://localhost:4003/v1/chat/completions"
DIRECT_URL = "https://models.inference.ai.azure.com/chat/completions"
PROMPT     = "In one sentence, what is the capital of France?"
MAX_TOKENS = 80
TIMEOUT    = 60

# friendly alias -> GitHub Models model id
MODELS: dict[str, str] = {
    "gpt-4o":              "gpt-4o",
    "gpt-4o-mini":         "gpt-4o-mini",
    "phi-4":               "Phi-4",
    "llama-3.3-70b":       "Llama-3.3-70B-Instruct",
    "llama-3.2-90b-vision":"Llama-3.2-90B-Vision-Instruct",
    "llama-3.2-11b-vision":"Llama-3.2-11B-Vision-Instruct",
    "llama-3.1-405b":      "Meta-Llama-3.1-405B-Instruct",
    "llama-3.1-8b":        "Meta-Llama-3.1-8B-Instruct",
    "llama-4-scout":       "Llama-4-Scout-17B-16E-Instruct",
    "llama-4-maverick":    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "codestral":           "Codestral-2501",
    "mistral-medium":      "mistral-medium-2505",
    "deepseek-r1":         "DeepSeek-R1",
    "deepseek-r1-0528":    "DeepSeek-R1-0528",
    "deepseek-v3":         "DeepSeek-V3-0324",
    "grok-3":              "grok-3",
    "grok-3-mini":         "grok-3-mini",
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

_root = os.path.join(os.path.dirname(__file__), "..")
_env  = load_env(os.path.join(_root, ".env"))

PROXY_KEY = _env.get("LITELLM_MASTER_KEY", "sk-test")

def _get_gh_token() -> str:
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except Exception:
        return ""

GITHUB_TOKEN = (
    _env.get("GITHUB_TOKEN")
    or os.environ.get("GITHUB_TOKEN")
    or os.environ.get("GH_TOKEN")
    or _get_gh_token()
)

# ── Result dataclass ────────────────────────────────────────────────────────────

@dataclass
class Result:
    alias:     str
    via:       str          # "proxy" | "direct"
    ok:        bool
    latency_ms: float
    response:  str = ""
    error:     str = ""
    prompt_tokens:     int = 0
    completion_tokens: int = 0


# ── HTTP helpers (sync, run in thread) ─────────────────────────────────────────

def _post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def call_proxy(alias: str) -> Result:
    t0 = time.monotonic()
    try:
        data = _post(PROXY_URL, {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PROXY_KEY}",
        }, {
            "model": alias,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
        }, TIMEOUT)
        latency = (time.monotonic() - t0) * 1000
        content = data["choices"][0]["message"]["content"].strip()
        usage   = data.get("usage", {})
        return Result(
            alias=alias, via="proxy", ok=True, latency_ms=latency,
            response=content[:100],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
    except urllib.error.HTTPError as e:
        latency = (time.monotonic() - t0) * 1000
        return Result(alias=alias, via="proxy", ok=False, latency_ms=latency,
                      error=f"HTTP {e.code}: {e.read().decode()[:120]}")
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return Result(alias=alias, via="proxy", ok=False, latency_ms=latency,
                      error=str(e)[:120])


def call_direct(alias: str) -> Result:
    model_id = MODELS[alias]
    t0 = time.monotonic()
    try:
        data = _post(DIRECT_URL, {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GITHUB_TOKEN}",
        }, {
            "model": model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
        }, TIMEOUT)
        latency = (time.monotonic() - t0) * 1000
        content = data["choices"][0]["message"]["content"].strip()
        usage   = data.get("usage", {})
        return Result(
            alias=alias, via="direct", ok=True, latency_ms=latency,
            response=content[:100],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
    except urllib.error.HTTPError as e:
        latency = (time.monotonic() - t0) * 1000
        return Result(alias=alias, via="direct", ok=False, latency_ms=latency,
                      error=f"HTTP {e.code}: {e.read().decode()[:120]}")
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return Result(alias=alias, via="direct", ok=False, latency_ms=latency,
                      error=str(e)[:120])


# ── Async runner ────────────────────────────────────────────────────────────────

async def run_all(aliases: list[str], via: str) -> list[Result]:
    """Run all calls in parallel using a thread pool (urllib is sync)."""
    loop = asyncio.get_event_loop()
    fn   = call_proxy if via == "proxy" else call_direct
    tasks = [loop.run_in_executor(None, fn, a) for a in aliases]
    return list(await asyncio.gather(*tasks))


# ── Output ──────────────────────────────────────────────────────────────────────

def print_results(results: list[Result], via: str):
    ok  = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]

    tag = f"[{via.upper()}]"

    print(f"\n{'═'*110}")
    print(f"  {tag}  OK: {len(ok)}  Failed: {len(err)}  |  Prompt: {PROMPT!r}")
    print(f"{'═'*110}")
    print(f"  {'Model alias':<28}  {'Latency':>9}  {'In':>6}  {'Out':>5}  Response")
    print(f"  {'─'*106}")

    for r in sorted(ok, key=lambda x: x.latency_ms):
        resp = r.response[:62] + "…" if len(r.response) > 62 else r.response
        print(f"  {r.alias:<28}  {r.latency_ms:>8.0f}ms  {r.prompt_tokens:>6}  "
              f"{r.completion_tokens:>5}  {resp}")

    if err:
        print(f"\n  {'─'*106}")
        print(f"  {'Model alias':<28}  {'Latency':>9}  Error")
        print(f"  {'─'*106}")
        for r in sorted(err, key=lambda x: x.alias):
            print(f"  {r.alias:<28}  {r.latency_ms:>8.0f}ms  {r.error}")

    print(f"{'═'*110}\n")


def print_comparison(proxy_res: list[Result], direct_res: list[Result]):
    """Side-by-side latency comparison for single-model runs."""
    p = proxy_res[0]
    d = direct_res[0]

    print(f"\n{'═'*80}")
    print(f"  {p.alias}")
    print(f"{'═'*80}")
    for r in (p, d):
        status = "OK" if r.ok else "FAIL"
        latency = f"{r.latency_ms:.0f}ms"
        print(f"  [{r.via:>6}]  {status}  {latency:>8}  {r.response or r.error}")
    print(f"{'═'*80}\n")


# ── Interactive picker ──────────────────────────────────────────────────────────

def pick_models() -> list[str]:
    aliases = list(MODELS.keys())
    print("\nDostupne modely:")
    for i, a in enumerate(aliases, 1):
        print(f"  {i:>2})  {a}")
    print()
    choice = input("Cislo modelu (Enter = vsechny): ").strip()
    if not choice:
        return aliases
    try:
        return [aliases[int(choice) - 1]]
    except (ValueError, IndexError):
        print("Neplatna volba.")
        sys.exit(1)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    only_direct = "direct" in args
    args = [a for a in args if a != "direct"]

    if not args:
        selected = pick_models()
        via_list = ["proxy"] if not only_direct else ["direct"]
    elif args[0] == "all":
        selected = list(MODELS.keys())
        via_list = ["proxy"] if not only_direct else ["direct"]
    else:
        alias = args[0]
        if alias not in MODELS:
            print(f"Neznam alias '{alias}'. Dostupne: {', '.join(MODELS)}")
            sys.exit(1)
        selected = [alias]
        via_list = ["proxy", "direct"] if not only_direct else ["direct"]

    if not GITHUB_TOKEN and ("direct" in via_list or len(selected) > 0):
        print("VAROVANI: GITHUB_TOKEN neni nastaven — prime volani selzou.")

    results_by_via: dict[str, list[Result]] = {}
    for via in via_list:
        print(f"\nTestuji {len(selected)} modelu pres [{via}] paralelne…")
        results = asyncio.run(run_all(selected, via))
        results_by_via[via] = results

    # Output
    if len(selected) == 1 and len(via_list) == 2:
        print_comparison(results_by_via["proxy"], results_by_via["direct"])
    else:
        for via, results in results_by_via.items():
            print_results(results, via)


if __name__ == "__main__":
    main()
