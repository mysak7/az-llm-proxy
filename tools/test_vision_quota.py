#!/usr/bin/env python3
"""
Vision quota stress test — sends an image to llama-4-scout 30x (or until failure)
to hit the 1M token/min quota limit.

Usage:
  python tools/test_vision_quota.py
  python tools/test_vision_quota.py --repeats 50
  python tools/test_vision_quota.py --proxy       # via LiteLLM proxy instead of direct
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ── Config ─────────────────────────────────────────────────────────────────────

PROXY_URL   = "http://localhost:4003/v1/chat/completions"
DIRECT_URL  = "https://models.inference.ai.azure.com/chat/completions"
MODEL_ALIAS = "llama-4-scout"
MODEL_ID    = "Llama-4-Scout-17B-16E-Instruct"
IMAGE_URL   = "https://goodal.cz/kid/dog85.jpg"  # fetched once at startup → sent as base64
QUESTION    = "Is it a car? Answer yes or no."
MAX_TOKENS  = 20
TIMEOUT          = 60
DEFAULT_REPEATS  = 30
QUOTA_LIMIT      = 1_000_000  # tokens/min limit to back-calculate real quota cost on 429

# ── .env loader ─────────────────────────────────────────────────────────────────

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

# ── Image fetch (once at startup) ────────────────────────────────────────────────

def fetch_image_b64(url: str) -> tuple[str, int]:
    """Download image, return (base64 data URI, raw byte size)."""
    print(f"Fetching image from {url} …", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=15) as resp:
        mime = resp.headers.get_content_type() or "image/jpeg"
        raw  = resp.read()
    data = base64.b64encode(raw).decode()
    print(f"OK ({len(raw)//1024} KB)")
    return f"data:{mime};base64,{data}", len(raw)

# ── Request builder ──────────────────────────────────────────────────────────────

def build_body(model: str, image_data_uri: str) -> dict:
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": QUESTION},
            ],
        }],
        "max_tokens": MAX_TOKENS,
    }

def post(url: str, headers: dict, body: dict) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


# ── Single call ──────────────────────────────────────────────────────────────────

def call_once(via_proxy: bool, image_data_uri: str) -> tuple[bool, float, int, int, str, dict]:
    """Returns (ok, latency_ms, prompt_tokens, completion_tokens, response_or_error, raw_usage)."""
    if via_proxy:
        url     = PROXY_URL
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {PROXY_KEY}"}
        body    = build_body(MODEL_ALIAS, image_data_uri)
    else:
        url     = DIRECT_URL
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GITHUB_TOKEN}"}
        body    = build_body(MODEL_ID, image_data_uri)

    t0 = time.monotonic()
    try:
        data    = post(url, headers, body)
        latency = (time.monotonic() - t0) * 1000
        content = data["choices"][0]["message"]["content"].strip()
        usage   = data.get("usage", {})
        return True, latency, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), content, usage
    except urllib.error.HTTPError as e:
        latency = (time.monotonic() - t0) * 1000
        err     = f"HTTP {e.code}: {e.read().decode()[:200]}"
        return False, latency, 0, 0, err, {}
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return False, latency, 0, 0, str(e)[:200], {}


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    args       = sys.argv[1:]
    via_proxy  = "--proxy" in args
    repeats    = DEFAULT_REPEATS
    if "--repeats" in args:
        idx = args.index("--repeats")
        repeats = int(args[idx + 1])

    via_label = "proxy" if via_proxy else "direct"
    print(f"\nVision quota stress test — {MODEL_ALIAS} via [{via_label}]")
    print(f"Image : {IMAGE_URL}")
    print(f"Query : {QUESTION!r}")
    print(f"Repeat: up to {repeats}x  |  max_tokens={MAX_TOKENS}")

    image_data_uri, image_bytes = fetch_image_b64(IMAGE_URL)
    print(f"Image size: {image_bytes // 1024} KB  (token cost per request reported after first call)")

    print(f"\n{'═'*90}\n")
    print(f"  {'#':>3}  {'Status':<6}  {'Latency':>9}  {'In/req':>8}  {'Out':>5}  {'TotalIn':>9}  {'TotalOut':>9}  Response")
    print(f"  {'─'*88}")

    total_prompt     = 0
    total_completion = 0
    attempts         = 0
    hit_429          = False
    run_start        = time.monotonic()

    for i in range(1, repeats + 1):
        attempts = i
        ok, latency, pt, ct, resp, _ = call_once(via_proxy, image_data_uri)
        total_prompt     += pt
        total_completion += ct
        status = "OK   " if ok else "FAIL "
        resp_short = resp[:40] + "…" if len(resp) > 40 else resp
        print(f"  {i:>3}  {status}  {latency:>8.0f}ms  {pt:>8}  {ct:>5}  {total_prompt:>9}  {total_completion:>9}  {resp_short}")

        if not ok:
            elapsed = time.monotonic() - run_start
            hit_429 = "429" in resp
            print(f"\n  --> Failed on attempt {i} after {elapsed:.1f}s")
            break

    elapsed     = time.monotonic() - run_start
    ok_count    = attempts - (1 if hit_429 else 0)
    avg_pt      = total_prompt // ok_count if ok_count else 0
    grand_total = total_prompt + total_completion

    print(f"\n{'═'*90}")
    print(f"  Attempts: {attempts}  |  OK: {ok_count}  |  Elapsed: {elapsed:.1f}s")
    print(f"  Reported by API — prompt: {total_prompt:>9}  completion: {total_completion:>7}  total: {grand_total:>9}")
    print(f"  Avg reported tokens/req : {avg_pt:>9}")
    if hit_429 and ok_count > 0:
        implied = QUOTA_LIMIT // ok_count
        ratio   = implied // avg_pt if avg_pt else "?"
        print(f"  ── 429 hit after {ok_count} OK requests ──────────────────────────────────────────")
        print(f"  Implied quota cost/req  : {implied:>9}  (= {QUOTA_LIMIT:,} limit ÷ {ok_count} requests)")
        print(f"  Quota vs reported ratio : {ratio:>9}x  (image tokens cost ~{ratio}x more than usage shows)")
    print(f"{'═'*90}\n")


if __name__ == "__main__":
    main()
