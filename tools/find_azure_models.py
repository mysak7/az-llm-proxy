#!/usr/bin/env python3
"""
Find all Azure AI Foundry models available via serverless API (pay-as-you-go).

Requirements:
  az login
  az extension add --name ml
"""
import subprocess
import json
import sys
import threading
import urllib.request
import urllib.error
from collections import defaultdict

REGISTRY = "azureml"

SERVERLESS_PUBLISHERS = {
    "Microsoft", "Meta", "Mistral AI", "DeepSeek", "Cohere",
    "AI21 Labs", "NVIDIA", "TII", "01.AI", "Qwen", "xAI",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def az_json(args: list) -> dict | list:
    result = subprocess.run(["az"] + args + ["-o", "json"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def get_token() -> str:
    return az_json(["account", "get-access-token"])["accessToken"]


def get_subscription_id() -> str:
    return az_json(["account", "show"])["id"]


def fetch_page(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def spinner(stop_event: threading.Event, msg: str):
    chars = "|/-\\"
    i = 0
    while not stop_event.is_set():
        print(f"\r  {chars[i % 4]}  {msg}", end="", flush=True)
        i += 1
        stop_event.wait(0.1)
    print("\r" + " " * (len(msg) + 6) + "\r", end="")


def get_publisher(m: dict) -> str:
    tags = m.get("tags") or {}
    props = m.get("properties") or {}
    return (tags.get("publisher") or tags.get("Publisher")
            or props.get("publisher") or "Unknown")


def is_serverless(m: dict) -> bool:
    tags  = m.get("tags") or {}
    props = m.get("properties") or {}
    api_type = (tags.get("inferenceAPIType", "")
                or props.get("inferenceAPIType", "")).lower()
    if api_type == "serverless":
        return True
    return get_publisher(m) in SERVERLESS_PUBLISHERS


def print_results(latest: dict):
    serverless = [m for m in latest.values() if is_serverless(m)]
    if not serverless:
        print("No serverless models detected — showing all.\n")
        serverless = list(latest.values())

    by_pub: dict[str, list] = defaultdict(list)
    for m in serverless:
        by_pub[get_publisher(m)].append(m)

    total = sum(len(v) for v in by_pub.values())
    print(f"\n{'Publisher':<22} {'Model name':<58} {'Ver'}")
    print("─" * 90)

    for pub in sorted(by_pub.keys()):
        for m in sorted(by_pub[pub], key=lambda x: x.get("name", "")):
            tags = m.get("tags") or {}
            task = tags.get("task") or (m.get("properties") or {}).get("task") or ""
            print(f"{pub:<22} {m.get('name',''):<58} {m.get('version',''):<6}  {task}")
        print()

    print(f"Total: {total} models | {len(by_pub)} publishers\n")
    for pub in sorted(by_pub.keys(), key=lambda p: -len(by_pub[p])):
        print(f"  {pub:<22} {len(by_pub[pub]):>3} models")


# ── strategy 1: REST API with subscription ID ─────────────────────────────────

def fetch_via_rest(token: str, subscription_id: str) -> dict | None:
    api_version = "2024-04-01-preview"
    base = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/providers/Microsoft.MachineLearningServices"
        f"/registries/{REGISTRY}/models?api-version={api_version}&$top=100"
    )

    latest: dict[str, dict] = {}
    page, url = 0, base
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=(stop, "Fetching via REST API..."), daemon=True)
    t.start()

    try:
        while url:
            page += 1
            stop.clear()
            data = fetch_page(url, token)
            for m in data.get("value", []):
                name = m.get("name", "")
                ver  = m.get("id", "").split("/versions/")[-1] \
                       if "/versions/" in m.get("id", "") else "0"
                m["version"] = ver
                if name not in latest or ver > latest[name]["version"]:
                    latest[name] = m
            url = data.get("nextLink")
            stop.set()
            print(f"  Page {page} — {len(latest)} unique models", flush=True)
        return latest
    except urllib.error.HTTPError as e:
        stop.set()
        print(f"  REST API error {e.code} — falling back to az ml...", flush=True)
        return None
    finally:
        stop.set()
        t.join()


# ── strategy 2: az ml model list (slow but reliable) ─────────────────────────

def fetch_via_az_ml() -> dict:
    stop = threading.Event()
    t = threading.Thread(
        target=spinner,
        args=(stop, "Running 'az ml model list --registry-name azureml' (this takes 1-3 min)..."),
        daemon=True,
    )
    t.start()

    result = subprocess.run(
        ["az", "ml", "model", "list", "--registry-name", REGISTRY, "-o", "json"],
        capture_output=True, text=True
    )
    stop.set()
    t.join()

    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    models = json.loads(result.stdout)
    latest: dict[str, dict] = {}
    for m in models:
        name = m.get("name", "")
        ver  = m.get("version", "0")
        m["version"] = ver
        if name not in latest or ver > latest[name]["version"]:
            latest[name] = m
    return latest


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Checking Azure login...")
    az_json(["account", "show"])

    token = get_token()
    sub   = get_subscription_id()
    print(f"Subscription: {sub}\n")

    latest = fetch_via_rest(token, sub)
    if latest is None:
        latest = fetch_via_az_ml()

    print(f"\nTotal unique models fetched: {len(latest)}")
    print_results(latest)


if __name__ == "__main__":
    main()
