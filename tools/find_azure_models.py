#!/usr/bin/env python3
"""
Find all Azure AI Foundry models available via serverless API (pay-as-you-go).
Uses REST API directly with pagination — much faster than 'az ml model list'.

Requirements:
  az login
  az extension add --name ml
"""
import subprocess
import json
import sys
import urllib.request
import urllib.error
from collections import defaultdict

API_VERSION = "2024-04-01-preview"
REGISTRY    = "azureml"
BASE_URL    = (
    f"https://management.azure.com/providers/Microsoft.MachineLearningServices"
    f"/registries/{REGISTRY}/models?api-version={API_VERSION}&$top=100"
)

# Publishers whose models are known to support serverless / pay-as-you-go
SERVERLESS_PUBLISHERS = {
    "Microsoft", "Meta", "Mistral AI", "DeepSeek", "Cohere",
    "AI21 Labs", "NVIDIA", "TII", "01.AI", "Qwen",
}


def get_token() -> str:
    result = subprocess.run(
        ["az", "account", "get-access-token", "--output", "json"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Not logged in. Run: az login", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)["accessToken"]


def fetch_page(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)


def get_publisher(model: dict) -> str:
    tags = model.get("tags") or {}
    props = (model.get("properties") or {})
    return (
        tags.get("publisher")
        or tags.get("Publisher")
        or props.get("publisher")
        or "Unknown"
    )


def is_serverless(model: dict) -> bool:
    tags  = model.get("tags") or {}
    props = model.get("properties") or {}
    api_type = (
        tags.get("inferenceAPIType", "")
        or props.get("inferenceAPIType", "")
    ).lower()
    if api_type == "serverless":
        return True
    # Fallback: publisher-based heuristic
    pub = get_publisher(model)
    return pub in SERVERLESS_PUBLISHERS


def main():
    print("Getting Azure token...")
    token = get_token()

    print("Scanning Azure ML registry (paginated, shows results live)...\n")

    # Track only latest version per model name
    latest: dict[str, dict] = {}
    page_num = 0
    url = BASE_URL

    while url:
        page_num += 1
        print(f"\r  Page {page_num}  ({len(latest)} unique models so far)...", end="", flush=True)
        data = fetch_page(url, token)
        items = data.get("value", [])

        for m in items:
            name = m.get("name", "")
            ver  = m.get("id", "").split("/versions/")[-1] if "/versions/" in m.get("id","") else m.get("version","0")
            m["version"] = ver
            if name not in latest or ver > latest[name]["version"]:
                latest[name] = m

        url = data.get("nextLink")

    print(f"\r  Done. {len(latest)} unique models found across {page_num} pages.\n")

    # Filter serverless
    serverless = [m for m in latest.values() if is_serverless(m)]

    if not serverless:
        print("No serverless models detected by tags — showing all models.\n")
        serverless = list(latest.values())

    # Group by publisher
    by_pub: dict[str, list] = defaultdict(list)
    for m in serverless:
        by_pub[get_publisher(m)].append(m)

    total = sum(len(v) for v in by_pub.values())
    print(f"{'Publisher':<22} {'Model name':<58} {'Ver'}")
    print("─" * 95)

    for pub in sorted(by_pub.keys()):
        models = sorted(by_pub[pub], key=lambda m: m.get("name", ""))
        for m in models:
            tags = m.get("tags") or {}
            task = tags.get("task") or (m.get("properties") or {}).get("task") or ""
            print(f"{pub:<22} {m.get('name',''):<58} {m['version']:<6}  {task}")
        print()

    print(f"Total: {total} serverless-capable models from {len(by_pub)} publishers")
    print("\nPublisher summary:")
    for pub in sorted(by_pub.keys(), key=lambda p: -len(by_pub[p])):
        print(f"  {pub:<22} {len(by_pub[pub]):>3} models")


if __name__ == "__main__":
    main()
