#!/usr/bin/env python3
"""
Find all Azure AI Foundry models available via serverless API (pay-as-you-go).
These are the models you can use with the $200 free Azure credit.

Requirements:
  az login
  az extension add --name ml
"""
import subprocess
import json
import sys
from collections import defaultdict

# Tags/properties that indicate a model supports serverless (pay-as-you-go) API.
# Azure uses these in the azureml registry to mark MaaS-capable models.
SERVERLESS_TAGS = {
    "inferenceAPIType": "serverless",
    "inference-min-sku-spec": None,   # presence only
}

def az(cmd: str) -> list | dict:
    result = subprocess.run(
        f"az {cmd} -o json",
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"\n[ERROR] az {cmd}\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def is_serverless(model: dict) -> bool:
    tags = model.get("tags") or {}
    props = model.get("properties") or {}

    # Check known serverless indicator tags
    if tags.get("inferenceAPIType", "").lower() == "serverless":
        return True
    # Some models use properties instead of tags
    if props.get("inferenceAPIType", "").lower() == "serverless":
        return True
    # Presence of inference-min-sku-spec usually means dedicated, not serverless
    # Absence + having task type = likely serverless candidate
    task = tags.get("task") or props.get("task") or ""
    if task and "inference-min-sku-spec" not in tags and "inference-min-sku-spec" not in props:
        return True
    return False


def get_publisher(model: dict) -> str:
    tags = model.get("tags") or {}
    return (
        tags.get("publisher")
        or tags.get("Publisher")
        or tags.get("author")
        or "Unknown"
    )


def main():
    print("Checking az login...")
    try:
        az("account show")
    except SystemExit:
        print("Run: az login")
        sys.exit(1)

    print("Fetching model list from Azure ML registry (azureml)...\n")
    all_models = az("ml model list --registry-name azureml")

    # Keep only the latest version per model name
    latest: dict[str, dict] = {}
    for m in all_models:
        name = m.get("name", "")
        ver  = m.get("version", "0")
        if name not in latest or ver > latest[name].get("version", "0"):
            latest[name] = m

    serverless = [m for m in latest.values() if is_serverless(m)]

    if not serverless:
        print("No serverless models found — try without the filter:")
        print("  az ml model list --registry-name azureml -o table\n")
        # Fall back: show everything grouped by publisher
        serverless = list(latest.values())

    # Group by publisher
    by_pub: dict[str, list] = defaultdict(list)
    for m in serverless:
        by_pub[get_publisher(m)].append(m)

    total = sum(len(v) for v in by_pub.values())
    print(f"Found {total} serverless / pay-as-you-go models across {len(by_pub)} publishers:\n")
    print(f"{'Publisher':<22} {'Model name':<55} {'Version'}")
    print("─" * 90)

    for pub in sorted(by_pub.keys()):
        models = sorted(by_pub[pub], key=lambda m: m.get("name", ""))
        for m in models:
            tags  = m.get("tags") or {}
            props = m.get("properties") or {}
            task  = tags.get("task") or props.get("task") or ""
            name  = m.get("name", "")
            ver   = m.get("version", "")
            print(f"{pub:<22} {name:<55} {ver:<10}  {task}")
        print()

    # Summary: unique publishers
    print("\nPublishers found:")
    for pub in sorted(by_pub.keys()):
        print(f"  • {pub} ({len(by_pub[pub])} models)")


if __name__ == "__main__":
    main()
