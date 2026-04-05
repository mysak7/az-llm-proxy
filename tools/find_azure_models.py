#!/usr/bin/env python3
"""
List all Microsoft/Azure models from the azureml registry.
Takes 1-3 minutes — az ml must download the full model list first.

Requirements:
  az login
  az extension add --name ml
"""
import subprocess
import json
import sys
import threading

QUERY = "[?tags.publisher=='Microsoft' || tags.Publisher=='Microsoft'].{name:name, version:version, task:tags.task}"


def spinner(stop: threading.Event):
    chars = "|/-\\"
    i = 0
    while not stop.is_set():
        print(f"\r  {chars[i % 4]}  Downloading model list (1-3 min, az ml fetches everything)...", end="", flush=True)
        i += 1
        stop.wait(0.15)
    print("\r" + " " * 70 + "\r", end="")


def main():
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=(stop,), daemon=True)
    t.start()

    result = subprocess.run(
        ["az", "ml", "model", "list",
         "--registry-name", "azureml",
         "--query", QUERY,
         "-o", "json"],
        capture_output=True, text=True
    )
    stop.set()
    t.join()

    if result.returncode != 0:
        print(f"Error:\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    models = json.loads(result.stdout)

    if not models:
        print("No Microsoft models found.")
        sys.exit(0)

    # Keep only latest version per model name
    latest: dict[str, dict] = {}
    for m in models:
        name = m.get("name", "")
        ver  = m.get("version", "0")
        if name not in latest or ver > latest[name]["version"]:
            latest[name] = m

    # Group by task
    by_task: dict[str, list] = {}
    for m in latest.values():
        task = m.get("task") or "other"
        by_task.setdefault(task, []).append(m["name"])

    print(f"Microsoft models in azureml registry: {len(latest)}\n")
    print(f"{'Model name':<55} {'Ver':<8} {'Task'}")
    print("─" * 85)

    for m in sorted(latest.values(), key=lambda x: (x.get("task") or "", x["name"])):
        print(f"{m['name']:<55} {m['version']:<8} {m.get('task') or ''}")

    print(f"\n── By task ──")
    for task in sorted(by_task.keys()):
        print(f"\n  {task}:")
        for name in sorted(by_task[task]):
            print(f"    {name}")


if __name__ == "__main__":
    main()
