#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    inv_file = repo_root / ".scratch" / "rocm-tensor-parallel" / "inventory.json"
    if not inv_file.exists():
        print(f"Error: Inventory file {inv_file} not found.", file=sys.stderr)
        sys.exit(1)

    with open(inv_file, "r") as f:
        data = json.load(f)

    total = len(data["entry_points"])
    implemented = sum(1 for ep in data["entry_points"] if ep["status"] == "implemented")
    stubbed = sum(1 for ep in data["entry_points"] if ep["status"] == "stubbed")

    print(f"ROCm TP Inventory Status: {implemented}/{total} implemented ({stubbed} stubbed)")
    for ep in data["entry_points"]:
        status_symbol = "✓" if ep["status"] == "implemented" else "✗"
        print(f"  [{status_symbol}] {ep['name']} ({ep['category']}) -> {ep['status']}")

    if data.get("summary", {}).get("total_entry_points") != total:
        print(f"Warning: summary.total_entry_points mismatch: expected {total}, found {data.get('summary', {}).get('total_entry_points')}", file=sys.stderr)
        sys.exit(1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
