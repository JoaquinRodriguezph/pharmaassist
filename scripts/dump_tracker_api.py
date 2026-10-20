"""Print the tracker's GET endpoints with their full parameter schemas and a sample response.

The tracker must be running. From the project root:

    py scripts/dump_tracker_api.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx

from app.tools.tracker_client import TrackerClient, TrackerError

SAMPLE_ITEMS = 3


def show(value) -> str:
    if isinstance(value, list):
        head = value[:SAMPLE_ITEMS]
        return f"list of {len(value)}; first {len(head)}:\n" + json.dumps(head, indent=1, ensure_ascii=False)
    return json.dumps(value, indent=1, ensure_ascii=False)


def main() -> int:
    client = TrackerClient()
    try:
        spec = httpx.get(f"{client.base_url}/openapi.json", timeout=5.0).json()
    except httpx.HTTPError as error:
        print(f"ERROR: cannot read {client.base_url}/openapi.json: {error}")
        return 1

    print(f"=== parameters ({client.base_url})")
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            print(f"{method.upper()} {path}")
            for p in op.get("parameters", []):
                print(f"    {p['name']} in={p['in']} required={p.get('required', False)} "
                      f"schema={json.dumps(p.get('schema', {}))}")

    print("\n=== sample responses")
    branch_name = None
    try:
        branches = client.branches()
        print(f"/branches -> {show(branches)}")
        if isinstance(branches, list) and branches and isinstance(branches[0], dict):
            branch_name = branches[0].get("name") or branches[0].get("branch_name")
    except TrackerError as error:
        print(f"/branches -> ERROR {error}")

    calls = [
        ("/health", {}),
        ("/reports/low-stock", {"branch_id": 1}),
        ("/reports/monthly-sales", {"branch_name": branch_name}),
        ("/reports/expiring", {"branch_id": 1, "days": 30}),
        ("/reports/top-sellers", {"branch_name": branch_name}),
        ("/reports/restock", {"branch_id": 1}),
    ]
    for path, params in calls:
        try:
            print(f"\n{path} {params} -> {show(client.get(path, **params))}")
        except TrackerError as error:
            print(f"\n{path} {params} -> ERROR {error}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())