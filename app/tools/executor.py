"""Run one tool call against the tracker and package the result for the model.

`run_tool` never raises for a bad tool call: unknown tools, invalid arguments and tracker
errors come back as a ToolResult with ok=False, so the model can be told what went wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.tools.registry import TOOLS_BY_NAME, ToolArgumentError, validate_args
from app.tools.tracker_client import TrackerClient, TrackerError, TrackerHTTPError

MAX_ROWS = 50


@dataclass(frozen=True)
class ToolResult:
    name: str
    args: dict = field(default_factory=dict)
    ok: bool = False
    data: Any = None
    error: str | None = None
    total_rows: int | None = None
    truncated: bool = False

    def for_model(self) -> str:
        """The JSON text given back to the model as the tool's output."""
        if not self.ok:
            payload = {"error": self.error}
        elif isinstance(self.data, list):
            payload = {"rows": self.data, "total_rows": self.total_rows, "truncated": self.truncated}
        else:
            payload = {"result": self.data}
        return json.dumps(payload, ensure_ascii=False)


def run_tool(client: TrackerClient, name: str, args: Any) -> ToolResult:
    raw = dict(args) if isinstance(args, dict) else {}
    try:
        cleaned = validate_args(name, args)
    except ToolArgumentError as error:
        return ToolResult(name, raw, ok=False, error=str(error))
    spec = TOOLS_BY_NAME[name]
    try:
        data = client.get(spec.path, **cleaned)
    except TrackerHTTPError as error:
        return ToolResult(name, cleaned, ok=False, error=f"tracker returned HTTP {error.status_code}: {error.detail}")
    except TrackerError as error:
        return ToolResult(name, cleaned, ok=False, error=str(error))
    if isinstance(data, list):
        total = len(data)
        return ToolResult(name, cleaned, ok=True, data=data[:MAX_ROWS], total_rows=total, truncated=total > MAX_ROWS)
    return ToolResult(name, cleaned, ok=True, data=data)