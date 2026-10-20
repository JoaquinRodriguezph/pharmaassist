"""The read-only tools the assistant may call: names, descriptions, argument schemas, validation.

Tool names match `expected_tool` in eval/test_set.json. Each tool maps to one allow-listed GET
endpoint of the tracker (app/tools/tracker_client.py). Parameter names, types and required
flags mirror the tracker's OpenAPI schema. The bounds (branch_id >= 1, 1 <= days <= 365,
names up to 100 characters) are this project's choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_DAYS = 365
MAX_NAME_LENGTH = 100
_NULL_STRINGS = {"null", "none"}


class ToolArgumentError(ValueError):
    """The tool name or its arguments are not valid."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    path: str
    parameters: dict  # JSON Schema of type "object"

    def json_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _object(properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


_BRANCH_ID = {
    "type": "integer",
    "minimum": 1,
    "description": "Branch id from get_branches, for example 1 for Manila Main Branch.",
}
_BRANCH_NAME = {
    "type": "string",
    "description": 'Exact branch name from get_branches, for example "Manila Main Branch".',
}
_DAYS = {
    "type": "integer",
    "minimum": 1,
    "maximum": MAX_DAYS,
    "description": "Look-ahead window in days. The tracker uses 60 if omitted.",
}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "get_branches",
        "List every pharmacy branch with its branch_id, name, city and address. "
        "Use it to find a branch_id or the exact branch name.",
        "/branches",
        _object({}),
    ),
    ToolSpec(
        "get_low_stock",
        "Stock batches that are low compared with their reorder level, with branch, product, supplier, "
        "batch number, quantity and reorder level. Optional branch_id limits it to one branch; "
        "omit it to see every branch.",
        "/reports/low-stock",
        _object({"branch_id": _BRANCH_ID}),
    ),
    ToolSpec(
        "get_monthly_sales",
        "Sales per month: number of sales, total revenue and average sale value. "
        "Optional branch_name limits it to one branch.",
        "/reports/monthly-sales",
        _object({"branch_name": _BRANCH_NAME}),
    ),
    ToolSpec(
        "get_expiring_batches",
        "Batches at one branch that expire within the given number of days, with product, "
        "batch number, quantity, expiry date and days left.",
        "/reports/expiring",
        _object({"branch_id": _BRANCH_ID, "days": _DAYS}, required=("branch_id",)),
    ),
    ToolSpec(
        "get_top_sellers",
        "Top-selling products by units sold, with revenue and rank within the branch. "
        "Optional branch_name limits it to one branch.",
        "/reports/top-sellers",
        _object({"branch_name": _BRANCH_NAME}),
    ),
    ToolSpec(
        "get_restock",
        "Restock recommendations for one branch: products below their reorder level, with supplier, "
        "current quantity, reorder level and a suggested order quantity.",
        "/reports/restock",
        _object({"branch_id": _BRANCH_ID}, required=("branch_id",)),
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {tool.name: tool for tool in TOOLS}


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().casefold() in _NULL_STRINGS)


def _integer(key: str, value: Any, schema: dict) -> int:
    if isinstance(value, bool):
        raise ToolArgumentError(f"{key} must be an integer, got {value!r}")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int):
        raise ToolArgumentError(f"{key} must be an integer, got {value!r}")
    low, high = schema.get("minimum"), schema.get("maximum")
    if low is not None and value < low:
        raise ToolArgumentError(f"{key} must be at least {low}, got {value}")
    if high is not None and value > high:
        raise ToolArgumentError(f"{key} must be at most {high}, got {value}")
    return value


def _string(key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ToolArgumentError(f"{key} must be a string, got {value!r}")
    value = value.strip()
    if not value:
        raise ToolArgumentError(f"{key} must not be empty")
    if len(value) > MAX_NAME_LENGTH:
        raise ToolArgumentError(f"{key} must be at most {MAX_NAME_LENGTH} characters")
    return value


def validate_args(name: str, args: Any) -> dict:
    """Return cleaned arguments for tool `name`, or raise ToolArgumentError.

    None and the strings "null" / "none" count as missing and are dropped. Integers also
    accept whole floats and digit strings, because models sometimes emit 1.0 or "1";
    booleans are refused.
    """
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        raise ToolArgumentError(f"unknown tool {name!r}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolArgumentError(f"arguments for {name} must be an object, got {type(args).__name__}")
    properties = spec.parameters["properties"]
    unknown = sorted(set(args) - set(properties))
    if unknown:
        valid = sorted(properties) or "none"
        raise ToolArgumentError(f"{name} does not take {unknown}; its arguments are {valid}")
    args = {key: (None if _is_null(value) else value) for key, value in args.items()}
    missing = [key for key in spec.parameters["required"] if args.get(key) is None]
    if missing:
        raise ToolArgumentError(f"{name} requires {missing}")
    cleaned = {}
    for key, value in args.items():
        if value is None:
            continue
        schema = properties[key]
        cleaned[key] = _integer(key, value, schema) if schema["type"] == "integer" else _string(key, value)
    return cleaned