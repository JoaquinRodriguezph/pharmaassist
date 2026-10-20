"""Tests for app.tools.registry and app.tools.executor. No tracker, no model."""

import json
from pathlib import Path

import httpx
import pytest

from app.tools.executor import MAX_ROWS, run_tool
from app.tools.registry import TOOLS, TOOLS_BY_NAME, ToolArgumentError, validate_args
from app.tools.tracker_client import ENDPOINTS, TrackerClient

REPO_ROOT = Path(__file__).resolve().parent.parent


def tracker(handler, seen=None):
    def wrapped(request):
        if seen is not None:
            seen.append(request)
        return handler(request)

    return TrackerClient(base_url="http://tracker.test", transport=httpx.MockTransport(wrapped))


def ok(body):
    return lambda request: httpx.Response(200, json=body)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_tool_names_are_unique():
    assert len(TOOLS_BY_NAME) == len(TOOLS)


def test_tool_names_are_the_test_set_names():
    assert set(TOOLS_BY_NAME) == {
        "get_branches", "get_low_stock", "get_monthly_sales", "get_expiring_batches", "get_top_sellers", "get_restock"}


@pytest.mark.parametrize("spec", TOOLS, ids=lambda s: s.name)
def test_every_tool_maps_to_an_allowed_endpoint_with_the_same_parameters(spec):
    assert spec.path in ENDPOINTS
    assert set(spec.parameters["properties"]) == set(ENDPOINTS[spec.path])
    assert spec.parameters["additionalProperties"] is False


def test_required_parameters_mirror_the_tracker_schema():
    assert TOOLS_BY_NAME["get_expiring_batches"].parameters["required"] == ["branch_id"]
    assert TOOLS_BY_NAME["get_restock"].parameters["required"] == ["branch_id"]
    for name in ("get_branches", "get_low_stock", "get_monthly_sales", "get_top_sellers"):
        assert TOOLS_BY_NAME[name].parameters["required"] == []


def test_json_schema_has_name_description_and_parameters():
    schema = TOOLS_BY_NAME["get_expiring_batches"].json_schema()
    assert set(schema) == {"name", "description", "parameters"}


def test_every_expected_tool_in_the_test_set_exists_and_its_args_validate():
    test_set = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
    problems = []
    for question in test_set:
        tools = question.get("expected_tool")
        if not tools:
            continue
        for tool in tools if isinstance(tools, list) else [tools]:
            if tool not in TOOLS_BY_NAME:
                problems.append((question["id"], tool, "unknown tool"))
                continue
            try:
                validate_args(tool, question.get("expected_args") or {})
            except ToolArgumentError as error:
                problems.append((question["id"], tool, str(error)))
    assert problems == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_valid_arguments_are_returned_cleaned():
    assert validate_args("get_expiring_batches", {"branch_id": 1, "days": 30}) == {"branch_id": 1, "days": 30}


@pytest.mark.parametrize("value, expected", [(1.0, 1), ("2", 2), (" 3 ", 3)])
def test_integers_accept_whole_floats_and_digit_strings(value, expected):
    assert validate_args("get_restock", {"branch_id": value}) == {"branch_id": expected}


def test_names_are_stripped():
    assert validate_args("get_top_sellers", {"branch_name": "  Manila Main Branch "}) == {
        "branch_name": "Manila Main Branch"}


def test_none_values_are_dropped():
    assert validate_args("get_low_stock", {"branch_id": None}) == {}


def test_no_arguments_is_an_empty_object():
    assert validate_args("get_branches", None) == {}


@pytest.mark.parametrize(
    "name, args, message",
    [
        ("get_inventory", {}, "unknown tool"),
        ("get_low_stock", [1], "must be an object"),
        ("get_low_stock", {"branch": 1}, "does not take"),
        ("get_expiring_batches", {"days": 30}, "requires"),
        ("get_restock", {"branch_id": True}, "must be an integer"),
        ("get_restock", {"branch_id": 1.5}, "must be an integer"),
        ("get_restock", {"branch_id": "one"}, "must be an integer"),
        ("get_restock", {"branch_id": 0}, "at least 1"),
        ("get_expiring_batches", {"branch_id": 1, "days": 366}, "at most 365"),
        ("get_top_sellers", {"branch_name": 5}, "must be a string"),
        ("get_top_sellers", {"branch_name": "   "}, "must not be empty"),
        ("get_top_sellers", {"branch_name": "x" * 101}, "at most 100"),
    ],
)
def test_invalid_arguments_are_refused(name, args, message):
    with pytest.raises(ToolArgumentError, match=message):
        validate_args(name, args)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def test_run_tool_calls_the_mapped_endpoint_with_cleaned_arguments():
    seen = []
    result = run_tool(tracker(ok([{"batch_number": "B050-3"}]), seen), "get_expiring_batches",
                      {"branch_id": "1", "days": 30})

    assert result.ok and result.args == {"branch_id": 1, "days": 30}
    assert result.data == [{"batch_number": "B050-3"}] and result.total_rows == 1 and not result.truncated
    assert seen[0].url.path == "/reports/expiring"
    assert dict(seen[0].url.params) == {"branch_id": "1", "days": "30"}


def test_long_lists_are_truncated_and_say_so():
    rows = [{"n": i} for i in range(MAX_ROWS + 5)]

    result = run_tool(tracker(ok(rows)), "get_low_stock", {})

    assert len(result.data) == MAX_ROWS and result.total_rows == MAX_ROWS + 5 and result.truncated


def test_invalid_arguments_make_no_request():
    seen = []

    result = run_tool(tracker(ok([]), seen), "get_restock", {})

    assert not result.ok and "requires" in result.error and seen == []


def test_an_unknown_tool_is_reported_not_raised():
    result = run_tool(tracker(ok([])), "delete_branch", {"branch_id": 1})

    assert not result.ok and "unknown tool" in result.error and result.args == {"branch_id": 1}


def test_a_tracker_http_error_is_reported():
    result = run_tool(tracker(lambda r: httpx.Response(422, json={"detail": "bad"})), "get_restock",
                      {"branch_id": 1})

    assert not result.ok and result.error == "tracker returned HTTP 422: bad"


def test_a_connection_error_is_reported():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    result = run_tool(tracker(refuse), "get_branches", {})

    assert not result.ok and "could not reach" in result.error


def test_for_model_shapes():
    rows = run_tool(tracker(ok([])), "get_low_stock", {})
    single = run_tool(tracker(ok({"status": "ok"})), "get_branches", {})
    failed = run_tool(tracker(ok([])), "get_restock", {})

    assert json.loads(rows.for_model()) == {"rows": [], "total_rows": 0, "truncated": False}
    assert json.loads(single.for_model()) == {"result": {"status": "ok"}}
    assert json.loads(failed.for_model()) == {"error": "get_restock requires ['branch_id']"}