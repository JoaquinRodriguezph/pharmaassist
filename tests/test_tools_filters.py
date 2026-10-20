"""Filters applied by the executor (ToolSpec.filters)."""

import httpx
import pytest

from app.tools.executor import run_tool
from app.tools.registry import TOOLS_BY_NAME, ToolArgumentError, validate_args
from app.tools.tracker_client import TrackerClient


def tracker(rows, seen):
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=rows)

    return TrackerClient(base_url="http://t.test", transport=httpx.MockTransport(handler))


def test_the_model_sees_the_filter_but_the_endpoint_parameters_do_not_include_it():
    spec = TOOLS_BY_NAME["get_low_stock"]

    assert "product" in spec.json_schema()["parameters"]["properties"]
    assert "product" not in spec.parameters["properties"]


def test_validation_accepts_the_filter():
    assert validate_args("get_low_stock", {"product": " Paracetamol ", "branch_id": 2}) == {
        "product": "Paracetamol", "branch_id": 2}


def test_a_filter_on_a_tool_without_it_is_refused():
    with pytest.raises(ToolArgumentError, match="does not take"):
        validate_args("get_restock", {"branch_id": 1, "product": "x"})


def test_rows_are_filtered_case_insensitively_and_the_filter_is_not_sent():
    seen = []
    rows = [{"product_name": "Paracetamol 500mg"}, {"product_name": "PARACETAMOL Syrup"}, {"product_name": "Ibuprofen"}]

    result = run_tool(tracker(rows, seen), "get_low_stock", {"product": "paracetamol", "branch_id": 1})

    assert dict(seen[0].url.params) == {"branch_id": "1"}
    assert [r["product_name"] for r in result.data] == ["Paracetamol 500mg", "PARACETAMOL Syrup"]
    assert result.total_rows == 2 and result.args == {"product": "paracetamol", "branch_id": 1}


def test_no_filter_keeps_every_row():
    seen = []

    result = run_tool(tracker([{"product_name": "A"}, {"product_name": "B"}], seen), "get_low_stock", {})

    assert result.total_rows == 2