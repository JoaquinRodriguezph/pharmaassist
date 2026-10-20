"""'null' strings and the unknown-argument hint in app.tools.registry.validate_args."""

import pytest

from app.tools.registry import ToolArgumentError, validate_args


@pytest.mark.parametrize("value", ["null", "None", " NULL "])
def test_null_strings_count_as_missing(value):
    assert validate_args("get_low_stock", {"branch_id": value}) == {}


def test_a_null_string_for_a_required_argument_is_missing():
    with pytest.raises(ToolArgumentError, match="requires"):
        validate_args("get_restock", {"branch_id": "null"})


def test_an_unknown_argument_error_lists_the_valid_arguments():
    with pytest.raises(ToolArgumentError,
                       match=r"does not take \['query'\]; its arguments are \['branch_id', 'product'\]"):
        validate_args("get_low_stock", {"query": "paracetamol", "branch_id": "null"})


def test_a_tool_without_arguments_says_none():
    with pytest.raises(ToolArgumentError, match="its arguments are none"):
        validate_args("get_branches", {"x": 1})