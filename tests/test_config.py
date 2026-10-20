"""Tests for app.rag.config (the RAG_WITH_CONTEXT setting)."""

import pytest

from app.rag.config import DEFAULT_WITH_CONTEXT, ENV_NAME, rag_with_context, resolve_with_context


def test_the_default_is_no_prefix_which_is_the_tie_break_chosen_before_the_run():
    assert DEFAULT_WITH_CONTEXT is False
    assert rag_with_context({}) is False


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_value_means_the_default(value):
    assert rag_with_context({ENV_NAME: value}) is DEFAULT_WITH_CONTEXT


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", " on "])
def test_true_values(value):
    assert rag_with_context({ENV_NAME: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "False", "NO", " off "])
def test_false_values(value):
    assert rag_with_context({ENV_NAME: value}) is False


def test_a_typo_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="RAG_WITH_CONTEXT='ture'"):
        rag_with_context({ENV_NAME: "ture"})


def test_it_reads_the_real_environment_when_none_is_passed(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "true")
    assert rag_with_context() is True
    monkeypatch.delenv(ENV_NAME)
    assert rag_with_context() is False


def test_a_command_line_choice_beats_the_environment():
    assert resolve_with_context(False, {ENV_NAME: "true"}) is False
    assert resolve_with_context(True, {ENV_NAME: "false"}) is True
    assert resolve_with_context(None, {ENV_NAME: "true"}) is True
    assert resolve_with_context(None, {}) is False