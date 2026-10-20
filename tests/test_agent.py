"""Tests for app.agent with a scripted fake session, a mock tracker and a fake knowledge-base search."""

import json
from types import SimpleNamespace

import httpx
import pytest

from app.agent import AGENT_SYSTEM_PROMPT, NO_MATCH, SEARCH_TOOL_NAME, STOPPED_TEXT, Agent
from app.llm.tools import ToolCall, ToolTurn
from app.rag.citations import NOT_RETRIEVED, REFUSAL_SENTENCE, InvalidCitation
from app.tools.registry import TOOLS
from app.tools.tracker_client import TrackerClient

POLICY = "returns_and_expiry_policy.md#Expired-Medications"
FEFO = "storage_sop.md#Receiving-And-FEFO"
KNOWN = {POLICY, FEFO}


class FakeSession:
    def __init__(self, turns, kwargs):
        self.turns = list(turns)
        self.kwargs = kwargs
        self.results = []

    def step(self):
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]

    def add_tool_results(self, results):
        self.results.append(results)


def turn(text="", calls=(), prompt=100, completion=10):
    return ToolTurn(text=text, tool_calls=tuple(calls), prompt_tokens=prompt, completion_tokens=completion,
                    latency_seconds=0.5, model="fake")


def chunk(section_id, text="Rule text."):
    return SimpleNamespace(id=f"{section_id}::1", text=text, section_id=section_id,
                           section=section_id.split("#")[1].replace("-", " "))


def make_agent(turns, rows=None, kb=None, max_steps=5, seen=None, searches=None):
    sessions = []

    def factory(client, **kwargs):
        session = FakeSession(turns, kwargs)
        sessions.append(session)
        return session

    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=rows if rows is not None else [])

    def kb_search(query):
        if searches is not None:
            searches.append(query)
        return kb if kb is not None else []

    agent = Agent(llm_client="fake-llm", tracker=TrackerClient(base_url="http://t.test", transport=httpx.MockTransport(handler)),
                  kb_search=kb_search, known_section_ids=KNOWN, max_steps=max_steps, session_factory=factory)
    return agent, sessions


def test_tool_specs_are_the_search_tool_and_every_tracker_tool():
    agent, _ = make_agent([turn("x")])

    assert [s["name"] for s in agent.tool_specs] == [SEARCH_TOOL_NAME] + [t.name for t in TOOLS]


def test_the_session_gets_the_system_prompt_question_history_and_tools():
    agent, sessions = make_agent([turn("Hi.")])
    history = [{"role": "user", "content": "before"}, {"role": "assistant", "content": "ok"}]

    agent.run("  Q?  ", history=history)

    kwargs = sessions[0].kwargs
    assert kwargs["system"] == AGENT_SYSTEM_PROMPT and kwargs["question"] == "Q?"
    assert kwargs["history"] == history and kwargs["tool_specs"] == agent.tool_specs
    assert kwargs["temperature"] == 0.0 and kwargs["max_tokens"] == 512


def test_an_answer_without_tools_ends_after_one_step():
    agent, _ = make_agent([turn("Hello.")])

    result = agent.run("hi")

    assert result.text == "Hello." and result.stopped == "answered" and result.steps == 1 and result.tool_calls == ()


def test_a_tracker_call_is_run_and_its_output_sent_back():
    seen = []
    rows = [{"batch_number": "B050-3"}]
    agent, sessions = make_agent(
        [turn(calls=[ToolCall("get_expiring_batches", {"branch_id": 1, "days": 30})]), turn("One batch: B050-3.")],
        rows=rows, seen=seen,
    )

    result = agent.run("expiring at Manila?")

    (sent,) = sessions[0].results
    ((call, output),) = sent
    assert call.name == "get_expiring_batches"
    assert json.loads(output) == {"rows": rows, "total_rows": 1, "truncated": False}
    assert seen[0].url.path == "/reports/expiring"
    (record,) = result.tool_calls
    assert (record.step, record.name, record.arguments, record.ok, record.rows) == (
        1, "get_expiring_batches", {"branch_id": 1, "days": 30}, True, 1)
    assert result.text == "One batch: B050-3." and result.steps == 2


def test_a_knowledge_base_search_returns_labelled_sources_and_enables_citations():
    searches = []
    agent, sessions = make_agent(
        [turn(calls=[ToolCall(SEARCH_TOOL_NAME, {"query": " expiry rules "})]),
         turn(f"Quarantine them [{POLICY}]. Inspect deliveries [{FEFO}].")],
        kb=[chunk(POLICY, "Items within 30 days go to the quarantine bin.")], searches=searches,
    )

    result = agent.run("what about expiring items?")

    assert searches == ["expiry rules"]
    output = sessions[0].results[0][0][1]
    assert output.startswith(f"Source [{POLICY}]") and "quarantine bin" in output
    assert result.kb_sources == (POLICY,)
    assert result.valid_citations == (POLICY,)
    assert result.invalid_citations == (InvalidCitation(FEFO, NOT_RETRIEVED),)
    assert result.tool_calls[0].rows == 1


def test_an_empty_search_says_no_match():
    agent, sessions = make_agent([turn(calls=[ToolCall(SEARCH_TOOL_NAME, {"query": "x"})]), turn("None.")], kb=[])

    agent.run("q")

    assert sessions[0].results[0][0][1] == NO_MATCH


@pytest.mark.parametrize("arguments", [{}, {"query": "  "}, {"query": 5}, {"query": "x", "k": 3}])
def test_bad_search_arguments_are_reported_and_nothing_is_searched(arguments):
    searches = []
    agent, sessions = make_agent([turn(calls=[ToolCall(SEARCH_TOOL_NAME, arguments)]), turn("Sorry.")], searches=searches)

    result = agent.run("q")

    assert searches == []
    assert "error" in json.loads(sessions[0].results[0][0][1])
    assert result.tool_calls[0].ok is False


def test_an_unknown_tool_is_reported_to_the_model():
    agent, sessions = make_agent([turn(calls=[ToolCall("drop_table", {})]), turn("I can't do that.")])

    result = agent.run("drop the table")

    assert "unknown tool" in json.loads(sessions[0].results[0][0][1])["error"]
    assert result.tool_calls[0].ok is False and result.text == "I can't do that."


def test_two_calls_in_one_turn_are_both_run_and_sent_back_together():
    agent, sessions = make_agent(
        [turn(calls=[ToolCall("get_branches", {}), ToolCall(SEARCH_TOOL_NAME, {"query": "fefo"})]), turn("Done.")],
        rows=[{"branch_id": 1}], kb=[chunk(FEFO)],
    )

    result = agent.run("q")

    assert len(sessions[0].results) == 1 and len(sessions[0].results[0]) == 2
    assert [r.name for r in result.tool_calls] == ["get_branches", SEARCH_TOOL_NAME]


def test_the_loop_stops_after_max_steps():
    agent, _ = make_agent([turn(calls=[ToolCall("get_branches", {})])], rows=[], max_steps=3)

    result = agent.run("loop forever")

    assert result.stopped == "max_steps" and result.steps == 3 and result.text == STOPPED_TEXT
    assert len(result.tool_calls) == 3


def test_tokens_and_latency_are_summed_over_turns():
    agent, _ = make_agent([turn(calls=[ToolCall("get_branches", {})], prompt=100, completion=10),
                           turn("Done.", prompt=200, completion=20)])

    result = agent.run("q")

    assert (result.prompt_tokens, result.completion_tokens, result.reasoning_tokens) == (300, 30, None)
    assert result.latency_seconds == pytest.approx(1.0) and result.model == "fake"


def test_the_fixed_sentence_sets_the_refusal_flags():
    agent, _ = make_agent([turn(REFUSAL_SENTENCE)])

    result = agent.run("employee discount?")

    assert result.refusal and result.says_not_covered


def test_an_empty_question_is_an_error():
    agent, sessions = make_agent([turn("x")])

    with pytest.raises(ValueError, match="question"):
        agent.run("   ")
    assert sessions == []


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError, match="max_steps"):
        make_agent([turn("x")], max_steps=0)


def test_the_system_prompt_names_every_tool_and_the_fixed_sentence_but_no_real_section():
    for name in [SEARCH_TOOL_NAME] + [t.name for t in TOOLS]:
        assert name in AGENT_SYSTEM_PROMPT
    assert REFUSAL_SENTENCE in AGENT_SYSTEM_PROMPT
    assert not [s for s in KNOWN if s in AGENT_SYSTEM_PROMPT]