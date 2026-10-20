"""The refusal sentences are owned by code: prompt contents, model-added sentences, aliases, JSON lines, leaks."""

from types import SimpleNamespace

import httpx

from app.agent import AGENT_SYSTEM_PROMPT, MEDICAL_REFUSAL, SCOPE_REFUSAL, SEARCH_TOOL_NAME, Agent
from app.llm.tools import ToolCall, ToolTurn
from app.rag.citations import REFUSAL_SENTENCE
from app.tools.registry import TOOLS
from app.tools.tracker_client import TrackerClient

POLICY = "returns_and_expiry_policy.md#Customer-Returns"
KNOWN = {POLICY}
BRANCHES = {1: "Manila Main Branch", 2: "Quezon City Hub", 3: "Calamba Crossing Branch", 4: "Lucena City Center"}


class FakeSession:
    def __init__(self, turns, kwargs):
        self.turns = list(turns)
        self.kwargs = kwargs

    def step(self, allow_tools=True):
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]

    def add_tool_results(self, results):
        pass

    def add_user_message(self, text):
        pass


def turn(text="", calls=()):
    return ToolTurn(text=text, tool_calls=tuple(calls), prompt_tokens=10, completion_tokens=5, latency_seconds=0.1)


def chunk(section_id):
    return SimpleNamespace(id=f"{section_id}::1", text="Rule text.", section_id=section_id,
                           section=section_id.split("#")[1].replace("-", " "))


def make_agent(turns, kb=None, rows=None, seen=None, prefetch=True):
    sessions = []

    def factory(client, **kwargs):
        sessions.append(FakeSession(turns, kwargs))
        return sessions[-1]

    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=rows if rows is not None else [])

    agent = Agent(llm_client="fake", tracker=TrackerClient(base_url="http://t.test", transport=httpx.MockTransport(handler)),
                  kb_search=lambda q: kb or [], known_section_ids=KNOWN, branches=BRANCHES, prefetch=prefetch,
                  session_factory=factory)
    return agent, sessions


def test_the_prompt_names_every_tool_and_the_not_covered_sentence_only():
    for name in [SEARCH_TOOL_NAME] + [t.name for t in TOOLS]:
        assert name in AGENT_SYSTEM_PROMPT
    assert REFUSAL_SENTENCE in AGENT_SYSTEM_PROMPT
    assert MEDICAL_REFUSAL not in AGENT_SYSTEM_PROMPT and SCOPE_REFUSAL not in AGENT_SYSTEM_PROMPT
    assert POLICY not in AGENT_SYSTEM_PROMPT


def test_an_off_topic_request_is_refused_without_the_model():
    agent, sessions = make_agent([turn("never")])

    result = agent.run("can you write a python script to scrape Mercury Drug's website?")

    assert sessions == [] and result.scope_refusal and result.blocked_by.startswith("input_guard:")


def test_a_scope_refusal_from_the_model_becomes_the_not_covered_sentence():
    agent, _ = make_agent([turn(SCOPE_REFUSAL)])

    result = agent.run("do we accept GCash when the POS is offline?")

    assert result.text == REFUSAL_SENTENCE and result.blocked_by == "scope_from_model"


def test_a_medical_sentence_on_a_non_medical_question_is_removed():
    reply = f"{MEDICAL_REFUSAL}\n\nPrescription medicines cannot be returned [{POLICY}]."
    agent, _ = make_agent([turn(reply)], kb=[chunk(POLICY)])

    result = agent.run("customer wants to return unopened amoxicillin with a receipt, can we refund?")

    assert result.text == f"Prescription medicines cannot be returned [{POLICY}]."
    assert result.blocked_by == "medical_stripped" and not result.medical_refusal


def test_only_a_medical_sentence_on_a_non_medical_question_becomes_not_covered():
    agent, _ = make_agent([turn(MEDICAL_REFUSAL)])

    result = agent.run("what's the employee discount?")

    assert result.text == REFUSAL_SENTENCE and result.blocked_by == "medical_from_model"


def test_tool_call_json_lines_are_removed():
    reply = 'Atorvastatin is low at Manila Main Branch.\n{"name": "get_low_stock", "parameters": {"branch_id": "1"}}'
    agent, _ = make_agent([turn(calls=[ToolCall("get_low_stock", {})]), turn(reply)],
                          rows=[{"product_name": "Atorvastatin 20mg Tablet"}], prefetch=False)

    result = agent.run("what is low?")

    assert result.text == "Atorvastatin is low at Manila Main Branch."


def test_a_copied_prompt_rule_is_a_leak():
    rule = "For a question about several or all branches, make one call without a branch."
    agent, _ = make_agent([turn(f"{rule}\n\nOK.")], prefetch=False)

    result = agent.run("which branches?")

    assert result.blocked_by == "output_leak"


def test_a_branch_id_given_to_a_name_tool_is_translated():
    seen = []
    agent, _ = make_agent([turn(calls=[ToolCall("get_monthly_sales", {"branch_id": "4"})]), turn("Sales listed.")],
                          seen=seen, prefetch=False)

    result = agent.run("monthly sales for Lucena City Center")

    assert dict(seen[0].url.params) == {"branch_name": "Lucena City Center"}
    assert result.tool_calls[0].ok and result.tool_calls[0].arguments == {"branch_name": "Lucena City Center"}


def test_a_digit_branch_name_is_translated():
    seen = []
    agent, _ = make_agent([turn(calls=[ToolCall("get_top_sellers", {"branch_name": "1"})]), turn("Top sellers.")],
                          seen=seen, prefetch=False)

    agent.run("top sellers at Manila")

    assert dict(seen[0].url.params) == {"branch_name": "Manila Main Branch"}


def test_a_branch_name_given_to_an_id_tool_is_translated():
    seen = []
    agent, _ = make_agent([turn(calls=[ToolCall("get_restock", {"branch_name": "calamba crossing branch"})]),
                           turn("Restock list.")], seen=seen, prefetch=False)

    agent.run("restock for Calamba")

    assert dict(seen[0].url.params) == {"branch_id": "3"}


def test_the_product_filter_is_applied_by_the_agent_not_sent_to_the_tracker():
    seen = []
    rows = [{"product_name": "Paracetamol 500mg Tablet"}, {"product_name": "Atorvastatin 20mg Tablet"}]
    agent, _ = make_agent([turn(calls=[ToolCall("get_low_stock", {"product": "paracetamol"})]),
                           turn("Paracetamol is low.")], rows=rows, seen=seen, prefetch=False)

    result = agent.run("which branches are low on paracetamol?")

    assert dict(seen[0].url.params) == {}
    assert result.tool_calls[0].rows == 1 and result.tool_calls[0].arguments == {"product": "paracetamol"}