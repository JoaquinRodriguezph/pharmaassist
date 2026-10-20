"""The citation repair turn runs only for answers that rest on the documents."""

from types import SimpleNamespace

import httpx

from app.agent import SEARCH_TOOL_NAME, Agent
from app.llm.tools import ToolCall, ToolTurn
from app.tools.tracker_client import TrackerClient

POLICY = "returns_and_expiry_policy.md#Expired-Medications"


class FakeSession:
    def __init__(self, turns):
        self.turns = list(turns)
        self.user_messages = []

    def step(self, allow_tools=True):
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]

    def add_tool_results(self, results):
        pass

    def add_user_message(self, text):
        self.user_messages.append(text)


def turn(text="", calls=()):
    return ToolTurn(text=text, tool_calls=tuple(calls), prompt_tokens=10, completion_tokens=5)


def run(turns, question="q", rows=None):
    sessions = []

    def factory(client, **kwargs):
        sessions.append(FakeSession(turns))
        return sessions[-1]

    tracker = TrackerClient(base_url="http://t.test",
                            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=rows or [{"n": 1}])))
    kb = [SimpleNamespace(id=f"{POLICY}::1", text="Rule.", section_id=POLICY, section="Expired Medications")]
    agent = Agent(llm_client="fake", tracker=tracker, kb_search=lambda q: kb, known_section_ids={POLICY},
                  branches={2: "Quezon City Hub"}, prefetch=True, session_factory=factory)
    return agent.run(question), sessions[0]


def test_tracker_data_with_only_prefetched_sources_is_not_repaired():
    reply = "Amoxicillin 500mg (B001-2) expires on 2026-11-13."
    result, session = run([turn(calls=[ToolCall("get_expiring_batches", {"branch_id": 2, "days": 30})]), turn(reply)])

    assert not result.repair_attempted and session.user_messages == [] and result.text == reply


def test_a_document_answer_without_tracker_data_is_repaired():
    result, session = run([turn("Quarantine them."), turn(f"Quarantine them [{POLICY}].")])

    assert result.repair_attempted and result.repair_applied and len(session.user_messages) == 1


def test_a_mixed_answer_where_the_model_searched_is_repaired():
    result, _ = run([
        turn(calls=[ToolCall("get_expiring_batches", {"branch_id": 2, "days": 30}),
                    ToolCall(SEARCH_TOOL_NAME, {"query": "expiry policy"})]),
        turn("Four batches expire. Quarantine them."),
        turn(f"Four batches expire. Quarantine them [{POLICY}]."),
    ])

    assert result.repair_attempted and result.repair_applied