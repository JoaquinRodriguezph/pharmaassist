"""Pre-retrieval, medical mode, output cleaning and the new output guards in app.agent."""

from types import SimpleNamespace

import httpx

from app.agent import MEDICAL_REFUSAL, SEARCH_TOOL_NAME, Agent
from app.llm.tools import ToolCall, ToolTurn
from app.rag.citations import REFUSAL_SENTENCE
from app.tools.tracker_client import TrackerClient

POLICY = "returns_and_expiry_policy.md#Expired-Medications"
IBU = "ibuprofen_leaflet.md#Drug-Interactions"
KNOWN = {POLICY, IBU}
BRANCHES = {1: "Manila Main Branch", 2: "Quezon City Hub"}


class FakeSession:
    def __init__(self, turns, kwargs):
        self.turns = list(turns)
        self.kwargs = kwargs
        self.allow_tools = []

    def step(self, allow_tools=True):
        self.allow_tools.append(allow_tools)
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]

    def add_tool_results(self, results):
        pass

    def add_user_message(self, text):
        pass


def turn(text="", calls=(), prompt=100):
    return ToolTurn(text=text, tool_calls=tuple(calls), prompt_tokens=prompt, completion_tokens=10,
                    latency_seconds=0.1, model="fake")


def chunk(section_id, text="Rule text."):
    return SimpleNamespace(id=f"{section_id}::1", text=text, section_id=section_id,
                           section=section_id.split("#")[1].replace("-", " "))


def make_agent(turns, kb=None, rows=None, prefetch=True, searches=None, context_window=None):
    sessions = []

    def factory(client, **kwargs):
        session = FakeSession(turns, kwargs)
        sessions.append(session)
        return session

    def kb_search(query):
        if searches is not None:
            searches.append(query)
        return kb or []

    tracker = TrackerClient(base_url="http://t.test",
                            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=rows or [])))
    agent = Agent(llm_client="fake", tracker=tracker, kb_search=kb_search, known_section_ids=KNOWN,
                  branches=BRANCHES, prefetch=prefetch, context_window=context_window, session_factory=factory)
    return agent, sessions


# ---------------------------------------------------------------------------
# Pre-retrieval and context window
# ---------------------------------------------------------------------------

def test_prefetched_sources_are_given_with_the_question_and_can_be_cited():
    searches = []
    agent, sessions = make_agent([turn(f"Quarantine them [{POLICY}].")], kb=[chunk(POLICY)], searches=searches)

    result = agent.run("  where do expiring items go?  ")

    assert searches == ["where do expiring items go?"]
    message = sessions[0].kwargs["question"]
    assert message.startswith("Sources found for this question")
    assert f"Source [{POLICY}]" in message and message.endswith("Question: where do expiring items go?")
    assert result.kb_sources == (POLICY,) and result.valid_citations == (POLICY,)
    assert not result.repair_attempted and result.tool_calls == ()


def test_no_prefetched_sources_leaves_the_question_alone():
    agent, sessions = make_agent([turn("Hi.")], kb=[])

    agent.run("hello")

    assert sessions[0].kwargs["question"] == "hello"


def test_prefetch_off_does_not_search():
    searches = []
    agent, _ = make_agent([turn("Hi.")], kb=[chunk(POLICY)], prefetch=False, searches=searches)

    agent.run("hello")

    assert searches == []


def test_the_context_window_is_passed_to_the_session_and_the_largest_prompt_is_reported():
    agent, sessions = make_agent([turn(calls=[ToolCall("get_branches", {})], prompt=900), turn("Done.", prompt=1400)],
                                 context_window=8192)

    result = agent.run("list branches")

    assert sessions[0].kwargs["num_ctx"] == 8192
    assert result.max_prompt_tokens == 1400 and result.prompt_tokens == 2300


# ---------------------------------------------------------------------------
# Medical mode
# ---------------------------------------------------------------------------

def test_a_personal_dose_question_gets_only_the_medical_refusal():
    agent, _ = make_agent([turn("Give her 7.5 mL of the syrup now.")])

    result = agent.run("my 4yo daughter has a fever, how many mL of paracetamol syrup should i give her?")

    assert result.medical_mode and result.text == MEDICAL_REFUSAL
    assert result.medical_refusal and result.blocked_by == "medical_filter"


def test_medical_mode_keeps_only_cited_leaflet_sentences():
    reply = (f"I can't give personal medical advice.\nThe leaflet says ibuprofen may make blood pressure "
             f"medicines work less well [{IBU}]. Try a cold compress on your forehead.")
    agent, _ = make_agent([turn(reply)], kb=[chunk(IBU)])

    result = agent.run("Can I take ibuprofen together with my blood pressure medicine?")

    assert result.text == (f"{MEDICAL_REFUSAL}\n\nThe leaflet says ibuprofen may make blood pressure medicines "
                           f"work less well [{IBU}].")
    assert result.valid_citations == (IBU,) and result.medical_refusal


def test_an_already_correct_medical_answer_is_not_marked_as_filtered():
    agent, _ = make_agent([turn(MEDICAL_REFUSAL)])

    result = agent.run("should I tell them to switch to azithromycin?")

    assert result.text == MEDICAL_REFUSAL and result.blocked_by is None


def test_a_general_leaflet_question_is_not_medical_mode():
    agent, _ = make_agent([turn(f"Do not take it with other NSAIDs [{IBU}].")], kb=[chunk(IBU)])

    result = agent.run("which patients should NOT take ibuprofen according to our leaflet?")

    assert not result.medical_mode and result.blocked_by is None


# ---------------------------------------------------------------------------
# Output cleaning and guards
# ---------------------------------------------------------------------------

def test_echoed_source_label_lines_are_removed():
    reply = f"Quarantine them [{POLICY}].\n\nSource [{POLICY}] (Expired Medications)\n"
    agent, _ = make_agent([turn(reply)], kb=[chunk(POLICY)])

    result = agent.run("where do expiring items go?")

    assert result.text == f"Quarantine them [{POLICY}]."


def test_a_made_up_reference_without_md_is_blocked():
    agent, _ = make_agent([turn("Pull items 7 days before expiry [Expired Inventory Procedure#Expired Inventory Procedure].")])

    result = agent.run("how many days before expiry do we pull items?")

    assert result.blocked_by == "unverified_citation" and result.text == REFUSAL_SENTENCE


def test_a_chunk_id_of_a_real_section_is_not_treated_as_made_up():
    agent, _ = make_agent([turn(f"Quarantine them [{POLICY}::1].")], kb=[chunk(POLICY)])

    result = agent.run("where do expiring items go?")

    assert result.blocked_by != "unverified_citation"


def test_an_early_not_stated_hedge_becomes_the_not_covered_sentence():
    reply = f"The employee discount is not explicitly stated in the documents. However, returns are allowed [{POLICY}]."
    agent, _ = make_agent([turn(reply)], kb=[chunk(POLICY)])

    result = agent.run("what's our employee discount?")

    assert result.blocked_by == "not_covered_hedge" and result.refusal


def test_a_late_hedge_does_not_replace_a_real_answer():
    reply = (f"Quarantine them [{POLICY}]. Log the batch number [{POLICY}]. "
             f"The SOP does not specify who does the logging.")
    agent, _ = make_agent([turn(reply)], kb=[chunk(POLICY)])

    result = agent.run("what do we do with expiring items?")

    assert result.blocked_by is None and result.text == reply


def test_a_hedge_next_to_tracker_data_is_kept():
    agent, _ = make_agent(
        [turn(calls=[ToolCall("get_low_stock", {})]), turn("The data does not mention paracetamol. Atorvastatin is low.")],
        rows=[{"product_name": "Atorvastatin 20mg Tablet"}], prefetch=False,
    )

    result = agent.run("which branches are low on paracetamol?")

    assert result.blocked_by is None


def test_search_tool_name_is_unchanged():
    assert SEARCH_TOOL_NAME == "search_knowledge_base"