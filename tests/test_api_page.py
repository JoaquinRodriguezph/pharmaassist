"""The chat page is served at /."""

from fastapi.testclient import TestClient

from app.api.main import create_app


class FakeAgent:
    def run(self, message, history):
        raise AssertionError("not called")


def test_the_chat_page_is_served():
    with TestClient(create_app(agent_factory=FakeAgent)) as client:
        response = client.get("/")

    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert "PharmaAssist" in response.text and 'fetch("/chat"' in response.text
    assert "textContent" in response.text and "innerHTML" not in response.text  # model text is never parsed as HTML