"""Tests for app.tools.tracker_client with httpx.MockTransport. No tracker needed."""

import httpx
import pytest

from app.tools.tracker_client import (
    DEFAULT_BASE_URL,
    ENDPOINTS,
    ENV_BASE_URL,
    TrackerClient,
    TrackerConnectionError,
    TrackerHTTPError,
    TrackerResponseError,
)

BASE = "http://tracker.test"


def make(handler, seen=None):
    def wrapped(request):
        if seen is not None:
            seen.append(request)
        return handler(request)

    return TrackerClient(base_url=BASE, transport=httpx.MockTransport(wrapped))


def ok(body):
    return lambda request: httpx.Response(200, json=body)


def test_expiring_sends_a_get_with_its_parameters():
    seen = []
    client = make(ok([{"batch_number": "B050-3"}]), seen)

    assert client.expiring(branch_id=1, days=30) == [{"batch_number": "B050-3"}]
    (request,) = seen
    assert request.method == "GET"
    assert request.url.path == "/reports/expiring"
    assert dict(request.url.params) == {"branch_id": "1", "days": "30"}


@pytest.mark.parametrize(
    "method, kwargs, path, params",
    [
        ("health", {}, "/health", {}),
        ("branches", {}, "/branches", {}),
        ("low_stock", {"branch_id": 2}, "/reports/low-stock", {"branch_id": "2"}),
        ("monthly_sales", {"branch_name": "Manila Main Branch"}, "/reports/monthly-sales",
         {"branch_name": "Manila Main Branch"}),
        ("expiring", {"branch_id": 1, "days": 60}, "/reports/expiring", {"branch_id": "1", "days": "60"}),
        ("top_sellers", {"branch_name": "Cebu"}, "/reports/top-sellers", {"branch_name": "Cebu"}),
        ("restock", {"branch_id": 3}, "/reports/restock", {"branch_id": "3"}),
    ],
)
def test_each_method_calls_its_endpoint(method, kwargs, path, params):
    seen = []
    client = make(ok([]), seen)

    getattr(client, method)(**kwargs)

    (request,) = seen
    assert request.method == "GET" and request.url.path == path and dict(request.url.params) == params


def test_every_allowed_endpoint_has_a_method():
    paths = {"/health", "/branches", "/reports/low-stock", "/reports/monthly-sales", "/reports/expiring",
             "/reports/top-sellers", "/reports/restock"}
    assert set(ENDPOINTS) == paths


def test_none_parameters_are_left_out():
    seen = []
    client = make(ok([]), seen)

    client.expiring(branch_id=1)

    assert dict(seen[0].url.params) == {"branch_id": "1"}


def test_an_unknown_path_is_refused_without_a_request():
    seen = []
    client = make(ok([]), seen)

    with pytest.raises(ValueError, match="not an allowed"):
        client.get("/products")
    assert seen == []


def test_an_unknown_parameter_is_refused_without_a_request():
    seen = []
    client = make(ok([]), seen)

    with pytest.raises(ValueError, match="does not take"):
        client.get("/reports/expiring", branch_id=1, limit=5)
    assert seen == []


@pytest.mark.parametrize("verb", ["post", "put", "patch", "delete"])
def test_the_client_has_no_write_methods(verb):
    assert not hasattr(TrackerClient, verb)


def test_an_http_error_carries_the_status_and_detail():
    client = make(lambda r: httpx.Response(422, json={"detail": [{"msg": "field required"}]}))

    with pytest.raises(TrackerHTTPError) as caught:
        client.expiring(days=30)
    assert caught.value.status_code == 422
    assert caught.value.detail == [{"msg": "field required"}]
    assert "/reports/expiring" in str(caught.value)


def test_a_non_json_error_body_is_kept_as_text():
    client = make(lambda r: httpx.Response(500, text="Internal Server Error"))

    with pytest.raises(TrackerHTTPError, match="Internal Server Error"):
        client.branches()


def test_a_non_json_success_body_is_an_error():
    client = make(lambda r: httpx.Response(200, text="<html>"))

    with pytest.raises(TrackerResponseError):
        client.branches()


def test_a_refused_connection_is_a_connection_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TrackerConnectionError, match="could not reach"):
        make(refuse).health()


def test_a_timeout_is_a_connection_error():
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(TrackerConnectionError, match="in time"):
        make(slow).health()


def test_the_base_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_BASE_URL, "http://elsewhere:9000/")

    assert TrackerClient().base_url == "http://elsewhere:9000"


def test_the_default_base_url(monkeypatch):
    monkeypatch.delenv(ENV_BASE_URL, raising=False)

    assert TrackerClient().base_url == DEFAULT_BASE_URL