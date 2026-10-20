"""Read-only client for the Pharmacy Inventory & Sales Tracker API.

Only the GET endpoints in ENDPOINTS can be called, and only with their listed query
parameters. The client has no way to send POST, PUT, PATCH or DELETE. A parameter
passed as None is left out of the query, so the tracker applies its own default or
returns 422 if the parameter is required.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
ENV_BASE_URL = "TRACKER_BASE_URL"
DEFAULT_TIMEOUT = 5.0

ENDPOINTS: dict[str, frozenset[str]] = {
    "/health": frozenset(),
    "/branches": frozenset(),
    "/reports/low-stock": frozenset({"branch_id"}),
    "/reports/monthly-sales": frozenset({"branch_name"}),
    "/reports/expiring": frozenset({"branch_id", "days"}),
    "/reports/top-sellers": frozenset({"branch_name"}),
    "/reports/restock": frozenset({"branch_id"}),
}


class TrackerError(Exception):
    """Base class for tracker client errors."""


class TrackerConnectionError(TrackerError):
    """The tracker could not be reached, or did not answer in time."""


class TrackerHTTPError(TrackerError):
    """The tracker answered with an HTTP error status."""

    def __init__(self, path: str, status_code: int, detail: Any) -> None:
        self.path = path
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{path} returned HTTP {status_code}: {detail}")


class TrackerResponseError(TrackerError):
    """The tracker answered 2xx, but the body is not JSON."""


def _detail(response: httpx.Response) -> Any:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "detail" in body:
        return body["detail"]
    return body


class TrackerClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "TrackerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, path: str, **params: Any) -> Any:
        """GET an allow-listed path with allow-listed query parameters; return the parsed JSON."""
        if path not in ENDPOINTS:
            raise ValueError(f"{path!r} is not an allowed tracker endpoint")
        unknown = sorted(set(params) - ENDPOINTS[path])
        if unknown:
            raise ValueError(f"{path} does not take parameter(s) {unknown}")
        query = {name: value for name, value in params.items() if value is not None}
        try:
            response = self._http.get(path, params=query)
        except httpx.TimeoutException as error:
            raise TrackerConnectionError(f"the tracker at {self.base_url} did not answer {path} in time") from error
        except httpx.TransportError as error:
            raise TrackerConnectionError(f"could not reach the tracker at {self.base_url}: {error}") from error
        if response.status_code >= 400:
            raise TrackerHTTPError(path, response.status_code, _detail(response))
        try:
            return response.json()
        except ValueError as error:
            raise TrackerResponseError(f"{path} did not return JSON") from error

    def health(self) -> Any:
        return self.get("/health")

    def branches(self) -> Any:
        return self.get("/branches")

    def low_stock(self, branch_id: int | None = None) -> Any:
        return self.get("/reports/low-stock", branch_id=branch_id)

    def monthly_sales(self, branch_name: str | None = None) -> Any:
        return self.get("/reports/monthly-sales", branch_name=branch_name)

    def expiring(self, branch_id: int | None = None, days: int | None = None) -> Any:
        return self.get("/reports/expiring", branch_id=branch_id, days=days)

    def top_sellers(self, branch_name: str | None = None) -> Any:
        return self.get("/reports/top-sellers", branch_name=branch_name)

    def restock(self, branch_id: int | None = None) -> Any:
        return self.get("/reports/restock", branch_id=branch_id)