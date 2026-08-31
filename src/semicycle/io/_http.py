"""Small resilient HTTP helper shared by the loaders."""

from __future__ import annotations

import time
import urllib.request

import requests

_UA = "Mozilla/5.0 (semicycle research pipeline; +https://github.com/ryanappuhamy)"


def fetch_bytes(url: str, *, timeout: int = 60, retries: int = 3) -> bytes:
    """GET `url` and return the body. Tries `requests`, then stdlib `urllib`
    (some hosts behave differently with each client), retrying with backoff."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": _UA})
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001 - fall through to urllib
            last = exc
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
                return fh.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise ConnectionError(f"could not fetch {url}: {last}")
