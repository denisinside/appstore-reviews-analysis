"""Persistent JSON cache with optional Upstash REST storage."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)


class FileCache:
    def __init__(self, directory: str | Path = ".cache", clock=time.time):
        self.directory = Path(directory)
        self.clock = clock

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str):
        path = self._path(key)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope["expires_at"] <= self.clock():
                path.unlink(missing_ok=True)
                return None
            return envelope["value"]
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("Could not read cache entry %s: %s", key, exc)
            return None

    def set(self, key: str, value, ttl: int) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            temp = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
            temp.write_text(
                json.dumps({"expires_at": self.clock() + ttl, "value": value}, ensure_ascii=False),
                encoding="utf-8",
            )
            temp.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning("Could not write cache entry %s: %s", key, exc)


class UpstashCache:
    def __init__(self, url: str, token: str):
        self.client = httpx.Client(
            base_url=url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )

    def _command(self, *parts):
        response = self.client.post("", json=list(parts))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("error"):
            raise ValueError(f"Upstash error: {payload.get('error') if isinstance(payload, dict) else payload}")
        return payload.get("result")

    def get(self, key: str):
        result = self._command("GET", key)
        return json.loads(result) if result is not None else None

    def set(self, key: str, value, ttl: int) -> None:
        self._command("SET", key, json.dumps(value, ensure_ascii=False), "EX", ttl)


class FallbackCache:
    """Switch to the file cache after an Upstash failure in this process."""

    def __init__(self, primary: UpstashCache, fallback: FileCache):
        self.primary = primary
        self.fallback = fallback

    def _use(self, method: str, *args):
        if self.primary is not None:
            try:
                return getattr(self.primary, method)(*args)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                LOGGER.warning("Upstash unavailable; using local cache: %s", exc)
                self.primary = None
        return getattr(self.fallback, method)(*args)

    def get(self, key: str):
        return self._use("get", key)

    def set(self, key: str, value, ttl: int) -> None:
        self._use("set", key, value, ttl)


def make_cache(directory: str | Path = ".cache"):
    fallback = FileCache(directory)
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        try:
            return FallbackCache(UpstashCache(url, token), fallback)
        except ValueError as exc:
            LOGGER.warning("Invalid Upstash configuration; using local cache: %s", exc)
            return fallback
    if url or token:
        LOGGER.warning("Both Upstash environment variables are required; using local cache")
    return fallback
