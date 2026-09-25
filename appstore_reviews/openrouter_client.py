"""Small asynchronous OpenRouter JSON client for one-off scans."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from jsonschema import Draft202012Validator

MODEL_ID = "z-ai/glm-5.3-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROVIDER_ORDER = "parasail,baseten,together"
EXTRACTION_INPUT_LIMIT = 25_000
NORMALIZATION_INPUT_LIMIT = 100_000


def estimated_input_tokens(system: str, user: str) -> int:
    """Conservative, dependency-free estimate including message overhead."""
    return (len(system.encode("utf-8")) + len(user.encode("utf-8")) + 1) // 2 + 256


def split_input_records(records: list[dict], *, key: str, system: str,
                        limit: int, max_records: int) -> list[list[dict]]:
    """Keep whole records while bounding each serialized request payload."""
    chunks: list[list[dict]] = []
    chunk: list[dict] = []
    for record in records:
        candidate = [*chunk, record]
        user = json.dumps({key: candidate}, ensure_ascii=False)
        if len(candidate) > max_records or estimated_input_tokens(system, user) > limit:
            if chunk:
                chunks.append(chunk)
                candidate = [record]
                user = json.dumps({key: candidate}, ensure_ascii=False)
            if estimated_input_tokens(system, user) > limit:
                raise ValueError(f"single {key} record exceeds the {limit}-token input limit")
        chunk = candidate
    if chunk:
        chunks.append(chunk)
    return chunks


def _provider_preferences() -> list[str] | None:
    """Return configured provider order, or None to use OpenRouter routing."""
    raw_order = os.environ.get("OPENROUTER_PROVIDER_ORDER", DEFAULT_PROVIDER_ORDER)
    providers = [provider.strip() for provider in raw_order.split(",") if provider.strip()]
    return providers or None


class OpenRouterError(RuntimeError):
    """A request failed after retries or returned unusable data."""


class CostLimitError(OpenRouterError):
    """The configured test budget has been reached."""


@dataclass
class APIUsage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_reported_requests: int = 0

    def as_dict(self) -> dict:
        return vars(self).copy()


class OpenRouterClient:
    def __init__(self, *, api_key: str | None = None, concurrency: int = 48,
                 requests_per_minute: int = 480, timeout: float = 90,
                 max_retries: int = 3, max_cost_usd: float | None = None,
                 prior_cost_usd: float = 0.0,
                 transport: httpx.AsyncBaseTransport | None = None,
                 usage_callback: Callable[[], None] | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        if concurrency < 1 or requests_per_minute < 1 or timeout <= 0 or max_retries < 0:
            raise ValueError("invalid OpenRouter request limits")
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate_lock = asyncio.Lock()
        self.budget_lock = asyncio.Lock()
        self.request_times: list[float] = []
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.max_cost_usd = max_cost_usd
        self.prior_cost_usd = prior_cost_usd
        self.usage = APIUsage()
        self.usage_callback = usage_callback
        self.provider_order = _provider_preferences()
        self.http = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.http.aclose()

    async def _rate_limit(self) -> None:
        async with self.rate_lock:
            now = time.monotonic()
            self.request_times = [t for t in self.request_times if now - t < 60]
            if len(self.request_times) >= self.requests_per_minute:
                await asyncio.sleep(max(0, 60 - (now - self.request_times[0])))
                now = time.monotonic()
                self.request_times = [t for t in self.request_times if now - t < 60]
            self.request_times.append(now)

    async def _check_budget(self) -> None:
        async with self.budget_lock:
            if self.max_cost_usd is not None and self.prior_cost_usd + self.usage.cost_usd >= self.max_cost_usd:
                raise CostLimitError("OpenRouter cost limit reached")

    def _add_usage(self, body: dict) -> None:
        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            return
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(field)
            if isinstance(value, int) and value >= 0:
                setattr(self.usage, field, getattr(self.usage, field) + value)
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and cost >= 0:
            self.usage.cost_usd += float(cost)
            self.usage.cost_reported_requests += 1
        if self.usage_callback:
            self.usage_callback()

    async def json_completion(self, *, schema_name: str, schema: dict,
                              system: str, user: str, max_tokens: int = 25_000,
                              reasoning_max_tokens: int = 20_000) -> dict[str, Any]:
        """Return locally schema-validated JSON; retry transient or invalid replies.

        When a provider rejects strict schema routing, one compatible JSON-object
        request is attempted. Local schema validation applies to both routes.
        """
        validator = Draft202012Validator(schema)
        if max_tokens < 1 or reasoning_max_tokens < 1 or reasoning_max_tokens > max_tokens:
            raise ValueError("invalid OpenRouter token limits")
        input_limit = EXTRACTION_INPUT_LIMIT if max_tokens == 25_000 else NORMALIZATION_INPUT_LIMIT
        if estimated_input_tokens(system, user) > input_limit:
            raise ValueError(f"OpenRouter input exceeds the {input_limit}-token input limit")
        strict = True
        reasoning_enabled = True
        last_error = "unknown error"
        for attempt in range(self.max_retries + 3):
            await self._check_budget()
            provider: dict[str, Any] = {"require_parameters": True}
            if self.provider_order:
                provider.update({"order": self.provider_order, "allow_fallbacks": False})
            payload: dict[str, Any] = {
                "model": MODEL_ID,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "provider": provider,
                "response_format": (
                    {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
                    if strict else {"type": "json_object"}
                ),
                "usage": {"include": True},
                "max_tokens": max_tokens,
            }
            if reasoning_enabled:
                payload["reasoning"] = {"max_tokens": reasoning_max_tokens}
            async with self.semaphore:
                await self._rate_limit()
                self.usage.requests += 1
                if self.usage_callback:
                    self.usage_callback()
                try:
                    response = await self.http.post(URL, json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"})
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = type(exc).__name__
                    response = None
            if response is None:
                pass
            elif response.status_code in (400, 422) and reasoning_enabled and "reasoning" in response.text.lower():
                # Some providers reject this optional parameter; max_tokens remains mandatory.
                reasoning_enabled = False
                continue
            elif response.status_code in (400, 404, 422) and strict:
                # Only a parameter/schema rejection is eligible for JSON mode.
                message = response.text.lower()[:1000]
                if any(term in message for term in ("response_format", "json_schema", "strict", "require_parameters")):
                    strict = False
                    continue
                raise OpenRouterError(f"OpenRouter HTTP {response.status_code}: {response.text[:300]}")
            elif response.status_code in (429, 500, 502, 503, 504):
                last_error = f"OpenRouter HTTP {response.status_code}"
                retry_after = response.headers.get("retry-after")
                try:
                    delay = min(60, max(0.0, float(retry_after))) if retry_after else None
                except ValueError:
                    delay = None
            elif response.status_code >= 400:
                raise OpenRouterError(f"OpenRouter HTTP {response.status_code}: {response.text[:300]}")
            else:
                try:
                    body = response.json()
                    self._add_usage(body)
                    content = body["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    errors = list(validator.iter_errors(parsed))
                    if errors:
                        raise ValueError(errors[0].message)
                    return parsed
                except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    last_error = f"invalid JSON response: {exc}"
            if attempt >= self.max_retries + (0 if strict else 1):
                break
            if response is None or response.status_code not in (429, 500, 502, 503, 504):
                delay = None
            await asyncio.sleep(delay if delay is not None else min(15.0, 0.6 * 2 ** attempt + random.random() * 0.2))
        raise OpenRouterError(last_error)
