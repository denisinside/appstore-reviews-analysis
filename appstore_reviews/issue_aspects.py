"""Grounded aspect, issue and feature-request extraction from original reviews."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping

from .config import ASPECT_CATEGORIES, ASPECT_TAXONOMY_VERSION
from .openrouter_client import MODEL_ID, OpenRouterClient, OpenRouterError

TAXONOMY_VERSION = ASPECT_TAXONOMY_VERSION
CATEGORIES = ASPECT_CATEGORIES
SCHEMA_VERSION = "1.0"

SIGNAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"description": {"type": "string", "minLength": 1},
                   "evidence": {"type": "string", "minLength": 1}},
    "required": ["description", "evidence"],
}
ASPECT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "aspect": {"type": "string", "minLength": 1},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "evidence": {"type": "string", "minLength": 1},
        "issues": {"type": "array", "items": SIGNAL_SCHEMA},
        "feature_requests": {"type": "array", "items": SIGNAL_SCHEMA},
    },
    "required": ["category", "aspect", "sentiment", "evidence", "issues", "feature_requests"],
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"review_id": {"type": "string", "minLength": 1},
                   "aspects": {"type": "array", "items": ASPECT_SCHEMA}},
    "required": ["review_id", "aspects"],
}
EXTRACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"results": {"type": "array", "items": REVIEW_SCHEMA}},
    "required": ["results"],
}

SYSTEM_PROMPT = """Analyze each App Store review independently, using only its title and text. Return JSON matching the schema.
For every mentioned app aspect, assign exactly one fixed category and a concise specific aspect. Aspect sentiment is about that aspect, never inferred from star rating or overall sentiment. A review can include positive and negative aspects, several issues, and several feature requests.
An issue is a concrete user-observed problem or obstacle. General dislike without a concrete problem is not an issue. A feature request is a request for a new capability or improvement; it need not be negative. Do not assume technical root causes, infer absent facts, or turn every feature mention into a problem.
For each aspect, issue, and feature request quote a contiguous fragment from the ORIGINAL title or text as evidence. Include negation, numbers, and restrictions needed to understand the statement. Preserve distinct issues but avoid duplicates within a review. Write concise descriptions in English without broad labels. Do not use any category outside the supplied enum. Return every review ID exactly once."""


def _source_text(review: Mapping[str, object]) -> tuple[str, str]:
    title = review.get("title")
    body = review.get("text") if review.get("text") not in (None, "") else review.get("content")
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be text or null")
    if body is not None and not isinstance(body, str):
        raise ValueError("text must be text or null")
    return title or "", body or ""


def _text_hash(title: str, body: str) -> str:
    return hashlib.sha256(json.dumps([title, body], ensure_ascii=False).encode("utf-8")).hexdigest()


def _grounded(evidence: str, title: str, body: str) -> bool:
    if any(evidence in part for part in (title, body)):
        return True
    # Whitespace and Unicode composition can differ after LLM tokenization;
    # lexical content must still be an exact contiguous normalized span.
    norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()
    needle = norm(evidence)
    return bool(needle) and any(needle in norm(part) for part in (title, body))


def _dedup_key(description: str) -> str:
    return re.sub(r"\s+", " ", description.casefold()).strip(" .,!?:;\t\n")


def validate_review_result(result: dict, review: Mapping[str, object]) -> dict:
    """Keep original evidence and flag unsupported claims without inventing spans."""
    title, body = _source_text(review)
    if result["review_id"] != review["review_id"]:
        raise ValueError("review ID mismatch")
    aspects = []
    seen_signals: dict[str, set[tuple[str, str]]] = {"issues": set(), "feature_requests": set()}
    for aspect in result["aspects"]:
        item = {**aspect, "evidence_verified": _grounded(aspect["evidence"], title, body)}
        for kind in ("issues", "feature_requests"):
            signals = []
            for signal in aspect[kind]:
                key = (aspect["category"], _dedup_key(signal["description"]))
                if key in seen_signals[kind]:
                    continue
                seen_signals[kind].add(key)
                signals.append({**signal, "evidence_verified": _grounded(signal["evidence"], title, body),
                                "canonical_id": None})
            item[kind] = signals
        aspects.append(item)
    return {"review_id": result["review_id"], "status": "success", "error": None,
            "input_hash": _text_hash(title, body), "original_title": title, "original_text": body,
            "taxonomy_version": TAXONOMY_VERSION, "schema_version": SCHEMA_VERSION,
            "model": MODEL_ID, "aspects": aspects}


class IssueAspectExtractor:
    def __init__(self, client: OpenRouterClient, *, batch_size: int = 8) -> None:
        if not 1 <= batch_size <= 10:
            raise ValueError("batch_size must be 1–10")
        self.client = client
        self.batch_size = batch_size

    async def extract_reviews(self, reviews: list[Mapping[str, object]], *,
                              existing: Mapping[str, dict] | None = None,
                              on_update: Callable[[dict], None] | None = None) -> dict[str, dict]:
        if not isinstance(reviews, list):
            raise TypeError("reviews must be a list")
        saved = dict(existing or {})
        seen: set[str] = set()
        pending: list[Mapping[str, object]] = []
        for review in reviews:
            if not isinstance(review, Mapping):
                raise ValueError("reviews must contain dictionaries")
            rid = review.get("review_id")
            if not isinstance(rid, str) or not rid or rid in seen:
                raise ValueError("review IDs must be unique nonempty strings")
            seen.add(rid)
            try:
                title, body = _source_text(review)
                if not (title.strip() or body.strip()):
                    raise ValueError("empty review text")
            except ValueError as exc:
                row = {"review_id": rid, "status": "error", "error": str(exc), "aspects": []}
                saved[rid] = row
                if on_update:
                    on_update(row)
                continue
            if (saved.get(rid, {}).get("status") == "success"
                    and saved[rid].get("input_hash") == _text_hash(title, body)
                    and saved[rid].get("model") == MODEL_ID
                    and saved[rid].get("schema_version") == SCHEMA_VERSION
                    and saved[rid].get("taxonomy_version") == TAXONOMY_VERSION):
                continue
            pending.append(review)

        async def one_batch(batch: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
            payload = {"reviews": [{"review_id": row["review_id"], "title": _source_text(row)[0],
                                    "text": _source_text(row)[1]} for row in batch]}
            try:
                response = await self.client.json_completion(schema_name="review_aspects_v1",
                    schema=EXTRACTION_SCHEMA, system=SYSTEM_PROMPT,
                    user=json.dumps(payload, ensure_ascii=False))
                expected = {row["review_id"] for row in batch}
                returned: set[str] = set()
                for result in response["results"]:
                    rid = result["review_id"]
                    if rid not in expected or rid in returned:
                        raise ValueError("unknown or duplicate review ID in model response")
                    returned.add(rid)
                validated = []
                for result in response["results"]:
                    rid = result["review_id"]
                    review = next(row for row in batch if row["review_id"] == rid)
                    validated.append(validate_review_result(result, review))
                for row in validated:
                    rid = row["review_id"]
                    saved[rid] = row
                    if on_update:
                        on_update(row)
                return [row for row in batch if row["review_id"] not in returned]
            except (OpenRouterError, ValueError, KeyError) as exc:
                for review in batch:
                    rid = review["review_id"]
                    row = {"review_id": rid, "status": "error", "error": str(exc), "aspects": []}
                    saved[rid] = row
                    if on_update:
                        on_update(row)
                return []

        batches = [pending[i:i+self.batch_size] for i in range(0, len(pending), self.batch_size)]
        missing = [row for part in await asyncio.gather(*(one_batch(batch) for batch in batches)) for row in part]
        # Retry only IDs omitted by otherwise valid batch responses.
        for review in missing:
            for _ in range(2):
                if not await one_batch([review]):
                    break
            else:
                rid = review["review_id"]
                row = {"review_id": rid, "status": "error", "error": "review ID omitted by model", "aspects": []}
                saved[rid] = row
                if on_update:
                    on_update(row)
        return {rid: saved[rid] for rid in seen if rid in saved}
