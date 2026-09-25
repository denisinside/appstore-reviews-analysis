"""One-off NLP scan: local models, LLM extraction, normalization, metrics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .issue_aspects import IssueAspectExtractor
from .nlp_metrics import calculate_nlp_metrics
from .normalization import normalize_signals
from .openrouter_client import OpenRouterClient


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def load_reviews(path: str | Path) -> list[dict]:
    """Read collector JSON, a JSON array, or saved JSONL reviews."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        reviews = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        reviews = parsed.get("reviews") if isinstance(parsed, dict) else parsed
    if not isinstance(reviews, list):
        raise ValueError("input must be a JSON review list or collector result with reviews")
    return reviews


def recalculate_saved_metrics(output_dir: str | Path) -> dict:
    """Recompute all deterministic NLP metrics without an API client."""
    folder = Path(output_dir)
    reviews = _read_json(folder / "reviews.json", None)
    if reviews is None:
        raise FileNotFoundError("reviews.json is missing from scan directory")
    metrics = calculate_nlp_metrics(
        reviews, _read_json(folder / "review_analyses.json", {}),
        _read_json(folder / "issue_catalog.json", []),
        _read_json(folder / "feature_request_catalog.json", []),
        _read_json(folder / "sentiment_results.json", []),
    )
    _write_json(folder / "nlp_metrics.json", metrics)
    return metrics


async def analyze_full_pipeline(reviews: list[dict], output_dir: str | Path, *,
                                run_local_nlp: bool = True,
                                sentiment_analyzer=None, keyword_extractor=None,
                                batch_size: int = 8, concurrency: int = 2,
                                requests_per_minute: int = 30,
                                max_cost_usd: float | None = None,
                                client: OpenRouterClient | None = None) -> dict:
    """Run or resume a scan; all LLM results persist as JSON after each batch.

    The optional local stages retain the established sentiment->negative-keywords
    workflow. LLM extraction always sees original text from every valid review.
    """
    if not isinstance(reviews, list):
        raise TypeError("reviews must be a list")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / "reviews.json", reviews)
    state_path = folder / "scan_state.json"
    state = _read_json(state_path, {"api_usage": {}})
    state.setdefault("api_usage", {})
    for stage in ("extraction", "issue_normalization", "feature_normalization"):
        state["api_usage"].setdefault(stage, {"requests": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
            "cost_reported_requests": 0})

    sentiments = _read_json(folder / "sentiment_results.json", [])
    keywords = _read_json(folder / "keyword_results.json", [])
    if run_local_nlp:
        local_digest = hashlib.sha256(json.dumps([
            {key: r.get(key) for key in ("review_id", "title", "text", "content", "language")}
            for r in reviews], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if state.get("local_input_hash") != local_digest:
            sentiments, keywords = [], []
        review_ids = {r.get("review_id") for r in reviews}
        if len(sentiments) != len(reviews) or {r.get("review_id") for r in sentiments} != review_ids:
            if sentiment_analyzer is None:
                from .sentiment import SentimentAnalyzer
                sentiment_analyzer = SentimentAnalyzer()
            sentiments = sentiment_analyzer.analyze_reviews(reviews)
            _write_json(folder / "sentiment_results.json", sentiments)
        negative = [r for r in reviews if any(s.get("review_id") == r.get("review_id") and
                    s.get("sentiment") == "negative" and not s.get("error") for s in sentiments)]
        if len(keywords) != len(negative) or {r.get("review_id") for r in keywords} != {r.get("review_id") for r in negative}:
            if keyword_extractor is None:
                from .keywords import KeywordExtractor
                keyword_extractor = KeywordExtractor()
            keywords = keyword_extractor.extract_reviews(negative)
            _write_json(folder / "keyword_results.json", keywords)
        state["local_input_hash"] = local_digest
        _write_json(state_path, state)

    own_client = client is None
    prior_cost = sum(float(v.get("cost_usd", 0)) for v in state["api_usage"].values())
    if own_client:
        client = OpenRouterClient(concurrency=concurrency, requests_per_minute=requests_per_minute,
                                  max_cost_usd=max_cost_usd, prior_cost_usd=prior_cost)
    assert client is not None
    recorded = client.usage.as_dict()

    def capture(stage: str) -> None:
        now = client.usage.as_dict()
        for field, value in now.items():
            state["api_usage"][stage][field] += value - recorded[field]
            recorded[field] = value
        _write_json(state_path, state)

    current_stage = ["extraction"]
    client.usage_callback = lambda: capture(current_stage[0])

    async def run() -> dict:
        analyses = _read_json(folder / "review_analyses.json", {})
        if not isinstance(analyses, dict):
            raise ValueError("review_analyses.json must be an object keyed by review ID")

        def on_update(row: dict) -> None:
            analyses[row["review_id"]] = row
            _write_json(folder / "review_analyses.json", analyses)
            capture("extraction")

        extractor = IssueAspectExtractor(client, batch_size=batch_size)
        analyses = await extractor.extract_reviews(reviews, existing=analyses, on_update=on_update)
        _write_json(folder / "review_analyses.json", analyses)
        capture("extraction")
        current_stage[0] = "issue_normalization"
        analyses, issues = await normalize_signals("issues", analyses, client,
            folder / "normalization_cache")
        _write_json(folder / "review_analyses.json", analyses)
        _write_json(folder / "issue_catalog.json", issues)
        capture("issue_normalization")
        current_stage[0] = "feature_normalization"
        analyses, features = await normalize_signals("feature_requests", analyses, client,
            folder / "normalization_cache")
        _write_json(folder / "review_analyses.json", analyses)
        _write_json(folder / "feature_request_catalog.json", features)
        capture("feature_normalization")
        metrics = recalculate_saved_metrics(folder)
        return {"review_analyses": analyses, "issue_catalog": issues,
                "feature_request_catalog": features, "nlp_metrics": metrics,
                "api_usage": state["api_usage"], "sentiment_results": sentiments,
                "keyword_results": keywords}

    if own_client:
        async with client:
            return await run()
    return await run()


def run_full_pipeline(*args, **kwargs) -> dict:
    """Synchronous wrapper for scripts and the package CLI."""
    return asyncio.run(analyze_full_pipeline(*args, **kwargs))
