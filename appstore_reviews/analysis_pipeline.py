"""One-off NLP scan: local models, LLM extraction, normalization, metrics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

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
                                batch_size: int = 12, concurrency: int = 48,
                                requests_per_minute: int = 480,
                                max_cost_usd: float | None = None,
                                client: OpenRouterClient | None = None,
                                progress_callback: Callable[[str, float, int | None, int | None], None] | None = None) -> dict:
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

    def report(stage: str, fraction: float, completed: int | None = None, total: int | None = None) -> None:
        if progress_callback:
            progress_callback(stage, fraction, completed, total)

    def local_nlp() -> tuple[list[dict], list[dict], str]:
        local_sentiments, local_keywords = sentiments, keywords
        local_digest = hashlib.sha256(json.dumps([
            {key: r.get(key) for key in ("review_id", "title", "text", "content", "language")}
            for r in reviews], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if state.get("local_input_hash") != local_digest:
            local_sentiments, local_keywords = [], []
        review_ids = {r.get("review_id") for r in reviews}
        if len(local_sentiments) != len(reviews) or {r.get("review_id") for r in local_sentiments} != review_ids:
            analyzer = sentiment_analyzer
            if analyzer is None:
                from .sentiment import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
            local_sentiments = analyzer.analyze_reviews(reviews)
            _write_json(folder / "sentiment_results.json", local_sentiments)
        report("sentiment", 1.0)
        negative_ids = {s.get("review_id") for s in local_sentiments
                        if s.get("sentiment") == "negative" and not s.get("error")}
        negative = [r for r in reviews if r.get("review_id") in negative_ids]
        if len(local_keywords) != len(negative) or {r.get("review_id") for r in local_keywords} != negative_ids:
            extractor = keyword_extractor
            if extractor is None:
                from .keywords import KeywordExtractor
                extractor = KeywordExtractor()
            local_keywords = extractor.extract_reviews(negative)
            _write_json(folder / "keyword_results.json", local_keywords)
        report("keywords", 1.0)
        return local_sentiments, local_keywords, local_digest

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

    current_stage: ContextVar[str] = ContextVar("analysis_stage", default="extraction")
    client.usage_callback = lambda: capture(current_stage.get())

    async def run() -> dict:
        async def local_branch() -> tuple[list[dict], list[dict]]:
            if not run_local_nlp:
                report("sentiment", 1.0)
                report("keywords", 1.0)
                return sentiments, keywords
            local_sentiments, local_keywords, digest = await asyncio.to_thread(local_nlp)
            state["local_input_hash"] = digest
            _write_json(state_path, state)
            return local_sentiments, local_keywords

        analyses = _read_json(folder / "review_analyses.json", {})
        if not isinstance(analyses, dict):
            raise ValueError("review_analyses.json must be an object keyed by review ID")

        processed = {rid for rid, row in analyses.items() if isinstance(row, dict) and row.get("status") == "success"}
        report("issue_extraction", len(processed) / max(1, len(reviews)), len(processed), len(reviews))
        updates_since_save = 0

        def on_update(row: dict) -> None:
            nonlocal updates_since_save
            analyses[row["review_id"]] = row
            processed.add(row["review_id"])
            updates_since_save += 1
            if updates_since_save >= batch_size:
                _write_json(folder / "review_analyses.json", analyses)
                updates_since_save = 0
            try:
                report("issue_extraction", len(processed) / max(1, len(reviews)), len(processed), len(reviews))
            except Exception:
                _write_json(folder / "review_analyses.json", analyses)
                raise

        extractor = IssueAspectExtractor(client, batch_size=batch_size)
        extraction, local_result = await asyncio.gather(
            extractor.extract_reviews(reviews, existing=analyses, on_update=on_update),
            local_branch(),
            return_exceptions=True,
        )
        if isinstance(extraction, BaseException):
            _write_json(folder / "review_analyses.json", analyses)
            raise extraction
        analyses = extraction
        _write_json(folder / "review_analyses.json", analyses)
        if isinstance(local_result, BaseException):
            raise local_result
        sentiments, keywords = local_result
        report("issue_extraction", 1.0, len(reviews), len(reviews))

        async def normalize(kind: str, stage: str) -> tuple[dict, list[dict]]:
            token = current_stage.set(stage)
            report(stage, 0.0)
            try:
                result = await normalize_signals(kind, analyses, client, folder / "normalization_cache")
                report(stage, 1.0)
                return result
            finally:
                current_stage.reset(token)

        (issue_analyses, issues), (feature_analyses, features) = await asyncio.gather(
            normalize("issues", "issue_normalization"),
            normalize("feature_requests", "feature_normalization"),
        )
        # Both normalizers copy the same extraction results and only modify their
        # respective signal field. Merge those fields before saving one artifact.
        analyses = issue_analyses
        for rid, result in analyses.items():
            other_aspects = feature_analyses.get(rid, {}).get("aspects", [])
            for aspect, other in zip(result.get("aspects", []), other_aspects):
                aspect["feature_requests"] = other.get("feature_requests", [])
        _write_json(folder / "review_analyses.json", analyses)
        _write_json(folder / "issue_catalog.json", issues)
        _write_json(folder / "feature_request_catalog.json", features)
        report("metrics", 0.0)
        metrics = recalculate_saved_metrics(folder)
        report("metrics", 1.0)
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
