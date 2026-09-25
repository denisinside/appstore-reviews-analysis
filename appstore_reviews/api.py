"""FastAPI REST API for App Store review collection and analysis."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .analysis_pipeline import run_full_pipeline
from .insights import run_saved_insights
from .keywords import aggregate_keywords
from .metrics import calculate_metrics
from .scraper import AppStoreReviews


DATA_ROOT = Path(os.environ.get("APPSTORE_SCAN_DIR", "scans"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)
_SCAN_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _scan_dir(scan_id: str) -> Path:
    if not _SCAN_ID_RE.fullmatch(scan_id):
        raise HTTPException(status_code=404, detail="Scan not found")
    folder = DATA_ROOT / scan_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Scan not found")
    return folder


def _load_meta(folder: Path) -> dict[str, Any]:
    meta = _read_json(folder / "scan.json")
    if not isinstance(meta, dict):
        raise HTTPException(status_code=404, detail="Scan metadata not found")
    return meta


def _save_meta(folder: Path, meta: dict[str, Any]) -> None:
    meta["updated_at"] = _utc_now()
    _write_json(folder / "scan.json", meta)


def _sentiment_summary(rows: list[dict]) -> dict[str, Any]:
    valid = [
        row for row in rows
        if isinstance(row, dict)
        and not row.get("error")
        and row.get("sentiment") in {"positive", "neutral", "negative"}
    ]
    counts = Counter(row["sentiment"] for row in valid)
    total = len(valid)
    return {
        "sample_size": total,
        "distribution": {
            label: {"count": counts[label], "share": counts[label] / total if total else None}
            for label in ("positive", "neutral", "negative")
        },
    }


class CollectRequest(BaseModel):
    app_id: str = Field(..., pattern=r"^[0-9]+$", description="Numeric Apple App Store application ID.")
    mode: Literal["country", "top"] = "top"
    country: str | None = Field(default=None, min_length=2, max_length=2)
    top_n: int = Field(default=10, ge=1, le=32)
    max_pages: int = Field(default=10, ge=1, le=10)
    force_refresh: bool = False


class AnalyzeRequest(BaseModel):
    batch_size: int = Field(default=8, ge=1, le=10)
    concurrency: int = Field(default=10, ge=1, le=10)
    requests_per_minute: int = Field(default=120, ge=1, le=120)
    max_cost_usd: float | None = Field(default=1.0, gt=0)
    run_local_nlp: bool = True


app = FastAPI(
    title="App Store Reviews Analysis API",
    version="1.0.0",
    description=(
        "Collect App Store reviews, run the NLP/LLM analysis pipeline, "
        "return metrics and actionable insights, and download raw review data."
    ),
)

origins = [
    item.strip()
    for item in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/apps/{app_id}/discovery")
def discover_app(app_id: str, force_refresh: bool = False) -> dict[str, Any]:
    try:
        with AppStoreReviews() as scraper:
            return scraper.discovery(app_id, force_refresh=force_refresh)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/scans", status_code=status.HTTP_201_CREATED)
def create_scan(payload: CollectRequest) -> dict[str, Any]:
    if payload.mode == "country" and not payload.country:
        raise HTTPException(status_code=422, detail="country is required when mode='country'")

    try:
        with AppStoreReviews() as scraper:
            if payload.mode == "country":
                collection = scraper.get_reviews(
                    payload.app_id,
                    payload.country,
                    max_pages=payload.max_pages,
                    force_refresh=payload.force_refresh,
                )
            else:
                collection = scraper.get_top_reviews(
                    payload.app_id,
                    top_n=payload.top_n,
                    max_pages=payload.max_pages,
                    force_refresh=payload.force_refresh,
                )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    reviews = collection.get("reviews") or []
    if not isinstance(reviews, list):
        raise HTTPException(status_code=500, detail="Collector returned invalid review data")

    scan_id = uuid4().hex[:12]
    folder = DATA_ROOT / scan_id
    folder.mkdir(parents=True, exist_ok=False)
    basic_metrics = calculate_metrics(reviews)

    meta = {
        "scan_id": scan_id,
        "app_id": payload.app_id,
        "collection_mode": payload.mode,
        "country": payload.country,
        "top_n": payload.top_n if payload.mode == "top" else None,
        "max_pages": payload.max_pages,
        "collection_status": collection.get("status"),
        "analysis_status": "not_started",
        "review_count": len(reviews),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "error": None,
    }

    _write_json(folder / "scan.json", meta)
    _write_json(folder / "collection.json", collection)
    _write_json(folder / "reviews.json", reviews)
    _write_json(folder / "basic_metrics.json", basic_metrics)

    return {
        **meta,
        "basic_metrics": basic_metrics,
        "collection_errors": collection.get("errors") or [],
    }


@app.get("/api/scans")
def list_scans() -> dict[str, Any]:
    items = []
    for folder in DATA_ROOT.iterdir():
        if not folder.is_dir():
            continue
        meta = _read_json(folder / "scan.json")
        if isinstance(meta, dict):
            items.append(meta)
    items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"items": items}


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: str) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    meta = _load_meta(folder)
    state = _read_json(folder / "scan_state.json", {})
    return {
        **meta,
        "api_usage": state.get("api_usage", {}) if isinstance(state, dict) else {},
    }


def _run_analysis_job(scan_id: str, payload: AnalyzeRequest) -> None:
    folder = DATA_ROOT / scan_id
    meta = _read_json(folder / "scan.json", {})
    try:
        meta["analysis_status"] = "running"
        meta["error"] = None
        _save_meta(folder, meta)

        reviews = _read_json(folder / "reviews.json", [])
        run_full_pipeline(
            reviews,
            folder,
            run_local_nlp=payload.run_local_nlp,
            batch_size=payload.batch_size,
            concurrency=payload.concurrency,
            requests_per_minute=payload.requests_per_minute,
            max_cost_usd=payload.max_cost_usd,
        )
        run_saved_insights(folder, max_cost_usd=payload.max_cost_usd)

        meta = _read_json(folder / "scan.json", meta)
        meta["analysis_status"] = "completed"
        meta["error"] = None
        _save_meta(folder, meta)
    except Exception as exc:
        meta = _read_json(folder / "scan.json", meta)
        meta["analysis_status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        _save_meta(folder, meta)


@app.post("/api/scans/{scan_id}/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze_scan(
    scan_id: str,
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    meta = _load_meta(folder)
    if meta.get("analysis_status") in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Analysis is already running")

    meta["analysis_status"] = "queued"
    meta["error"] = None
    _save_meta(folder, meta)
    background_tasks.add_task(_run_analysis_job, scan_id, payload)
    return {"scan_id": scan_id, "analysis_status": "queued"}


@app.get("/api/scans/{scan_id}/metrics")
def get_metrics(scan_id: str) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    basic = _read_json(folder / "basic_metrics.json")
    if basic is None:
        reviews = _read_json(folder / "reviews.json", [])
        basic = calculate_metrics(reviews)
        _write_json(folder / "basic_metrics.json", basic)

    sentiment_rows = _read_json(folder / "sentiment_results.json", [])
    keyword_rows = _read_json(folder / "keyword_results.json", [])
    nlp = _read_json(folder / "nlp_metrics.json")

    return {
        "basic": basic,
        "sentiment": _sentiment_summary(sentiment_rows),
        "common_keywords_by_language": aggregate_keywords(keyword_rows),
        "nlp": nlp,
    }


@app.get("/api/scans/{scan_id}/insights")
def get_insights(scan_id: str) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    insights = _read_json(folder / "insights.json")
    if insights is None:
        raise HTTPException(status_code=409, detail="Insights are not available yet")
    return insights


@app.get("/api/scans/{scan_id}/issues")
def get_issues(scan_id: str) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    return {
        "issues": _read_json(folder / "issue_catalog.json", []),
        "feature_requests": _read_json(folder / "feature_request_catalog.json", []),
    }


@app.get("/api/scans/{scan_id}/reviews")
def get_reviews(
    scan_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    rating: int | None = Query(default=None, ge=1, le=5),
    sentiment: Literal["positive", "neutral", "negative"] | None = None,
    language: str | None = None,
    country: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    reviews = _read_json(folder / "reviews.json", [])
    sentiment_rows = _read_json(folder / "sentiment_results.json", [])
    keyword_rows = _read_json(folder / "keyword_results.json", [])
    analyses = _read_json(folder / "review_analyses.json", {})

    sentiment_map = {
        str(row.get("review_id")): row
        for row in sentiment_rows
        if isinstance(row, dict) and row.get("review_id") is not None
    }
    keyword_map = {
        str(row.get("review_id")): row
        for row in keyword_rows
        if isinstance(row, dict) and row.get("review_id") is not None
    }

    enriched = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        rid = str(review.get("review_id", ""))
        sentiment_row = sentiment_map.get(rid, {})
        keyword_row = keyword_map.get(rid, {})
        item = {
            **review,
            "sentiment": sentiment_row.get("sentiment"),
            "sentiment_scores": sentiment_row.get("scores"),
            "keywords": keyword_row.get("keywords") or [],
            "issue_analysis": analyses.get(rid),
        }

        if rating is not None and item.get("rating") != rating:
            continue
        if sentiment is not None and item.get("sentiment") != sentiment:
            continue
        if language is not None and item.get("language") != language:
            continue
        if country is not None:
            observed = item.get("observed_countries") or []
            if item.get("country") != country and country not in observed:
                continue
        if q:
            haystack = " ".join(str(item.get(key) or "") for key in ("title", "text", "content")).casefold()
            if q.casefold() not in haystack:
                continue
        enriched.append(item)

    total = len(enriched)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": enriched[offset: offset + limit],
    }


@app.get("/api/scans/{scan_id}/reviews/download")
def download_reviews(scan_id: str) -> FileResponse:
    folder = _scan_dir(scan_id)
    path = folder / "reviews.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Raw review file not found")

    meta = _load_meta(folder)
    filename = f"app_{meta.get('app_id', 'unknown')}_{scan_id}_reviews.json"
    return FileResponse(path, media_type="application/json", filename=filename)
