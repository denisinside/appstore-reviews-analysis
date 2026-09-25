"""FastAPI REST API for App Store review collection and analysis."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
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
from .config import COUNTRY_NAMES, MAX_RSS_PAGES, SUPPORTED_COUNTRIES
from .insights import run_saved_insights
from .keywords import aggregate_keywords
from .metrics import calculate_metrics
from .scraper import AppStoreReviews
from . import scan_runtime


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
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


def _scan_dir(scan_id: str) -> Path:
    if not _SCAN_ID_RE.fullmatch(scan_id):
        raise HTTPException(status_code=404, detail="Scan not found")
    scan_runtime.reload()
    folder = DATA_ROOT / scan_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Scan not found")
    return folder


def _load_meta(folder: Path) -> dict[str, Any]:
    meta = _read_json(folder / "scan.json")
    if not isinstance(meta, dict):
        raise HTTPException(status_code=404, detail="Scan metadata not found")
    return meta


class AnalysisInterrupted(Exception):
    """A scan stop was requested from the API."""


def _check_stop_requested(folder: Path) -> None:
    scan_runtime.reload()
    if (folder / "stop.requested").exists():
        raise AnalysisInterrupted()


def _available_results(folder: Path) -> dict[str, Any]:
    """Describe saved artifacts without implying that analysis completed."""
    sentiments = _read_json(folder / "sentiment_results.json", [])
    keywords = _read_json(folder / "keyword_results.json", [])
    analyses = _read_json(folder / "review_analyses.json", {})
    issues = _read_json(folder / "issue_catalog.json", None)
    features = _read_json(folder / "feature_request_catalog.json", None)
    return {
        "sentiment_reviews": len(sentiments) if isinstance(sentiments, list) else 0,
        "keyword_reviews": len(keywords) if isinstance(keywords, list) else 0,
        "extracted_reviews": sum(isinstance(row, dict) and row.get("status") == "success"
                                 for row in analyses.values()) if isinstance(analyses, dict) else 0,
        "issue_count": len(issues) if isinstance(issues, list) else None,
        "feature_request_count": len(features) if isinstance(features, list) else None,
        "metrics_available": (folder / "nlp_metrics.json").exists(),
        "insights_available": (folder / "insights.json").exists(),
    }


def _save_meta(folder: Path, meta: dict[str, Any]) -> None:
    meta["updated_at"] = _utc_now()
    _write_json(folder / "scan.json", meta)


def _lookup_app_name(app_id: str) -> str | None:
    """Best-effort name lookup; metadata outages must never block review collection."""
    try:
        request = urllib.request.Request(
            f"https://itunes.apple.com/lookup?id={app_id}&country=us",
            headers={"User-Agent": "AppStoreReviewsAnalyzer/1.0"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results = payload.get("results") if isinstance(payload, dict) else None
        if results and isinstance(results[0], dict):
            name = results[0].get("trackName")
            return name.strip() if isinstance(name, str) and name.strip() else None
    except Exception:
        return None
    return None


def _history_summary(folder: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Build a compact history row exclusively from saved scan artifacts."""
    try:
        basic = _read_json(folder / "basic_metrics.json", {})
    except (OSError, ValueError):
        basic = {}
    try:
        nlp = _read_json(folder / "nlp_metrics.json", {})
    except (OSError, ValueError):
        nlp = {}
    try:
        sentiments = _read_json(folder / "sentiment_results.json", [])
    except (OSError, ValueError):
        sentiments = []
    try:
        issues = _read_json(folder / "issue_catalog.json", None)
    except (OSError, ValueError):
        issues = None
    if not isinstance(basic, dict):
        basic = {}
    if not isinstance(nlp, dict):
        nlp = {}
    if not isinstance(sentiments, list):
        sentiments = []
    sentiment = _sentiment_summary(sentiments)
    negative = sentiment["distribution"]["negative"]["share"]
    issue_rows = issues if isinstance(issues, list) else nlp.get("issue_metrics", {}).get("issues") if isinstance(nlp.get("issue_metrics"), dict) else None
    app_id = str(meta.get("app_id") or "unknown")
    return {
        "scan_id": meta.get("scan_id") or folder.name,
        "app_id": app_id,
        "app_name": meta.get("app_name") or f"App {app_id}",
        "created_at": meta.get("created_at"),
        "analysis_status": meta.get("analysis_status") or "not_started",
        "collection_mode": meta.get("collection_mode"),
        "country": meta.get("country"),
        "top_n": meta.get("top_n"),
        "review_count": meta.get("review_count", basic.get("review_count")),
        "average_rating": basic.get("average_rating"),
        "negative_sentiment_share": negative,
        "issue_count": len(issue_rows) if isinstance(issue_rows, list) else None,
    }


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
    top_n: int = Field(default=10, ge=1, le=len(SUPPORTED_COUNTRIES))
    max_pages: int = Field(default=MAX_RSS_PAGES, ge=1, le=MAX_RSS_PAGES)
    force_refresh: bool = False


class AnalyzeRequest(BaseModel):
    batch_size: int = Field(default=12, ge=1, le=12)
    concurrency: int = Field(default=48, ge=1, le=48)
    requests_per_minute: int = Field(default=480, ge=1, le=480)
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


@app.get("/api/scan-options")
def scan_options() -> dict[str, Any]:
    return {
        "countries": sorted(
            ({"code": code, "name": COUNTRY_NAMES[code]} for code in SUPPORTED_COUNTRIES),
            key=lambda item: item["name"],
        ),
        "max_pages": MAX_RSS_PAGES,
        "max_top_countries": len(SUPPORTED_COUNTRIES),
    }


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

    app_name = _lookup_app_name(payload.app_id)
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

    scan_runtime.reload()
    scan_id = uuid4().hex[:12]
    folder = DATA_ROOT / scan_id
    folder.mkdir(parents=True, exist_ok=False)
    basic_metrics = calculate_metrics(reviews)
    meta = {
        "scan_id": scan_id,
        "app_id": payload.app_id,
        "app_name": app_name or f"App {payload.app_id}",
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
    scan_runtime.commit()

    return {
        **meta,
        "basic_metrics": basic_metrics,
        "collection_errors": collection.get("errors") or [],
    }


@app.get("/api/scans")
def list_scans() -> dict[str, Any]:
    scan_runtime.reload()
    items = []
    if DATA_ROOT.exists():
        for folder in DATA_ROOT.iterdir():
            if not folder.is_dir():
                continue
            try:
                meta = _read_json(folder / "scan.json")
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict):
                items.append(_history_summary(folder, meta))
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
        "available_results": _available_results(folder) if meta.get("analysis_status") in {"cancelling", "interrupted"} else None,
    }


_PROGRESS_WEIGHTS = {
    "sentiment": 10, "keywords": 10, "issue_extraction": 50,
    "issue_normalization": 10, "feature_normalization": 5,
    "metrics": 5, "insights": 10,
}
_PROGRESS_MESSAGES = {
    "sentiment": "Analyzing sentiment",
    "keywords": "Extracting keywords",
    "issue_extraction": "Extracting issues and aspects",
    "issue_normalization": "Normalizing issues",
    "feature_normalization": "Normalizing feature requests",
    "metrics": "Calculating metrics",
    "insights": "Generating insights",
}


class _AnalysisProgress:
    """Serialize progress from the CPU thread and async LLM branch."""

    def __init__(self, folder: Path, meta: dict[str, Any]) -> None:
        self.folder, self.meta = folder, meta
        self.parts = {stage: 0.0 for stage in _PROGRESS_WEIGHTS}
        self.lock = threading.Lock()
        self.persisted_percent = 0
        self.persisted_stage = "running"
        self.persisted_at = time.monotonic()

    def update(self, stage: str, fraction: float, completed: int | None = None,
               total: int | None = None) -> None:
        with self.lock:
            self.parts[stage] = max(self.parts[stage], min(1.0, max(0.0, fraction)))
            previous = self.meta.get("progress", {})
            percent = max(previous.get("percent", 0),
                          round(sum(_PROGRESS_WEIGHTS[name] * value for name, value in self.parts.items())))
            # Keep showing extraction while a completed local branch reports
            # its weight and the slower LLM branch is still active.
            if (stage in {"sentiment", "keywords"} and previous.get("stage") == "issue_extraction"
                    and self.parts["issue_extraction"] < 1):
                progress = {**previous, "percent": percent}
            else:
                progress = {"percent": percent, "stage": stage, "message": _PROGRESS_MESSAGES[stage]}
                if completed is not None and total is not None:
                    progress.update({"completed": completed, "total": total})
            self.meta["progress"] = progress
            now = time.monotonic()
            if (progress["stage"] != self.persisted_stage or percent - self.persisted_percent >= 3
                    or now - self.persisted_at >= 5):
                _save_meta(self.folder, self.meta)
                scan_runtime.commit()
                self.persisted_percent, self.persisted_stage, self.persisted_at = percent, progress["stage"], now
                _check_stop_requested(self.folder)


def _run_analysis_job(scan_id: str, payload: AnalyzeRequest) -> None:
    scan_runtime.reload()
    folder = DATA_ROOT / scan_id
    meta = _read_json(folder / "scan.json", {})
    progress = _AnalysisProgress(folder, meta)
    try:
        _check_stop_requested(folder)
        meta["analysis_status"] = "running"
        meta["error"] = None
        meta["progress"] = {"percent": 0, "stage": "running", "message": "Starting analysis"}
        _save_meta(folder, meta)
        scan_runtime.commit()
        _check_stop_requested(folder)

        reviews = _read_json(folder / "reviews.json", [])
        run_full_pipeline(
            reviews,
            folder,
            run_local_nlp=payload.run_local_nlp,
            batch_size=payload.batch_size,
            concurrency=payload.concurrency,
            requests_per_minute=payload.requests_per_minute,
            max_cost_usd=payload.max_cost_usd,
            progress_callback=progress.update,
        )
        scan_runtime.commit()
        _check_stop_requested(folder)
        progress.update("insights", 0.0)
        run_saved_insights(folder, max_cost_usd=payload.max_cost_usd)
        scan_runtime.commit()
        _check_stop_requested(folder)

        meta = _read_json(folder / "scan.json", meta)
        meta["analysis_status"] = "completed"
        meta["error"] = None
        meta["progress"] = {"percent": 100, "stage": "completed", "message": "Analysis complete"}
        _save_meta(folder, meta)
        scan_runtime.commit()
    except AnalysisInterrupted:
        meta = _read_json(folder / "scan.json", meta)
        meta["analysis_status"] = "interrupted"
        meta["error"] = None
        meta["progress"] = {"percent": meta.get("progress", {}).get("percent", 0),
                            "stage": "interrupted", "message": "Analysis interrupted"}
        _save_meta(folder, meta)
        scan_runtime.commit()
    except Exception as exc:
        meta = _read_json(folder / "scan.json", meta)
        meta["analysis_status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["progress"] = {"percent": meta.get("progress", {}).get("percent", 0),
                            "stage": "failed", "message": "Analysis failed"}
        _save_meta(folder, meta)
        scan_runtime.commit()


@app.post("/api/scans/{scan_id}/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze_scan(
    scan_id: str,
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    meta = _load_meta(folder)
    if meta.get("analysis_status") in {"queued", "running", "cancelling"}:
        raise HTTPException(status_code=409, detail="Analysis is already running")

    (folder / "stop.requested").unlink(missing_ok=True)

    meta["analysis_status"] = "queued"
    meta["error"] = None
    meta["progress"] = {"percent": 0, "stage": "queued", "message": "Waiting to start"}
    _save_meta(folder, meta)
    scan_runtime.commit()
    if scan_runtime.dispatch_analysis is None:
        background_tasks.add_task(_run_analysis_job, scan_id, payload)
    else:
        try:
            scan_runtime.dispatch_analysis(scan_id, payload.model_dump())
        except Exception as exc:
            meta["analysis_status"] = "failed"
            meta["error"] = f"{type(exc).__name__}: {exc}"
            meta["progress"] = {"percent": 0, "stage": "failed", "message": "Analysis failed"}
            _save_meta(folder, meta)
            scan_runtime.commit()
            raise HTTPException(status_code=503, detail="Could not start analysis") from exc
    return {"scan_id": scan_id, "analysis_status": "queued"}


@app.post("/api/scans/{scan_id}/stop", status_code=status.HTTP_202_ACCEPTED)
def stop_scan(scan_id: str) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    meta = _load_meta(folder)
    if meta.get("analysis_status") == "cancelling":
        return {"scan_id": scan_id, "analysis_status": "cancelling"}
    if meta.get("analysis_status") not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Analysis is not running")
    (folder / "stop.requested").write_text(_utc_now(), encoding="utf-8")
    meta["analysis_status"] = "cancelling"
    meta["progress"] = {"percent": meta.get("progress", {}).get("percent", 0),
                        "stage": "cancelling", "message": "Stopping analysis"}
    _save_meta(folder, meta)
    scan_runtime.commit()
    return {"scan_id": scan_id, "analysis_status": "cancelling"}


@app.get("/api/scans/{scan_id}/metrics")
def get_metrics(scan_id: str) -> dict[str, Any]:
    folder = _scan_dir(scan_id)
    basic = _read_json(folder / "basic_metrics.json")
    if basic is None:
        reviews = _read_json(folder / "reviews.json", [])
        basic = calculate_metrics(reviews)
        _write_json(folder / "basic_metrics.json", basic)
        scan_runtime.commit()

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


@app.get("/api/scans/{scan_id}/report/download")
def download_analysis_report(scan_id: str) -> FileResponse:
    """Assemble a downloadable report from artifacts already saved for a scan."""
    folder = _scan_dir(scan_id)
    meta = _load_meta(folder)
    reviews = _read_json(folder / "reviews.json", [])
    if not isinstance(reviews, list):
        reviews = []

    sentiment_rows = _read_json(folder / "sentiment_results.json", [])
    keyword_rows = _read_json(folder / "keyword_results.json", [])
    analyses = _read_json(folder / "review_analyses.json", {})
    if not isinstance(sentiment_rows, list):
        sentiment_rows = []
    if not isinstance(keyword_rows, list):
        keyword_rows = []
    if not isinstance(analyses, dict):
        analyses = {}

    sentiment_by_id = {
        str(row["review_id"]): row for row in sentiment_rows
        if isinstance(row, dict) and row.get("review_id") is not None
    }
    keywords_by_id = {
        str(row["review_id"]): row for row in keyword_rows
        if isinstance(row, dict) and row.get("review_id") is not None
    }
    enriched_reviews = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        review_id = str(review.get("review_id", ""))
        sentiment = sentiment_by_id.get(review_id, {})
        keyword = keywords_by_id.get(review_id, {})
        enriched_reviews.append({
            **review,
            "sentiment": sentiment.get("sentiment"),
            "sentiment_scores": sentiment.get("sentiment_scores", sentiment.get("scores")),
            "keywords": keyword.get("keywords") or [],
            "issue_analysis": analyses.get(review_id),
        })

    basic_metrics = _read_json(folder / "basic_metrics.json")
    sentiment_summary = _sentiment_summary(sentiment_rows)
    nlp_metrics = _read_json(folder / "nlp_metrics.json")
    insights = _read_json(folder / "insights.json")
    report = {
        "report_version": "1.0",
        "scan": {
            "scan_id": meta.get("scan_id", scan_id),
            "app_id": meta.get("app_id"),
            "app_name": meta.get("app_name"),
            "created_at": meta.get("created_at"),
            "collection_mode": meta.get("collection_mode"),
            "country": meta.get("country"),
            "top_n": meta.get("top_n"),
            "review_count": meta.get("review_count", len(reviews)),
            "analysis_status": meta.get("analysis_status"),
        },
        "basic_metrics": basic_metrics,
        "sentiment_summary": sentiment_summary,
        "nlp_metrics": nlp_metrics,
        "issues": _read_json(folder / "issue_catalog.json", []),
        "feature_requests": _read_json(folder / "feature_request_catalog.json", []),
        "insights": insights,
        "reviews": enriched_reviews,
    }
    report_path = folder / f"app_{meta.get('app_id', 'unknown')}_{scan_id}_analysis_report.json"
    _write_json(report_path, report)
    return FileResponse(report_path, media_type="application/json", filename=report_path.name)


@app.get("/api/scans/{scan_id}/report/download.pdf")
def download_analysis_report_pdf(scan_id: str) -> FileResponse:
    """Print the frontend's report page using headless Chromium."""
    folder = _scan_dir(scan_id)
    meta = _load_meta(folder)
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="PDF export requires Playwright. Install the browser with 'playwright install chromium'.",
        ) from exc

    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173").rstrip("/")
    report_url = f"{frontend_url}/scans/{scan_id}/report"
    pdf_path = folder / f"app_{meta.get('app_id', 'unknown')}_{scan_id}_analysis_report.pdf"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                response = page.goto(report_url, wait_until="domcontentloaded", timeout=30_000)
                if response is None or not response.ok:
                    raise HTTPException(status_code=502, detail="Could not load the report page for PDF export")
                page.wait_for_selector('[data-report-ready="true"]', timeout=45_000)
                page.evaluate("document.fonts.ready")
                page.wait_for_function(
                    "Array.from(document.querySelectorAll('svg')).every(svg => svg.querySelector('*'))",
                    timeout=15_000,
                )
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "14mm", "right": "14mm", "bottom": "14mm", "left": "14mm"},
                )
            finally:
                browser.close()
    except HTTPException:
        raise
    except PlaywrightError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PDF export failed: {str(exc)[:1000]}",
        ) from exc

    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
