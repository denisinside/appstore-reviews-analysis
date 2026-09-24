"""One-off access to public Apple App Store customer review feeds."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx

from .cache import make_cache
from .config import (
    CACHE_TTL_SECONDS,
    EMPTY_PAGE_TTL_SECONDS,
    HTTP_ATTEMPTS,
    HTTP_TIMEOUT_SECONDS,
    MAX_CONCURRENT_REQUESTS,
    MAX_RSS_PAGES,
    RETRY_BACKOFF_SECONDS,
    SUPPORTED_COUNTRIES,
)
from .language import LanguageDetector

LOGGER = logging.getLogger(__name__)


class PageFetchError(RuntimeError):
    pass


def _validate_app_id(app_id) -> str:
    if isinstance(app_id, bool) or not re.fullmatch(r"[0-9]+", str(app_id or "")) or int(app_id) <= 0:
        raise ValueError("app_id must be a positive numeric identifier")
    return str(app_id)


def _validate_country(country) -> str:
    if not isinstance(country, str) or country.lower() not in SUPPORTED_COUNTRIES:
        raise ValueError(f"country must be one of: {', '.join(SUPPORTED_COUNTRIES)}")
    return country.lower()


def _validate_number(value, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _label(value):
    if isinstance(value, dict):
        value = value.get("label")
    return value if isinstance(value, str) else None


def _date(value):
    raw = _label(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return parsed.isoformat()
    except ValueError:
        LOGGER.warning("Skipping invalid review date: %s", raw)
        return None


def _feed_entries(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("feed"), dict):
        raise ValueError("Apple response is missing a valid feed object")
    entries = payload["feed"].get("entry", [])
    if isinstance(entries, dict):
        return [entries]
    if isinstance(entries, list):
        return entries
    raise ValueError("Apple feed.entry must be an array or object")


def _normalize_review(entry, app_id: str, country: str):
    if not isinstance(entry, dict):
        LOGGER.warning("Skipping non-object feed entry in %s", country)
        return None
    # The feed can contain an app metadata entry. It has no review rating.
    if "im:name" in entry or "im:artist" in entry:
        return None
    raw_id = _label(entry.get("id"))
    if not raw_id or not raw_id.isascii() or not raw_id.isdigit() or int(raw_id) <= 0:
        LOGGER.warning("Skipping review with missing or invalid review ID in %s", country)
        return None
    raw_rating = _label(entry.get("im:rating"))
    if not raw_rating or not raw_rating.isascii() or not raw_rating.isdigit() or not 1 <= int(raw_rating) <= 5:
        LOGGER.warning("Skipping review %s with missing or invalid rating", raw_id)
        return None
    title = _label(entry.get("title"))
    text = _label(entry.get("content"))
    version = _label(entry.get("im:version"))
    return {
        "review_id": raw_id,
        "app_id": app_id,
        "country": country,
        "observed_countries": [country],
        "title": title,
        "text": text,
        "rating": int(raw_rating),
        "app_version": version,
        "updated_at": _date(entry.get("updated")),
        "language": None,
        "language_confidence": None,
    }


class AppStoreReviews:
    """Reusable synchronous scraper. Calls are safe to run one mode at a time."""

    def __init__(
        self,
        *,
        cache=None,
        language_detector=None,
        client: httpx.Client | None = None,
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
        sleep=time.sleep,
    ):
        self.cache = cache if cache is not None else make_cache()
        self.language_detector = language_detector if language_detector is not None else LanguageDetector()
        self.client = client if client is not None else httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
        self.max_concurrent_requests = _validate_number(max_concurrent_requests, "max_concurrent_requests", 32)
        self.sleep = sleep

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _fetch_page(self, app_id: str, country: str, page: int, force_refresh: bool):
        key = f"appstore:page:{app_id}:{country}:{page}"
        if not force_refresh:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        for attempt in range(HTTP_ATTEMPTS):
            try:
                response = self.client.get(url)
                response.raise_for_status()
                payload = response.json()
                entries = _feed_entries(payload)
                ttl = EMPTY_PAGE_TTL_SECONDS if not entries else CACHE_TTL_SECONDS
                self.cache.set(key, payload, ttl)
                return payload
            except httpx.HTTPStatusError as exc:
                retry = exc.response.status_code == 429 or exc.response.status_code >= 500
                error = f"HTTP {exc.response.status_code}"
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                retry = True
                error = f"{type(exc).__name__}: {exc}"
            except (ValueError, TypeError) as exc:
                raise PageFetchError(f"Invalid Apple JSON/feed: {exc}") from exc
            except httpx.HTTPError as exc:
                retry = False
                error = f"{type(exc).__name__}: {exc}"
            if not retry or attempt == HTTP_ATTEMPTS - 1:
                raise PageFetchError(error)
            self.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
        raise AssertionError("unreachable")

    def _collect_country(self, app_id: str, country: str, max_pages: int, force_refresh: bool):
        reviews = {}
        errors = []
        pages_succeeded = 0
        for page in range(1, max_pages + 1):
            try:
                payload = self._fetch_page(app_id, country, page, force_refresh)
                entries = _feed_entries(payload)
                pages_succeeded += 1
            except PageFetchError as exc:
                LOGGER.warning("%s page %s failed: %s", country, page, exc)
                errors.append({"country": country, "page": page, "error": str(exc)})
                continue
            for entry in entries:
                review = _normalize_review(entry, app_id, country)
                if review is not None:
                    reviews.setdefault(review["review_id"], review)
        status = "success" if pages_succeeded == max_pages else "partial" if pages_succeeded else "failed"
        return {
            "country": country,
            "status": status,
            "pages_requested": max_pages,
            "pages_succeeded": pages_succeeded,
            "reviews": list(reviews.values()),
            "errors": errors,
        }

    def _safe_collect(self, app_id: str, country: str, max_pages: int, force_refresh: bool):
        try:
            return self._collect_country(app_id, country, max_pages, force_refresh)
        except Exception as exc:
            LOGGER.exception("Unexpected collection failure in %s", country)
            return {
                "country": country,
                "status": "failed",
                "pages_requested": max_pages,
                "pages_succeeded": 0,
                "reviews": [],
                "errors": [{"country": country, "page": None, "error": str(exc)}],
            }

    def discovery(self, app_id, *, force_refresh: bool = False):
        app_id = _validate_app_id(app_id)
        key = f"appstore:discovery:{app_id}"
        if not force_refresh:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as pool:
            results = list(pool.map(
                lambda country: self._safe_collect(app_id, country, MAX_RSS_PAGES, force_refresh),
                SUPPORTED_COUNTRIES,
            ))
        countries = []
        errors = []
        for result in results:
            dates = [r["updated_at"] for r in result["reviews"] if r["updated_at"]]
            countries.append({
                "country": result["country"],
                "rank": None,
                "review_count": len(result["reviews"]) if result["status"] != "failed" else None,
                "newest_review_at": max(dates) if dates else None,
                "oldest_review_at": min(dates) if dates else None,
                "status": result["status"],
            })
            errors.extend(result["errors"])
        # Stable passes keep fully checked countries ahead of partial results.
        countries.sort(key=lambda row: row["country"])
        countries.sort(key=lambda row: row["newest_review_at"] or "", reverse=True)
        countries.sort(key=lambda row: row["review_count"] or 0, reverse=True)
        status_order = {"success": 0, "partial": 1, "failed": 2}
        countries.sort(key=lambda row: status_order[row["status"]])
        for rank, row in enumerate(countries, 1):
            row["rank"] = rank
        output = {"app_id": app_id, "mode": "discovery", "countries": countries, "errors": errors}
        if not errors:
            self.cache.set(key, output, CACHE_TTL_SECONDS)
        return output

    def get_reviews(self, app_id, country, *, max_pages: int = MAX_RSS_PAGES, force_refresh: bool = False):
        app_id = _validate_app_id(app_id)
        country = _validate_country(country)
        max_pages = _validate_number(max_pages, "max_pages", MAX_RSS_PAGES)
        self.language_detector.ensure_loaded()
        result = self._safe_collect(app_id, country, max_pages, force_refresh)
        for review in result["reviews"]:
            review["language"], review["language_confidence"] = self.language_detector.detect(
                review["title"], review["text"]
            )
        return {
            "app_id": app_id,
            "mode": "country",
            "country": country,
            "status": result["status"],
            "pages_requested": result["pages_requested"],
            "pages_succeeded": result["pages_succeeded"],
            "review_count": len(result["reviews"]) if result["status"] != "failed" else None,
            "reviews": result["reviews"],
            "errors": result["errors"],
        }

    def get_top_reviews(self, app_id, *, top_n: int = 10, max_pages: int = MAX_RSS_PAGES, force_refresh: bool = False):
        app_id = _validate_app_id(app_id)
        top_n = _validate_number(top_n, "top_n", len(SUPPORTED_COUNTRIES))
        max_pages = _validate_number(max_pages, "max_pages", MAX_RSS_PAGES)
        self.language_detector.ensure_loaded()
        discovery = self.discovery(app_id, force_refresh=force_refresh)
        selected = [row["country"] for row in discovery["countries"] if row["status"] == "success"][:top_n]
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as pool:
            results = list(pool.map(
                lambda country: self._safe_collect(app_id, country, max_pages, False),
                selected,
            ))
        all_reviews = {}
        countries = []
        errors = [{"stage": "discovery", **error} for error in discovery["errors"]]
        for result in results:
            added = 0
            for review in result["reviews"]:
                review_id = review["review_id"]
                if review_id in all_reviews:
                    observed = all_reviews[review_id]["observed_countries"]
                    if result["country"] not in observed:
                        observed.append(result["country"])
                else:
                    review["language"], review["language_confidence"] = self.language_detector.detect(
                        review["title"], review["text"]
                    )
                    all_reviews[review_id] = review
                    added += 1
            countries.append({
                "country": result["country"],
                "status": result["status"],
                "pages_requested": max_pages,
                "pages_succeeded": result["pages_succeeded"],
                "review_count": len(result["reviews"]) if result["status"] != "failed" else None,
                "added_review_count": added,
            })
            errors.extend({"stage": "reviews", **error} for error in result["errors"])
        status = "success" if len(selected) == top_n and not errors else "partial" if selected else "failed"
        return {
            "app_id": app_id,
            "mode": "top",
            "top_n": top_n,
            "max_pages": max_pages,
            "status": status,
            "countries": countries,
            "review_count": len(all_reviews),
            "reviews": list(all_reviews.values()),
            "errors": errors,
        }
