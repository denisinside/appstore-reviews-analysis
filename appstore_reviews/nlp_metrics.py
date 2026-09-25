"""Deterministic metrics for review issue/aspect analyses (standard library only)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable, Mapping


def _rid(row: Mapping[str, Any]) -> str | None:
    value = row.get("review_id", row.get("id"))
    return None if value is None else str(value)


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(v, review_id=str(k)) if isinstance(v, Mapping) else {"review_id": str(k)} for k, v in value.items()]
    return [dict(x) for x in (value or []) if isinstance(x, Mapping)]


def _analysis(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "error", "aspects": []}
    return dict(value)


def _field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return None


def _rating(row: Mapping[str, Any]) -> float | None:
    v = _field(row, "rating", "score", "stars")
    try:
        value = float(v) if v is not None else None
        return value if value is not None and 1 <= value <= 5 else None
    except (TypeError, ValueError):
        return None


def _ratio(n: int | float, d: int | float) -> float | None:
    return round(n / d, 6) if d else None


def _sentiment(row: Mapping[str, Any]) -> str | None:
    s = str(row.get("sentiment", "")).lower()
    if s in {"positive", "neutral", "negative"}:
        return s
    return None


def _diverse_samples(ids: set[str], reviews: dict[str, dict[str, Any]], limit: int = 5) -> list[str]:
    """Pick review IDs across countries, ratings, and versions where possible."""
    remaining = sorted(ids)
    selected: list[str] = []
    seen: set[tuple[str, str]] = set()
    for field in ("country", "rating", "app_version"):
        for rid in list(remaining):
            value = reviews.get(rid, {}).get(field)
            marker = (field, str(value))
            if value is not None and marker not in seen and len(selected) < limit:
                selected.append(rid)
                remaining.remove(rid)
                seen.add(marker)
    return (selected + remaining)[:limit]


def _catalog_metric(catalog: Iterable[Mapping[str, Any]], relation: dict[str, set[str]],
                    reviews: dict[str, dict[str, Any]], successful: set[str], *, kind: str) -> list[dict[str, Any]]:
    results = []
    seen = set()
    for entry in catalog or []:
        if not isinstance(entry, Mapping):
            continue
        cid = _field(entry, "canonical_id", "id")
        if cid is None:
            continue
        cid = str(cid)
        ids = relation.get(cid, set()) & successful
        # Catalog membership is a fallback for persisted catalogs lacking mappings in analyses.
        if not ids:
            ids = {str(x) for x in (entry.get("review_ids") or [])} & successful
        seen.add(cid)
        rows = [reviews[x] for x in sorted(ids) if x in reviews]
        ratings = [v for r in rows if (v := _rating(r)) is not None]
        results.append({
            "canonical_id": cid,
            "canonical_name": _field(entry, "canonical_name", "name"),
            "category": entry.get("category"),
            "review_count": len(ids),
            "denominator_review_count": len(successful),
            "review_ids": sorted(ids),
            "share_of_successful_reviews": _ratio(len(ids), len(successful)),
            **({
                "average_rating": round(sum(ratings) / len(ratings), 4) if ratings else None,
                "negative_rating_review_count": sum(1 for r in rows if (_rating(r) is not None and 1 <= _rating(r) <= 2)),
                "negative_rating_share": _ratio(sum(1 for r in rows if (_rating(r) is not None and 1 <= _rating(r) <= 2)), len(ratings)),
                "rating_sample_size": len(ratings),
                "sample_review_ids": _diverse_samples(ids, reviews),
            } if kind == "issue" else {"sample_review_ids": _diverse_samples(ids, reviews)}),
        })
    return results


def calculate_nlp_metrics(reviews, analyses, issue_catalog, feature_catalog, sentiment_results=None) -> dict:
    """Calculate metrics from review rows, review-id keyed analyses, and canonical catalogs.

    Failed or absent analyses are excluded from all success-based denominators. Evidence
    verification is honored: unverified signals are not counted. IDs are deduplicated.
    """
    review_rows = _rows(reviews)
    review_map = {rid: row for row in review_rows if (rid := _rid(row)) is not None}
    analysis_map = {str(k): _analysis(v) for k, v in (analyses or {}).items()} if isinstance(analyses, Mapping) else {}
    incoming_ids = set(review_map)
    status = {rid: str(a.get("status", "error")).lower() for rid, a in analysis_map.items() if rid in incoming_ids}
    successful = {rid for rid, s in status.items() if s in {"success", "succeeded", "completed", "ok"}}
    errors = incoming_ids - successful

    # Optional overall sentiment source: mappings or rows keyed by review ID.
    overall = {}
    if isinstance(sentiment_results, Mapping):
        for rid, item in sentiment_results.items():
            overall[str(rid)] = item if isinstance(item, Mapping) else {"sentiment": item}
    else:
        for item in _rows(sentiment_results):
            if (rid := _rid(item)) is not None:
                overall[rid] = item

    aspect_mentions: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    aspect_mention_counts: dict[str, Counter] = defaultdict(Counter)
    category_reviews: dict[str, set[str]] = defaultdict(set)
    issue_reviews: dict[str, set[str]] = defaultdict(set)
    feature_reviews: dict[str, set[str]] = defaultdict(set)
    issue_category: dict[str, str | None] = {}
    for rid in successful:
        a = analysis_map[rid]
        for aspect in a.get("aspects", []) or []:
            if not isinstance(aspect, Mapping):
                continue
            if aspect.get("evidence_verified", True) is False:
                continue
            category = str(aspect.get("category", "other"))
            s = _sentiment(aspect)
            if s:
                aspect_mentions[category][s].add(rid)
                aspect_mention_counts[category][s] += 1
                category_reviews[category].add(rid)
            for issue in aspect.get("issues", []) or []:
                if not isinstance(issue, Mapping) or issue.get("evidence_verified", issue.get("verified", True)) is False:
                    continue
                cid = _field(issue, "canonical_id", "canonical_issue_id")
                if cid is not None:
                    cid = str(cid)
                    issue_reviews[cid].add(rid)
                    issue_category[cid] = category
            for feature in aspect.get("feature_requests", []) or []:
                if not isinstance(feature, Mapping) or feature.get("evidence_verified", feature.get("verified", True)) is False:
                    continue
                cid = _field(feature, "canonical_id", "canonical_feature_id")
                if cid is not None:
                    feature_reviews[str(cid)].add(rid)

    categories = sorted(set(category_reviews) | set(aspect_mentions))
    aspects_out = []
    for cat in categories:
        class_reviews = aspect_mentions[cat]
        positive, neutral, negative = (class_reviews[x] for x in ("positive", "neutral", "negative"))
        any_reviews = category_reviews[cat]
        mixed = (positive & negative) | (positive & neutral) | (negative & neutral)
        n = len(any_reviews)
        aspects_out.append({
            "category": cat,
            "review_count": n,
            "denominator_review_count": len(successful),
            "mention_count": sum(aspect_mention_counts[cat].values()),
            "share_of_successful_reviews": _ratio(n, len(successful)),
            "sentiment_distribution": {
                s: {"review_count": len(class_reviews[s]), "mention_count": aspect_mention_counts[cat][s],
                    "share_of_aspect_reviews": _ratio(len(class_reviews[s]), n)}
                for s in ("positive", "neutral", "negative")
            },
            "mixed_review_count": len(mixed),
            "negative_aspect_share": _ratio(len(negative), n),
        })

    issues_out = _catalog_metric(issue_catalog, issue_reviews, review_map, successful, kind="issue")
    features_out = _catalog_metric(feature_catalog, feature_reviews, review_map, successful, kind="feature")
    # Include canonical IDs observed in analyses even if a catalog row has not yet been written.
    known_issue = {x["canonical_id"] for x in issues_out}
    for cid, ids in sorted(issue_reviews.items()):
        if cid not in known_issue:
            rows = [review_map[x] for x in ids]
            vals = [x for row in rows if (x := _rating(row)) is not None]
            neg = sum(1 for x in vals if 1 <= x <= 2)
            issues_out.append({"canonical_id": cid, "canonical_name": None, "category": issue_category.get(cid),
                               "review_count": len(ids), "review_ids": sorted(ids),
                               "denominator_review_count": len(successful),
                               "share_of_successful_reviews": _ratio(len(ids), len(successful)),
                               "average_rating": round(sum(vals) / len(vals), 4) if vals else None,
                               "negative_rating_review_count": neg, "negative_rating_share": _ratio(neg, len(vals)),
                               "rating_sample_size": len(vals), "sample_review_ids": _diverse_samples(ids, review_map)})
    known_features = {x["canonical_id"] for x in features_out}
    for cid, ids in sorted(feature_reviews.items()):
        if cid not in known_features:
            features_out.append({"canonical_id": cid, "canonical_name": None, "category": None,
                                 "review_count": len(ids), "review_ids": sorted(ids),
                                 "denominator_review_count": len(successful),
                                 "share_of_successful_reviews": _ratio(len(ids), len(successful)),
                                 "sample_review_ids": _diverse_samples(ids, review_map)})

    problem_ids = set().union(*issue_reviews.values()) if issue_reviews else set()
    feature_ids = set().union(*feature_reviews.values()) if feature_reviews else set()
    negative_overall = set()
    if overall:
        for rid in successful:
            s = str(overall.get(rid, {}).get("sentiment", "")).lower()
            if s == "negative":
                negative_overall.add(rid)
    else:
        # Preserve source sentiment metadata when available on the review records.
        negative_overall = {rid for rid in successful if str(review_map[rid].get("sentiment", "")).lower() == "negative"}
    coverage = {
        "submitted_review_count": len(incoming_ids),
        "successful_review_count": len(successful),
        "successful_share": _ratio(len(successful), len(incoming_ids)),
        "error_count": len(errors),
        "error_review_ids": sorted(errors),
        "reviews_with_issue_count": len(problem_ids),
        "reviews_with_issue_share": _ratio(len(problem_ids), len(successful)),
        "negative_sentiment_review_count": len(negative_overall),
        "negative_sentiment_denominator_review_count": len(negative_overall),
        "negative_reviews_with_issue_count": len(negative_overall & problem_ids),
        "negative_reviews_with_issue_share": _ratio(len(negative_overall & problem_ids), len(negative_overall)),
    }

    def breakdown(field: str) -> dict[str, Any]:
        groups: dict[str, set[str]] = defaultdict(set)
        for rid in successful:
            if field == "country":
                values = [review_map[rid].get("country"), *(review_map[rid].get("observed_countries") or [])]
            else:
                values = [review_map[rid].get("app_version")]
            for value in values:
                if value is not None and str(value).strip():
                    groups[str(value)].add(rid)
        result = {}
        for key, ids in sorted(groups.items()):
            issue_counts = {entry["canonical_id"]: len(issue_reviews.get(entry["canonical_id"], set()) & ids) for entry in issues_out}
            sent_counts: dict[str, dict[str, int]] = {}
            for cat in categories:
                sent_counts[cat] = {s: len(aspect_mentions[cat][s] & ids) for s in ("positive", "neutral", "negative")}
            result[key] = {
                "successful_review_count": len(ids),
                "denominator_review_count": len(ids),
                "issue_frequency": issue_counts,
                "issue_share": {cid: _ratio(count, len(ids)) for cid, count in issue_counts.items()},
                "aspect_sentiment": sent_counts,
            }
        return result

    time_groups: dict[str, set[str]] = defaultdict(set)
    time_field_seen = False
    for rid in successful:
        raw = _field(review_map[rid], "updated_at", "date", "review_date", "created_at", "timestamp")
        if raw is None:
            continue
        time_field_seen = True
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            key = dt.strftime("%Y-%m")
        except ValueError:
            key = str(raw)[:7]
        time_groups[key].add(rid)
    time_breakdown = {
        period: {
            "successful_review_count": len(ids),
            "denominator_review_count": len(ids),
            "issue_frequency": {e["canonical_id"]: len(issue_reviews.get(e["canonical_id"], set()) & ids) for e in issues_out},
            "issue_share": {e["canonical_id"]: _ratio(len(issue_reviews.get(e["canonical_id"], set()) & ids), len(ids)) for e in issues_out},
            "aspect_sentiment": {cat: {s: len(aspect_mentions[cat][s] & ids) for s in ("positive", "neutral", "negative")} for cat in categories},
        } for period, ids in sorted(time_groups.items())
    }

    overall_rating_vals = [_rating(review_map[rid]) for rid in successful]
    overall_rating_vals = [x for x in overall_rating_vals if x is not None]
    return {
        "analysis_summary": {
            "input_review_count": len(incoming_ids), "analyzed_review_count": len(successful),
            "analysis_error_count": len(errors), "rating_sample_size": len(overall_rating_vals),
            "average_rating": round(sum(overall_rating_vals) / len(overall_rating_vals), 4) if overall_rating_vals else None,
            "rules": ["Unique review IDs are counted once per metric.", "Issue and feature denominators use successful analyses.",
                      "Unverified evidence is excluded from counts.", "Rating shares use reviews with a valid rating.",
                      "Missing metadata is omitted; no values are imputed.",
                      "Time buckets describe accessible reviews; they do not establish population trends."],
            "metric_rules": {
                "aspect_frequency": "distinct reviews mentioning the category / successful analyses",
                "aspect_sentiment": "distinct reviews per class; mixed reviews can occur in multiple classes",
                "negative_aspect_share": "reviews with a negative category mention / reviews mentioning the category",
                "issue_and_feature_share": "distinct reviews with canonical signal / successful analyses",
                "average_rating": "mean of available ratings for reviews with the issue",
                "negative_rating_share": "ratings of 1 or 2 / available ratings for reviews with the issue",
                "negative_sentiment_issue_coverage": "negative overall sentiment reviews with a concrete issue / negative overall sentiment reviews",
                "breakdown_share": "distinct reviews with issue in bucket / successful analyses in bucket",
            },
        },
        "aspect_metrics": aspects_out,
        "issue_metrics": {"reviews_with_any_issue_count": len(problem_ids), "reviews_with_any_issue_share": _ratio(len(problem_ids), len(successful)), "issues": issues_out},
        "feature_request_metrics": {"reviews_with_any_feature_request_count": len(feature_ids), "reviews_with_any_feature_request_share": _ratio(len(feature_ids), len(successful)), "feature_requests": features_out},
        "coverage_metrics": coverage,
        "country_breakdown": breakdown("country"),
        "version_breakdown": breakdown("version"),
        "time_breakdown": time_breakdown if time_field_seen else {},
    }
