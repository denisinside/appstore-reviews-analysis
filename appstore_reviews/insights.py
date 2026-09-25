"""Actionable insights generated from deterministic NLP metrics and grounded reviews."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .openrouter_client import MODEL_ID, OpenRouterClient


INSIGHTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_summary": {
            "type": "string",
            "minLength": 1,
        },
        "issue_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical_id": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "finding": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "user_impact": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "recommended_actions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "supporting_review_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 15,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "caveat": {
                        "type": "string",
                    },
                },
                "required": [
                    "canonical_id",
                    "finding",
                    "user_impact",
                    "recommended_actions",
                    "supporting_review_ids",
                    "caveat",
                ],
            },
        },
        "feature_request_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical_id": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "finding": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "recommended_action": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "supporting_review_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 15,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "caveat": {
                        "type": "string",
                    },
                },
                "required": [
                    "canonical_id",
                    "finding",
                    "recommended_action",
                    "supporting_review_ids",
                    "caveat",
                ],
            },
        },
        "limitations": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
            },
        },
    },
    "required": [
        "overall_summary",
        "issue_insights",
        "feature_request_insights",
        "limitations",
    ],
}


SYSTEM_PROMPT = """
You generate grounded, actionable product insights from App Store review analysis.

The input already contains deterministic metrics calculated in Python and
representative original reviews.

Your job is qualitative interpretation only.

For every supplied issue:
- explain the recurring user-observed problem;
- explain the practical user impact;
- suggest 1-3 realistic product, UX, QA, or investigation actions;
- cite only review IDs supplied for that issue.

For feature requests:
- summarize what users are requesting;
- suggest a reasonable product action or validation step.

Important rules:

1. Do NOT calculate or invent counts, percentages, ratings, trends, or statistics.
2. Treat supplied metrics as facts for this collected review sample only.
3. Do NOT claim that reviews represent all users.
4. Do NOT invent technical root causes.
5. If the reviews support only a symptom, recommend investigating the cause.
6. Do NOT invent features, bugs, countries, versions, or user behavior.
7. Do NOT assume correlation is causation.
8. If examples inside one canonical issue actually describe multiple related
   sub-patterns, explicitly mention that instead of pretending they are one
   identical technical failure.
9. Recommendations must follow from the supplied evidence.
10. Supporting review IDs must come only from the examples supplied for the
    corresponding canonical ID.
11. Keep the output concise and useful for a product/engineering team.
12. Return exactly one issue_insight for every supplied issue and exactly one
    feature_request_insight for every supplied feature request.

Return schema-valid JSON only.
""".strip()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _review_id(review: Mapping[str, Any]) -> str | None:
    value = review.get("review_id", review.get("id"))
    return str(value) if value is not None else None


def _sample_count(review_count: int) -> int:
    if review_count <= 0:
        return 0

    ten_percent = int(review_count * 0.10)
    desired = max(5, min(15, ten_percent))

    return min(review_count, desired)


def _select_diverse_review_ids(
    review_ids: list[str],
    reviews: Mapping[str, Mapping[str, Any]],
    count: int,
) -> list[str]:
    remaining = sorted(set(str(x) for x in review_ids))
    selected: list[str] = []

    if count <= 0:
        return []

    # First try to cover different metadata values.
    for field in ("language", "country", "rating", "app_version"):
        seen_values: set[str] = set()

        for rid in list(remaining):
            if len(selected) >= count:
                break

            value = reviews.get(rid, {}).get(field)

            if value is None:
                continue

            marker = str(value)

            if marker in seen_values:
                continue

            seen_values.add(marker)
            selected.append(rid)
            remaining.remove(rid)

        if len(selected) >= count:
            break

    # Fill the rest deterministically.
    for rid in remaining:
        if len(selected) >= count:
            break
        selected.append(rid)

    return selected


def _signal_evidence(
    analysis: Mapping[str, Any],
    canonical_id: str,
    field: str,
) -> list[str]:
    evidence: list[str] = []

    for aspect in analysis.get("aspects", []) or []:
        if not isinstance(aspect, Mapping):
            continue

        for signal in aspect.get(field, []) or []:
            if not isinstance(signal, Mapping):
                continue

            if signal.get("canonical_id") != canonical_id:
                continue

            value = signal.get("evidence")

            if isinstance(value, str) and value.strip():
                evidence.append(value.strip())

            for alternate in signal.get("alternate_formulations", []) or []:
                if not isinstance(alternate, Mapping):
                    continue

                value = alternate.get("evidence")

                if isinstance(value, str) and value.strip():
                    evidence.append(value.strip())

    return list(dict.fromkeys(evidence))


def _review_payload(
    rid: str,
    review: Mapping[str, Any],
    analysis: Mapping[str, Any],
    canonical_id: str,
    signal_field: str,
) -> dict[str, Any]:
    text = review.get("text")

    if text in (None, ""):
        text = review.get("content", "")

    return {
        "review_id": rid,
        "title": review.get("title") or "",
        "text": text or "",
        "rating": review.get("rating"),
        "language": review.get("language"),
        "country": review.get("country"),
        "app_version": review.get("app_version"),
        "date": (
            review.get("updated_at")
            or review.get("date")
            or review.get("review_date")
        ),
        "issue_evidence": _signal_evidence(
            analysis,
            canonical_id,
            signal_field,
        ),
    }


def _issue_sort_key(row: Mapping[str, Any]) -> tuple:
    count = int(row.get("review_count") or 0)

    negative_share = row.get("negative_rating_share")
    negative_share = float(negative_share) if negative_share is not None else -1.0

    average_rating = row.get("average_rating")
    average_rating = float(average_rating) if average_rating is not None else 6.0

    return (
        -count,
        -negative_share,
        average_rating,
        str(row.get("canonical_id", "")),
    )


def _feature_sort_key(row: Mapping[str, Any]) -> tuple:
    return (
        -int(row.get("review_count") or 0),
        str(row.get("canonical_id", "")),
    )


def build_insights_input(
    reviews: list[dict],
    analyses: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    *,
    max_issues: int = 8,
    max_feature_requests: int = 5,
) -> dict[str, Any]:
    review_map = {
        rid: review
        for review in reviews
        if (rid := _review_id(review)) is not None
    }

    analyzed_count = int(
        metrics.get("analysis_summary", {}).get("analyzed_review_count") or 0
    )

    # For ~100 reviews this becomes 2.
    minimum_support = max(
        2,
        math.ceil(analyzed_count * 0.01),
    )

    all_issues = list(
        metrics.get("issue_metrics", {}).get("issues", []) or []
    )

    supported_issues = [
        row
        for row in all_issues
        if int(row.get("review_count") or 0) >= minimum_support
    ]

    # A very small scan should still be able to produce something useful.
    if not supported_issues:
        supported_issues = [
            row
            for row in all_issues
            if int(row.get("review_count") or 0) > 0
        ]

    selected_issues = sorted(
        supported_issues,
        key=_issue_sort_key,
    )[:max_issues]

    all_features = list(
        metrics.get(
            "feature_request_metrics",
            {},
        ).get(
            "feature_requests",
            [],
        )
        or []
    )

    supported_features = [
        row
        for row in all_features
        if int(row.get("review_count") or 0) >= minimum_support
    ]

    if not supported_features:
        supported_features = [
            row
            for row in all_features
            if int(row.get("review_count") or 0) > 0
        ]

    selected_features = sorted(
        supported_features,
        key=_feature_sort_key,
    )[:max_feature_requests]

    issue_payloads = []

    for issue in selected_issues:
        count = int(issue.get("review_count") or 0)
        sample_size = _sample_count(count)

        ids = _select_diverse_review_ids(
            list(issue.get("review_ids") or []),
            review_map,
            sample_size,
        )

        examples = [
            _review_payload(
                rid,
                review_map[rid],
                analyses.get(rid, {}),
                str(issue["canonical_id"]),
                "issues",
            )
            for rid in ids
            if rid in review_map
        ]

        issue_payloads.append(
            {
                "canonical_id": issue["canonical_id"],
                "canonical_name": issue.get("canonical_name"),
                "category": issue.get("category"),
                "metrics": {
                    "review_count": issue.get("review_count"),
                    "share_of_successful_reviews": issue.get(
                        "share_of_successful_reviews"
                    ),
                    "average_rating": issue.get("average_rating"),
                    "negative_rating_share": issue.get(
                        "negative_rating_share"
                    ),
                    "rating_sample_size": issue.get(
                        "rating_sample_size"
                    ),
                },
                "sample_size": len(examples),
                "reviews": examples,
            }
        )

    feature_payloads = []

    for feature in selected_features:
        count = int(feature.get("review_count") or 0)
        sample_size = _sample_count(count)

        ids = _select_diverse_review_ids(
            list(feature.get("review_ids") or []),
            review_map,
            sample_size,
        )

        examples = [
            _review_payload(
                rid,
                review_map[rid],
                analyses.get(rid, {}),
                str(feature["canonical_id"]),
                "feature_requests",
            )
            for rid in ids
            if rid in review_map
        ]

        feature_payloads.append(
            {
                "canonical_id": feature["canonical_id"],
                "canonical_name": feature.get("canonical_name"),
                "category": feature.get("category"),
                "metrics": {
                    "review_count": feature.get("review_count"),
                    "share_of_successful_reviews": feature.get(
                        "share_of_successful_reviews"
                    ),
                },
                "sample_size": len(examples),
                "reviews": examples,
            }
        )

    return {
        "analysis_summary": metrics.get(
            "analysis_summary",
            {},
        ),
        "coverage_metrics": metrics.get(
            "coverage_metrics",
            {},
        ),
        "aspect_metrics": metrics.get(
            "aspect_metrics",
            [],
        ),
        "selection": {
            "minimum_issue_support": minimum_support,
            "max_issues": max_issues,
            "max_feature_requests": max_feature_requests,
            "review_sampling_rule": (
                "min(issue_count, clamp(floor(issue_count * 0.10), 5, 15))"
            ),
        },
        "issues": issue_payloads,
        "feature_requests": feature_payloads,
    }


def _fingerprint(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_generated_ids(
    response: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    expected_issues = {
        str(x["canonical_id"])
        for x in payload.get("issues", [])
    }

    returned_issues = {
        str(x["canonical_id"])
        for x in response.get("issue_insights", [])
    }

    if returned_issues != expected_issues:
        raise ValueError(
            "LLM issue insight IDs do not match selected issues"
        )

    expected_features = {
        str(x["canonical_id"])
        for x in payload.get("feature_requests", [])
    }

    returned_features = {
        str(x["canonical_id"])
        for x in response.get(
            "feature_request_insights",
            [],
        )
    }

    if returned_features != expected_features:
        raise ValueError(
            "LLM feature request IDs do not match selected requests"
        )

    allowed_review_ids = {}

    for item in payload.get("issues", []):
        allowed_review_ids[str(item["canonical_id"])] = {
            str(review["review_id"])
            for review in item.get("reviews", [])
        }

    for insight in response.get("issue_insights", []):
        cid = str(insight["canonical_id"])

        returned = {
            str(x)
            for x in insight.get(
                "supporting_review_ids",
                [],
            )
        }

        if not returned <= allowed_review_ids.get(
            cid,
            set(),
        ):
            raise ValueError(
                f"Insight {cid} references unknown reviews"
            )

    for item in payload.get(
        "feature_requests",
        [],
    ):
        allowed_review_ids[str(item["canonical_id"])] = {
            str(review["review_id"])
            for review in item.get("reviews", [])
        }

    for insight in response.get(
        "feature_request_insights",
        [],
    ):
        cid = str(insight["canonical_id"])

        returned = {
            str(x)
            for x in insight.get(
                "supporting_review_ids",
                [],
            )
        }

        if not returned <= allowed_review_ids.get(
            cid,
            set(),
        ):
            raise ValueError(
                f"Feature insight {cid} references unknown reviews"
            )


def _merge_deterministic_metrics(
    generated: dict[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    issue_inputs = {
        str(row["canonical_id"]): row
        for row in payload.get("issues", [])
    }

    feature_inputs = {
        str(row["canonical_id"]): row
        for row in payload.get(
            "feature_requests",
            [],
        )
    }

    issue_results = []

    for insight in generated.get(
        "issue_insights",
        [],
    ):
        source = issue_inputs[str(insight["canonical_id"])]

        issue_results.append(
            {
                "canonical_id": source["canonical_id"],
                "canonical_name": source.get(
                    "canonical_name"
                ),
                "category": source.get("category"),
                "metrics": source.get("metrics", {}),
                "sample_size": source.get(
                    "sample_size"
                ),
                **insight,
            }
        )

    feature_results = []

    for insight in generated.get(
        "feature_request_insights",
        [],
    ):
        source = feature_inputs[
            str(insight["canonical_id"])
        ]

        feature_results.append(
            {
                "canonical_id": source["canonical_id"],
                "canonical_name": source.get(
                    "canonical_name"
                ),
                "category": source.get("category"),
                "metrics": source.get("metrics", {}),
                "sample_size": source.get(
                    "sample_size"
                ),
                **insight,
            }
        )

    return {
        "model": MODEL_ID,
        "selection": payload.get(
            "selection",
            {},
        ),
        "analysis_summary": payload.get(
            "analysis_summary",
            {},
        ),
        "coverage_metrics": payload.get(
            "coverage_metrics",
            {},
        ),
        "overall_summary": generated[
            "overall_summary"
        ],
        "issue_insights": issue_results,
        "feature_request_insights": feature_results,
        "limitations": generated.get(
            "limitations",
            [],
        ),
    }


async def generate_saved_insights(
    output_dir: str | Path,
    *,
    max_issues: int = 8,
    max_feature_requests: int = 5,
    max_cost_usd: float | None = None,
    client: OpenRouterClient | None = None,
) -> dict[str, Any]:
    folder = Path(output_dir)

    reviews = _read_json(
        folder / "reviews.json",
        None,
    )
    analyses = _read_json(
        folder / "review_analyses.json",
        None,
    )
    metrics = _read_json(
        folder / "nlp_metrics.json",
        None,
    )

    if reviews is None:
        raise FileNotFoundError(
            "reviews.json is missing"
        )

    if analyses is None:
        raise FileNotFoundError(
            "review_analyses.json is missing"
        )

    if metrics is None:
        raise FileNotFoundError(
            "nlp_metrics.json is missing"
        )

    payload = build_insights_input(
        reviews,
        analyses,
        metrics,
        max_issues=max_issues,
        max_feature_requests=max_feature_requests,
    )

    input_hash = _fingerprint(payload)

    output_path = folder / "insights.json"
    input_path = folder / "insights_input.json"

    previous = _read_json(
        output_path,
        None,
    )

    if (
        isinstance(previous, Mapping)
        and previous.get("input_hash")
        == input_hash
        and previous.get("model") == MODEL_ID
    ):
        return dict(previous)

    _write_json(
        input_path,
        payload,
    )

    own_client = client is None

    if own_client:
        state = _read_json(
            folder / "scan_state.json",
            {"api_usage": {}},
        )

        prior_cost = sum(
            float(stage.get("cost_usd", 0.0))
            for stage in state.get(
                "api_usage",
                {},
            ).values()
            if isinstance(stage, Mapping)
        )

        client = OpenRouterClient(
            max_cost_usd=max_cost_usd,
            prior_cost_usd=prior_cost,
        )

    assert client is not None

    async def run() -> dict[str, Any]:
        generated = await client.json_completion(
            schema_name="actionable_insights_v1",
            schema=INSIGHTS_SCHEMA,
            system=SYSTEM_PROMPT,
            user=json.dumps(
                payload,
                ensure_ascii=False,
            ),
            max_tokens=100_000,
            reasoning_max_tokens=50_000,
        )

        _validate_generated_ids(
            generated,
            payload,
        )

        result = _merge_deterministic_metrics(
            generated,
            payload,
        )

        result["input_hash"] = input_hash

        _write_json(
            output_path,
            result,
        )

        return result

    if own_client:
        assert client is not None

        async with client:
            result = await run()

        state_path = folder / "scan_state.json"

        state = _read_json(
            state_path,
            {"api_usage": {}},
        )

        state.setdefault(
            "api_usage",
            {},
        )

        state["api_usage"]["insights"] = (
            client.usage.as_dict()
        )

        _write_json(
            state_path,
            state,
        )

        return result

    return await run()


def run_saved_insights(
    *args,
    **kwargs,
) -> dict[str, Any]:
    return asyncio.run(
        generate_saved_insights(
            *args,
            **kwargs,
        )
    )
