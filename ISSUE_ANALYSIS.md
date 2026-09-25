# Issue and aspect analysis

`appstore_reviews.analysis_pipeline` adds a reusable GLM-5.3-Flash stage to the
existing one-off App Store review workflow. It leaves the Cardiff sentiment and
E5 keyword models and their settings unchanged. The established local sequence
is sentiment on all reviews, then keywords on negative reviews. The LLM stage
receives the **original title and body of every valid review**; neither keywords
nor star rating are sent as substitutes for review text.

## Run

Install the package dependencies and, for the complete local NLP path, the
existing optional groups:

```powershell
python -m pip install -e ".[sentiment,keywords]"
$env:OPENROUTER_API_KEY = "<your key>"
python -m appstore_reviews analyze --input spotify.jsonl --output-dir scan/spotify --max-cost-usd 1.00
```

`--input` accepts JSONL reviews, a JSON array, or the collector's JSON object
with a `reviews` array. Repeating the same command resumes successful review
analyses and normalization batches from `--output-dir`. `--batch-size` defaults
to 12 (allowed 1–12); `--concurrency` defaults to 48 and
`--requests-per-minute` to 480. `--skip-local-nlp` runs only the new LLM and
deterministic stages, useful when local model dependencies are unavailable.
It does not alter the full pipeline's default behavior. The API key is read
only from the process environment. The CLI does not load `.env` automatically.

To recalculate metrics from saved outputs without API calls:

```powershell
python -m appstore_reviews recalculate-nlp-metrics --output-dir scan/spotify
```

## Extraction schema and taxonomy

The OpenRouter Chat Completions request uses model `z-ai/glm-5.3-flash`,
`response_format.type=json_schema`, `strict=true`, and
`provider.require_parameters=true`. If a provider rejects that parameter,
the client retries in JSON-object mode. Every response is checked against the
local JSON Schema. The [structured-output guide](https://openrouter.ai/docs/guides/features/structured-outputs)
and [current model endpoints](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints)
were checked for parameter support. The schema requires `results[]`, each with `review_id` and
`aspects[]`. Each aspect requires:

```json
{
  "category": "pricing_subscriptions",
  "aspect": "subscription cancellation",
  "sentiment": "negative",
  "evidence": "I cannot cancel my subscription",
  "issues": [{"description": "Cannot cancel subscription", "evidence": "I cannot cancel my subscription"}],
  "feature_requests": []
}
```

The fixed, versioned category IDs are in `config.py`
(`ASPECT_TAXONOMY_VERSION = "1.0"`). The model chooses specific aspect and signal
descriptions dynamically. Per-aspect sentiment remains positive, neutral, or
negative; metrics can report a review as mixed when it contains multiple
sentiments in one category. Issue and feature evidence must be a contiguous
fragment of the original title or body (Unicode NFC and whitespace differences
are accepted). Unsupported evidence is retained with
`evidence_verified=false` and excluded from normalization and counts. Original
title, body, descriptions, and evidence remain in review analyses for audit.

## Batch normalization

Issue and feature requests are processed separately. Python first folds case,
spacing, and minor punctuation in descriptions, keeping negation and numbers;
the deduplication key also includes category and specific aspect. It assigns
deterministic temporary source IDs, then sends unique descriptions and short
evidence samples to GLM in batches. A small set uses one request. If its
grouping violates category boundaries, Python retries by category. Larger
categories are chunked and reconciled in a second pass across provisional
groups. The model supplies only canonical names and source-ID
grouping; Python checks that every source ID appears exactly once, rejects
unjustified cross-category groups, then creates stable hashed canonical IDs.
A final bounded reconciliation checks only plausible equivalent groups that
were assigned different categories; it requires a model explanation plus local
checks on shared terms, negation, and numbers. The chosen category is one of
the input categories and the explanation is saved in the catalog. Distinct
scenarios such as cancellation failure and a charge after cancellation remain
distinct. Successful batch results are cached by input fingerprint; empty
feature sets make no request.

## Saved scan files

The scan directory contains `reviews.json` (input reviews),
`review_analyses.json` (review-ID keyed extraction status, original signals,
grounding, and canonical IDs), `issue_catalog.json`,
`feature_request_catalog.json`, `nlp_metrics.json`, and `scan_state.json`
(separate request/token/actual reported cost totals for extraction and both
normalization kinds). Optional `sentiment_results.json` and
`keyword_results.json` retain prior NLP results. `normalization_cache/`
contains successful LLM grouping responses. Catalog entries include source
formulations, evidence, and deterministic related review IDs, without full
review text. `reviews.json` and the review IDs in metrics allow retrieving 3–5
diverse originals for an issue. Writes use atomic file replacement.

## Metrics and interpretation

`nlp_metrics.py` counts distinct review IDs in Python. Output sections are
`analysis_summary`, `aspect_metrics`, `issue_metrics`,
`feature_request_metrics`, `coverage_metrics`, `country_breakdown`,
`version_breakdown`, and `time_breakdown`. Aspect frequency and sentiment
distribution preserve mixed mentions. Issue rows include frequency, share of
successful reviews, mean available rating, 1–2-star share among rated reviews,
related IDs, and sample IDs. Feature rows include frequency and share. Coverage
includes successful/error counts and the share of negative overall sentiment
reviews containing a concrete issue. Country/version/month buckets use
successful reviews in that bucket as the denominator. Missing metadata is not
imputed. Month buckets describe the collected accessible sample; no population
trend is inferred. Reviews are not a representative sample of all app users.

## Verification

Run `python -m pytest -q tests` for the offline suite. The added tests cover
single and mixed aspects, opposite aspect and overall sentiments, feature-only
requests, vague complaints, English/Ukrainian evidence, invalid JSON,
partial responses, resume, normalization separation and cache reuse, unique
review counts, and metrics recalculation without API calls. A bounded live
scan of eight saved Spotify reviews was also used to check extraction through
metrics and to inspect grounded examples. Its exact request, token, and cost
totals are in that scan's `scan_state.json`.

## Limits

Model findings still require human review. Exact source evidence checks catch
ungrounded quotations but cannot prove that an interpretation is correct.
The spending cap is checked against reported cost before new calls; concurrent
in-flight requests can exceed it by a small amount. If a provider omits
`usage.cost`, the scan records the missing cost report rather than estimating
or claiming a zero-cost scan. The optional local models require their existing
dependencies and checkpoint downloads. The LLM output is not an insights
report, and there is no second scan mode.
