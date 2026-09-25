# App Store reviews collector

A small Python 3.11+ package for one-off collection of public Apple App Store RSS reviews. It supports 32 fixed storefront countries and pages 1–10. It uses `httpx`, fastText language identification, and a six-hour cache.

## Install

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
New-Item -ItemType Directory -Force models
Invoke-WebRequest https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -OutFile models/lid.176.ftz
```

On macOS/Linux, activate with `source .venv/bin/activate` and download the model with `curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -o models/lid.176.ftz`. The model is downloaded once, manually, and is excluded from Git. Set `FASTTEXT_MODEL_PATH` if it is stored elsewhere. Discovery does not need the model; Country and Top do. If the model is missing or cannot load, those modes report a setup error.

## Use

```powershell
python -m appstore_reviews discovery --app-id 389801252
python -m appstore_reviews country --app-id 389801252 --country ua
python -m appstore_reviews top --app-id 389801252 --top 10
python -m appstore_reviews country --app-id 389801252 --country us --max-pages 2 --force-refresh --output reviews.json
```

`--output` writes UTF-8 JSON to a file; otherwise JSON goes to stdout with non-ASCII characters escaped for compatibility with Windows console encodings. Parsing the JSON restores the original text. Logs and input/setup errors go to stderr. Exit code 1 means a partial or failed collection; exit code 2 means an invalid input or setup error. For a real RSS smoke test after setup, run `python -m appstore_reviews country --app-id 389801252 --country us --max-pages 1 --force-refresh --output smoke.json` and inspect `smoke.json`, its `status`, and `errors`.

Use from Python:

```python
from appstore_reviews import AppStoreReviews

with AppStoreReviews() as scraper:
    discovery = scraper.discovery(app_id="389801252")
    reviews_ua = scraper.get_reviews(app_id="389801252", country="ua")
    top_reviews = scraper.get_top_reviews(app_id="389801252", top_n=10)
```

For another Python project, install this directory with `python -m pip install -e path/to/appstore-reviews-analysis` (or a regular `pip install` from that path). The package dependencies are declared in `pyproject.toml`.

### Prepare text for later analysis

```python
from appstore_reviews import prepare_review_text

analysis_text = prepare_review_text(reviews_ua["reviews"][0])
```

This optional step combines the title and review body with a newline, applies Unicode NFC, and collapses repeated spaces, tabs, and line breaks within each part. It returns a string without adding a field or changing the original `title` and `text`. For example, `"и\u0306"` (two Unicode code points) becomes `"й"` (one code point) while retaining the same visible letter. Use `normalize_text` if you need to process one string separately.

### Calculate statistical metrics

```python
from appstore_reviews import calculate_metrics

metrics = calculate_metrics(top_reviews["reviews"])
```

The function accepts a list of review dictionaries and returns JSON-serializable statistics: `review_count`, `average_rating`, `rating_distribution` (counts and shares for ratings 1–5), `negative_rating_share` (ratings 1–2), `country_statistics`, `language_statistics`, and `rating_by_version`. Shares range from 0 to 1. A missing country, language, or app version appears as `null` in its group. For an empty list, averages and shares are `null`.

Pass the deduplicated `reviews` list returned by Country or Top. Global metrics count each review once. Country statistics include a review in every `observed_countries` market, so country counts can sum to more than `review_count`. The function leaves the input unchanged and raises `ValueError` if a rating is missing or outside 1–5.

All methods return JSON-serializable dictionaries. Discovery returns country statistics and errors without full review texts. Country returns reviews, collection status, page counts, and errors. Top returns a combined review list and separate country statistics. Every review has `review_id`, `app_id`, `country`, `observed_countries`, `title`, `text`, `rating`, `app_version`, `updated_at`, `language`, and `language_confidence`. Missing optional values are `null`; unknown languages are `und`. A failed country has `review_count: null` to distinguish it from a successful empty result.

Discovery ranks fully checked countries ahead of partial and failed ones, then sorts by accessible unique review count, newest available review, and country code. Top selects only fully checked countries and reports partial status if the discovery or a selected country had errors.

## Cache

Without configuration, the cache lives in `.cache/` in the current working directory and persists across runs. Set both `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` to use Upstash Redis over REST. If Upstash fails, the process logs the error and uses the file cache. `.env.example` lists optional variables; the CLI reads environment variables supplied by your shell and does not automatically load `.env`.

Successful populated RSS pages and complete Discovery results are cached for 21,600 seconds. Valid empty pages are cached for 300 seconds. Failed requests, invalid JSON, and partial Discovery results are not cached as success. `--force-refresh` bypasses current results and refreshes RSS pages. Top reuses the pages just fetched during its Discovery pass.

## Limits

Apple RSS exposes at most 10 pages per storefront, usually up to roughly 50 reviews per page. Counts are **accessible unique reviews**, not the app's total review counts. Pages may be empty between populated pages, so the collector checks every requested page. A numeric app ID that does not exist may still produce a valid empty feed; `success` means the feed was fetched, not that the app's existence was verified. Apple can return fewer reviews, missing fields, errors, or rate limits. The collector makes at most three concurrent requests and up to three attempts for transient failures; it does not bypass Apple limits. Language identification is a best-effort prediction based on the original title and text. Continuous monitoring is not included. Sentiment analysis is an optional, separate step described below.

Run offline tests with `python -m pytest -q`.

## Optional sentiment analysis

Install the additional dependencies only when you need sentiment inference:

```powershell
python -m pip install -e ".[sentiment]"
```

```python
from appstore_reviews import SentimentAnalyzer

analyzer = SentimentAnalyzer(device="cpu")
results = analyzer.analyze_reviews(top_reviews["reviews"])
single = analyzer.analyze_review(top_reviews["reviews"][0])
```

The analyzer combines the original title and text through `prepare_review_text`. It never uses the numerical rating, changes the original review, or loads a model during ordinary parsing. The first sentiment call downloads the pinned [Cardiff XLM-RoBERTa sentiment checkpoint](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) (about 1.11 GB) into the Hugging Face cache and then reuses it in memory. The default revision is `f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8`, exactly the checkpoint used in Model Arena. Each result contains `review_id`, `sentiment` (`positive`, `neutral`, or `negative`), `sentiment_scores` for all three classes, `sentiment_model` with model ID and revision, and `error`. Empty text yields a null sentiment and an error. Long text is truncated to 256 tokens. Inference is batched; failures in one batch are retried per review.

Cardiff was selected for this experimental integration because it had the highest multilingual and Ukrainian Macro-F1 and the best Ukrainian and English negative-class recall among the three tested checkpoints. The Cardiff checkpoint does **not** declare a license on its model card, so permission for commercial use is **not confirmed**. Clarify usage rights before any commercial deployment. The [arena report](models_arena/sentiment/REPORT.md) also documents model errors and limits of the reference labels. The evaluation data stays in `models_arena` and is never used at inference time.

## Optional keyword extraction

Install the local multilingual keyword model dependencies separately:

```powershell
python -m pip install -e ".[keywords]"
```

For the full sentiment-to-keyword pipeline, install both optional groups: `python -m pip install -e ".[sentiment,keywords]"`.

```python
from appstore_reviews import KeywordExtractor, analyze_negative_keywords

extractor = KeywordExtractor()
one = extractor.extract_review(top_reviews["reviews"][0])
batch = extractor.extract_reviews(top_reviews["reviews"])
analysis = analyze_negative_keywords(top_reviews["reviews"])
```

`KeywordExtractor` can process any valid review, regardless of sentiment. It combines `title` and `text` (or a caller's `content` alias) with the existing `prepare_review_text` helper, uses the review's language, and ranks up to five source phrases per review. The candidate rules allow one to six words, retain numeric restrictions and negation, and avoid crossing sentence boundaries. Each keyword has `text`, a model similarity `score`, `evidence_span` (the surrounding clause), and `source_start`/`source_end` offsets in the normalized title-plus-body analysis text. The original review remains unchanged. Extracted keywords are source keyphrases, not automatically confirmed product issues or per-phrase negative labels; inspect the evidence and original review text before interpreting or counting them. Empty text returns an empty keyword list without loading a model. The optional checkpoint loads once and is reused across batch calls; ratings and Model Arena files are not used.

`analyze_negative_keywords` runs the existing Cardiff sentiment analyzer first, sends only reviews predicted `negative` to the keyword extractor, then returns both per-review outputs and `common_keywords_by_language`. Filtering is at the review level: a negative prediction selects the whole review for extraction, but does not label each extracted phrase negative. Each common phrase reports a count of unique review IDs, its share among successfully analyzed negative reviews in that language, mean similarity score, and example IDs. Surface normalization merges case, punctuation, and whitespace variants; it deliberately avoids semantic clustering that could merge different complaints. Inspect evidence and original review text before treating phrases as product issues. The output has separate language lists. This step is a local, explicit function call, not a background task.

The v2 default is [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), pinned to revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. It uses the `query: ` prefix for both review text and candidate phrases, as configured for the v2 benchmark, with a limit of 120 candidates and five returned phrases per review. See the [keyword arena report](models_arena/keyword_extraction/REPORT.md) for comparisons with BGE-M3, MiniLM, and TF-IDF. Its published v2 quality scores precede the final local phrase fixes; the final production F1 has not been independently measured. Extraction remains experimental: source keyphrases need evidence and original-text inspection before they can support product issue counts. The end-to-end pipeline also uses the Cardiff sentiment checkpoint, whose commercial permission is unconfirmed as noted above. The E5 checkpoint is downloaded once to the Hugging Face cache on first use. These optional dependencies are not needed for RSS collection or statistical metrics.
