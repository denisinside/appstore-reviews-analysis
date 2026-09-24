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

All methods return JSON-serializable dictionaries. Discovery returns country statistics and errors without full review texts. Country returns reviews, collection status, page counts, and errors. Top returns a combined review list and separate country statistics. Every review has `review_id`, `app_id`, `country`, `observed_countries`, `title`, `text`, `rating`, `app_version`, `updated_at`, `language`, and `language_confidence`. Missing optional values are `null`; unknown languages are `und`. A failed country has `review_count: null` to distinguish it from a successful empty result.

Discovery ranks fully checked countries ahead of partial and failed ones, then sorts by accessible unique review count, newest available review, and country code. Top selects only fully checked countries and reports partial status if the discovery or a selected country had errors.

## Cache

Without configuration, the cache lives in `.cache/` in the current working directory and persists across runs. Set both `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` to use Upstash Redis over REST. If Upstash fails, the process logs the error and uses the file cache. `.env.example` lists optional variables; the CLI reads environment variables supplied by your shell and does not automatically load `.env`.

Successful populated RSS pages and complete Discovery results are cached for 21,600 seconds. Valid empty pages are cached for 300 seconds. Failed requests, invalid JSON, and partial Discovery results are not cached as success. `--force-refresh` bypasses current results and refreshes RSS pages. Top reuses the pages just fetched during its Discovery pass.

## Limits

Apple RSS exposes at most 10 pages per storefront, usually up to roughly 50 reviews per page. Counts are **accessible unique reviews**, not the app's total review counts. Pages may be empty between populated pages, so the collector checks every requested page. A numeric app ID that does not exist may still produce a valid empty feed; `success` means the feed was fetched, not that the app's existence was verified. Apple can return fewer reviews, missing fields, errors, or rate limits. The collector makes at most three concurrent requests and up to three attempts for transient failures; it does not bypass Apple limits. Language identification is a best-effort prediction based on the original title and text. No sentiment analysis or continuous monitoring is included.

Run offline tests with `python -m pytest -q`.
