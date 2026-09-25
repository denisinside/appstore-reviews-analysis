import json

import httpx
import pytest

from appstore_reviews import AppStoreReviews, ModelLoadError
from appstore_reviews.cache import FallbackCache, FileCache
from appstore_reviews.config import SUPPORTED_COUNTRIES
from appstore_reviews.language import LanguageDetector
import appstore_reviews.scraper as scraper_module
import appstore_reviews.__main__ as cli_module


class StubLanguage:
    def ensure_loaded(self):
        pass

    def detect(self, title, text):
        return ("en", 0.9) if title or text else ("und", None)


def review(review_id="101", *, rating="5", updated="2026-09-20T12:00:00+00:00", title="Good", text="Useful app"):
    entry = {
        "id": {"label": review_id},
        "im:rating": {"label": rating},
        "updated": {"label": updated},
        "title": {"label": title},
        "content": {"label": text},
    }
    return entry


def feed(entries=None):
    return {"feed": {"entry": entries if entries is not None else []}}


def make_scraper(tmp_path, handler, clock=lambda: 0):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AppStoreReviews(
        cache=FileCache(tmp_path / "cache", clock=clock),
        client=client,
        language_detector=StubLanguage(),
        sleep=lambda _: None,
    )


def test_supported_country_count_and_input_validation(tmp_path):
    assert len(SUPPORTED_COUNTRIES) == 32
    scraper = make_scraper(tmp_path, lambda _: pytest.fail("validation must happen before HTTP"))
    for bad in (None, "", "abc", "0", "-1", True):
        with pytest.raises(ValueError, match="app_id"):
            scraper.discovery(bad)
    with pytest.raises(ValueError, match="country"):
        scraper.get_reviews("123", "xx")
    for bad in (0, 11, "2", True):
        with pytest.raises(ValueError, match="max_pages"):
            scraper.get_reviews("123", "us", max_pages=bad)
    for bad in (0, 33, "2", True):
        with pytest.raises(ValueError, match="top_n"):
            scraper.get_top_reviews("123", top_n=bad)


def test_single_entry_metadata_and_missing_fields(tmp_path):
    entries = [
        {"im:name": {"label": "App"}, "id": {"label": "123"}},
        review("101"),
        {"id": {"label": "102"}, "im:rating": {"label": "4"}},
        {"im:rating": {"label": "3"}, "content": {"label": "No ID"}},
        review("103", rating="6"),
    ]
    scraper = make_scraper(tmp_path, lambda _: httpx.Response(200, json=feed(entries)))
    result = scraper.get_reviews("123", "US", max_pages=1)
    assert result["status"] == "success"
    assert [r["review_id"] for r in result["reviews"]] == ["101", "102"]
    first, second = result["reviews"]
    assert first["app_id"] == "123" and first["country"] == "us"
    assert first["updated_at"] == "2026-09-20T12:00:00Z"
    assert first["language"] == "en" and first["rating"] == 5
    assert second["title"] is None and second["text"] is None
    assert second["language"] == "und" and second["updated_at"] is None
    assert second["app_version"] is None


def test_singleton_entry_and_deduplication(tmp_path):
    def handler(request):
        page = int(request.url.path.split("/page=")[1].split("/")[0])
        return httpx.Response(200, json=feed(review("9") if page == 1 else [review("9"), review("10")] ))

    scraper = make_scraper(tmp_path, handler)
    result = scraper.get_reviews("123", "us", max_pages=2)
    assert result["review_count"] == 2
    assert result["reviews"][0]["observed_countries"] == ["us"]


def test_empty_page_between_populated_pages(tmp_path):
    def handler(request):
        page = int(request.url.path.split("/page=")[1].split("/")[0])
        entries = [review(str(page))] if page != 2 else []
        return httpx.Response(200, json=feed(entries))

    scraper = make_scraper(tmp_path, handler)
    result = scraper.get_reviews("123", "us", max_pages=3)
    assert [r["review_id"] for r in result["reviews"]] == ["1", "3"]
    assert result["pages_succeeded"] == 3


def test_partial_and_failed_collections(tmp_path):
    def handler(request):
        page = int(request.url.path.split("/page=")[1].split("/")[0])
        if page == 2:
            return httpx.Response(404)
        return httpx.Response(200, json=feed([review(str(page))]))

    scraper = make_scraper(tmp_path, handler)
    partial = scraper.get_reviews("123", "us", max_pages=3)
    assert partial["status"] == "partial" and partial["pages_succeeded"] == 2
    assert partial["review_count"] == 2 and partial["errors"][0]["page"] == 2
    attempts = []

    def limited(request):
        attempts.append(request)
        return httpx.Response(429)

    failed_scraper = make_scraper(tmp_path / "other", limited)
    failed = failed_scraper.get_reviews("123", "ca", max_pages=1)
    assert failed["status"] == "failed" and failed["review_count"] is None
    assert failed["errors"][0]["error"] == "HTTP 429"
    assert len(attempts) == 3


def test_discovery_sorting_and_failed_country(tmp_path, monkeypatch):
    monkeypatch.setattr(scraper_module, "SUPPORTED_COUNTRIES", ("ua", "us", "ca", "gb"))

    def handler(request):
        country = request.url.path.split("/")[1]
        if country == "gb":
            return httpx.Response(404)
        if country == "ca" and "page=2" in request.url.path:
            return httpx.Response(404)
        if "page=1" not in request.url.path:
            return httpx.Response(200, json=feed())
        dates = {"ua": "2026-09-20T00:00:00Z", "us": "2026-09-21T00:00:00Z", "ca": "2026-09-22T00:00:00Z"}
        return httpx.Response(200, json=feed([review("1", updated=dates[country])]))

    scraper = make_scraper(tmp_path, handler)
    result = scraper.discovery("123")
    assert [c["country"] for c in result["countries"]] == ["us", "ua", "ca", "gb"]
    assert [c["status"] for c in result["countries"]] == ["success", "success", "partial", "failed"]
    assert result["countries"][-1]["review_count"] is None
    assert result["countries"][0]["rank"] == 1
    assert len(result["errors"]) == 11


def test_top_merges_observed_countries_and_uses_discovery_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(scraper_module, "SUPPORTED_COUNTRIES", ("us", "ca"))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        country = request.url.path.split("/")[1]
        if "page=1" not in request.url.path:
            return httpx.Response(200, json=feed())
        entries = [review("1"), review("2")] if country == "us" else [review("1"), review("3")]
        return httpx.Response(200, json=feed(entries))

    scraper = make_scraper(tmp_path, handler)
    discovery = scraper.discovery("123")
    assert len(calls) == 20
    top = scraper.get_top_reviews("123", top_n=2)
    assert len(calls) == 20
    assert top["status"] == "success" and top["review_count"] == 3
    shared = next(r for r in top["reviews"] if r["review_id"] == "1")
    assert shared["country"] == discovery["countries"][0]["country"]
    assert shared["observed_countries"] == [c["country"] for c in top["countries"]]
    assert sum(c["added_review_count"] for c in top["countries"]) == 3


def test_top_keeps_other_country_data_when_one_fails(tmp_path, monkeypatch):
    def handler(request):
        country = request.url.path.split("/")[1]
        if country == "ca":
            return httpx.Response(503)
        return httpx.Response(200, json=feed([review("1")]))

    scraper = make_scraper(tmp_path, handler)
    monkeypatch.setattr(scraper, "discovery", lambda *_, **__: {
        "countries": [
            {"country": "us", "status": "success"},
            {"country": "ca", "status": "success"},
        ],
        "errors": [],
    })
    top = scraper.get_top_reviews("123", top_n=2, max_pages=1)
    assert top["status"] == "partial" and top["review_count"] == 1
    assert [c["status"] for c in top["countries"]] == ["success", "failed"]
    assert top["errors"][0]["country"] == "ca"


def test_partial_discovery_is_not_cached_as_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(scraper_module, "SUPPORTED_COUNTRIES", ("us",))
    calls = []

    def handler(request):
        calls.append(request)
        if "page=2" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json=feed([review("1")]))

    scraper = make_scraper(tmp_path, handler)
    assert scraper.discovery("123")["countries"][0]["status"] == "partial"
    assert scraper.discovery("123")["countries"][0]["status"] == "partial"
    assert len(calls) == 11  # the nine successful pages are reused


def test_file_cache_reused_across_instances_and_expires(tmp_path):
    current = [1000.0]
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json=feed([review("1")]))

    first = make_scraper(tmp_path, handler, clock=lambda: current[0])
    first.get_reviews("123", "us", max_pages=1)
    second = make_scraper(tmp_path, handler, clock=lambda: current[0])
    second.get_reviews("123", "us", max_pages=1)
    assert len(calls) == 1
    current[0] += 21601
    second.get_reviews("123", "us", max_pages=1)
    assert len(calls) == 2
    second.get_reviews("123", "us", max_pages=1, force_refresh=True)
    assert len(calls) == 3


def test_invalid_json_and_feed_are_not_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, text="not JSON")
        if len(calls) == 2:
            return httpx.Response(200, json={"wrong": {}})
        return httpx.Response(200, json=feed([review("1")]))

    scraper = make_scraper(tmp_path, handler)
    assert scraper.get_reviews("123", "us", max_pages=1)["status"] == "failed"
    assert scraper.get_reviews("123", "us", max_pages=1)["status"] == "failed"
    assert scraper.get_reviews("123", "us", max_pages=1)["status"] == "success"
    assert len(calls) == 3


def test_missing_fasttext_model_reports_download_failure(tmp_path, monkeypatch):
    from appstore_reviews import language

    def fail_download(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(language, "urlopen", fail_download)
    detector = LanguageDetector(tmp_path / "missing.ftz")
    with pytest.raises(ModelLoadError, match="Could not download lid.176.ftz"):
        detector.ensure_loaded()
    assert not detector.model_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_fasttext_model_downloads_once(tmp_path, monkeypatch):
    import io
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from appstore_reviews import language

    downloads = []
    monkeypatch.setattr(language, "urlopen", lambda url, timeout: (downloads.append((url, timeout)), io.BytesIO(b"model"))[1])
    monkeypatch.setitem(sys.modules, "fasttext", SimpleNamespace(load_model=lambda path: Path(path).read_bytes()))
    detector = LanguageDetector(tmp_path / "models" / "lid.176.ftz")
    detector.ensure_loaded()
    detector.ensure_loaded()
    LanguageDetector(detector.model_path).ensure_loaded()
    assert detector._model == b"model"
    assert detector.model_path.read_bytes() == b"model"
    assert downloads == [(language.FASTTEXT_MODEL_URL, 60)]


def test_language_thresholds_without_changing_text(tmp_path):
    detector = LanguageDetector(tmp_path / "unused.ftz")

    class FakeModel:
        def __init__(self):
            self.seen = []

        def predict(self, text, k):
            self.seen.append(text)
            return ["__label__uk"], [0.82]

    detector._model = FakeModel()
    assert detector.detect("Hi", None) == ("und", None)
    assert detector.detect("Привіт", "це чудовий додаток") == ("uk", 0.82)
    assert detector._model.seen == ["Привіт це чудовий додаток"]


def test_language_confidence_is_bounded(tmp_path):
    detector = LanguageDetector(tmp_path / "unused.ftz")

    class FakeModel:
        def predict(self, text, k):
            return ["__label__en"], [1.00004]

    detector._model = FakeModel()
    assert detector.detect("A longer title", "and review text") == ("en", 1.0)


def test_upstash_failure_switches_to_file_cache(tmp_path):
    class BrokenPrimary:
        def get(self, _):
            raise ValueError("unavailable")

    cache = FallbackCache(BrokenPrimary(), FileCache(tmp_path))
    assert cache.get("key") is None
    assert cache.primary is None
    cache.set("key", {"value": 1}, 60)
    assert cache.get("key") == {"value": 1}


def test_cli_stdout_is_ascii_safe_json(monkeypatch, capsys):
    class FakeScraper:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get_reviews(self, *_, **__):
            return {"mode": "country", "status": "success", "errors": [], "reviews": [{"text": "Привіт 🫤"}]}

    monkeypatch.setattr(cli_module, "AppStoreReviews", FakeScraper)
    assert cli_module.main(["country", "--app-id", "123", "--country", "ua"]) == 0
    output = capsys.readouterr().out
    assert output.isascii()
    assert json.loads(output)["reviews"][0]["text"] == "Привіт 🫤"
