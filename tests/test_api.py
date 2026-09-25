"""HTTP regression checks for API input errors."""

import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_module(tmp_path, monkeypatch):
    monkeypatch.setenv("APPSTORE_SCAN_DIR", str(tmp_path))
    from appstore_reviews import api

    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    yield api
    api.scan_runtime.configure()


def _stub_collection(api_module, monkeypatch):
    reviews = [{"review_id": "r1", "app_id": "123", "rating": 2, "text": "Broken", "country": "us"}]

    class StubScraper:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get_top_reviews(self, *_args, **_kwargs):
            return {"reviews": reviews, "status": "ok", "errors": []}

    monkeypatch.setattr(api_module, "AppStoreReviews", StubScraper)
    monkeypatch.setattr(api_module, "_lookup_app_name", lambda _app_id: "Test App")
    return reviews


def test_invalid_app_ids_and_country_return_client_errors(api_module):
    client = TestClient(api_module.app, raise_server_exceptions=False)

    response = client.get("/api/apps/invalid/discovery")
    assert response.status_code == 422
    assert "app_id" in response.json()["detail"]

    response = client.get("/api/apps/0/discovery")
    assert response.status_code == 422
    assert "app_id" in response.json()["detail"]

    response = client.post("/api/scans", json={"app_id": "0", "mode": "country", "country": "us", "max_pages": 1})
    assert response.status_code == 422
    assert "app_id" in response.json()["detail"]

    response = client.post("/api/scans", json={"app_id": "324684580", "mode": "country", "country": "zz", "max_pages": 1})
    assert response.status_code == 422
    assert "country" in response.json()["detail"]


def test_scan_not_found_and_insights_pending(api_module, tmp_path):
    client = TestClient(api_module.app)
    assert client.get("/api/scans/not-a-scan").status_code == 404

    folder = tmp_path / "abcdef123456"
    folder.mkdir()
    (folder / "scan.json").write_text('{"scan_id":"abcdef123456","analysis_status":"not_started"}', encoding="utf-8")
    response = client.get("/api/scans/abcdef123456/insights")
    assert response.status_code == 409
    assert response.json()["detail"] == "Insights are not available yet"


def test_discovery_endpoint_returns_collector_result(api_module, monkeypatch):
    class StubScraper:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def discovery(self, app_id, *, force_refresh=False):
            assert app_id == "324684580"
            assert force_refresh is True
            return {"app_id": app_id, "mode": "discovery", "countries": [], "errors": []}

    monkeypatch.setattr(api_module, "AppStoreReviews", StubScraper)
    response = TestClient(api_module.app).get("/api/apps/324684580/discovery?force_refresh=true")
    assert response.status_code == 200
    assert response.json() == {"app_id": "324684580", "mode": "discovery", "countries": [], "errors": []}


def test_atomic_json_write_survives_parallel_status_updates(api_module, tmp_path):
    path = tmp_path / "scan_state.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: api_module._write_json(path, {"value": value}), range(80)))
    assert json.loads(path.read_text(encoding="utf-8"))["value"] in range(80)
    assert not list(tmp_path.glob("*.tmp"))


def test_scan_history_returns_compact_sorted_summaries(api_module, tmp_path):
    client = TestClient(api_module.app)
    for scan_id, created_at, rating in (
        ("abcdef123456", "2026-09-24T10:00:00Z", 3.5),
        ("123456abcdef", "2026-09-25T10:00:00Z", 4.25),
    ):
        folder = tmp_path / scan_id
        folder.mkdir()
        (folder / "scan.json").write_text(json.dumps({
            "scan_id": scan_id, "app_id": "324684580", "created_at": created_at,
            "analysis_status": "completed", "collection_mode": "top", "top_n": 10,
            "review_count": 2,
        }), encoding="utf-8")
        (folder / "basic_metrics.json").write_text(json.dumps({"average_rating": rating}), encoding="utf-8")
        (folder / "sentiment_results.json").write_text(json.dumps([
            {"review_id": "a", "sentiment": "negative"},
            {"review_id": "b", "sentiment": "positive"},
        ]), encoding="utf-8")
        (folder / "issue_catalog.json").write_text(json.dumps([{"canonical_id": "one"}]), encoding="utf-8")

    response = client.get("/api/scans")
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["scan_id"] for item in items] == ["123456abcdef", "abcdef123456"]
    assert items[0]["app_name"] == "App 324684580"
    assert items[0]["average_rating"] == 4.25
    assert items[0]["negative_sentiment_share"] == 0.5
    assert items[0]["issue_count"] == 1
    assert "reviews" not in items[0]


def test_download_analysis_report_uses_saved_artifacts_and_keeps_raw_download(api_module, tmp_path):
    folder = tmp_path / "abcdef123456"
    folder.mkdir()
    artifacts = {
        "scan.json": {"scan_id": "abcdef123456", "app_id": "123", "created_at": "2026-01-01T00:00:00Z", "collection_mode": "top", "analysis_status": "completed", "review_count": 1},
        "reviews.json": [{"review_id": "r1", "title": "Bad", "text": "Broken", "rating": 1}],
        "basic_metrics.json": {"average_rating": 1.0},
        "sentiment_results.json": [{"review_id": "r1", "sentiment": "negative", "sentiment_scores": {"negative": 0.9}}],
        "keyword_results.json": [{"review_id": "r1", "keywords": [{"text": "broken"}]}],
        "review_analyses.json": {"r1": {"aspects": [{"aspect": "Playback"}]}},
        "nlp_metrics.json": {"languages": 1},
        "insights.json": {"summary": "Playback issues"},
        "issue_catalog.json": [{"canonical_id": "i1"}],
        "feature_request_catalog.json": [],
    }
    for name, value in artifacts.items():
        (folder / name).write_text(json.dumps(value), encoding="utf-8")

    client = TestClient(api_module.app)
    report_response = client.get("/api/scans/abcdef123456/report/download")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["basic_metrics"]["average_rating"] == 1.0
    assert report["insights"]["summary"] == "Playback issues"
    assert report["reviews"][0]["sentiment"] == "negative"
    assert report["reviews"][0]["sentiment_scores"]["negative"] == 0.9
    assert report["reviews"][0]["keywords"][0]["text"] == "broken"
    assert report["reviews"][0]["issue_analysis"]["aspects"][0]["aspect"] == "Playback"

    raw_response = client.get("/api/scans/abcdef123456/reviews/download")
    assert raw_response.status_code == 200
    assert raw_response.json() == artifacts["reviews.json"]


def test_pdf_report_uses_frontend_and_returns_named_pdf(api_module, tmp_path, monkeypatch):
    folder = tmp_path / "abcdef123456"
    folder.mkdir()
    (folder / "scan.json").write_text(
        json.dumps({"scan_id": "abcdef123456", "app_id": "123"}), encoding="utf-8"
    )
    observed = {}

    class FakePage:
        def goto(self, url, **kwargs):
            observed["url"] = url
            return types.SimpleNamespace(ok=True)

        def wait_for_selector(self, selector, **kwargs):
            observed["selector"] = selector

        def evaluate(self, _script):
            return None

        def wait_for_function(self, _script, **kwargs):
            return None

        def pdf(self, *, path, **kwargs):
            observed["pdf_options"] = kwargs
            (folder / Path(path).name).write_bytes(b"%PDF-1.4 fake")

    class FakeBrowser:
        def new_page(self):
            return FakePage()

        def close(self):
            pass

    class FakeChromium:
        def launch(self, **kwargs):
            observed["launch"] = kwargs
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    playwright_module = types.ModuleType("playwright")
    sync_api_module = types.ModuleType("playwright.sync_api")
    sync_api_module.Error = RuntimeError
    sync_api_module.sync_playwright = lambda: FakePlaywright()
    playwright_module.sync_api = sync_api_module
    monkeypatch.setitem(sys.modules, "playwright", playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api_module)
    monkeypatch.setenv("FRONTEND_URL", "http://frontend.test/")

    response = TestClient(api_module.app).get("/api/scans/abcdef123456/report/download.pdf")
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].endswith(
        'filename="app_123_abcdef123456_analysis_report.pdf"'
    )
    assert observed["url"] == "http://frontend.test/scans/abcdef123456/report"
    assert observed["selector"] == '[data-report-ready="true"]'
    assert observed["pdf_options"]["format"] == "A4"
    assert observed["pdf_options"]["print_background"] is True


def test_remote_dispatch_commits_queued_before_worker_and_completion_is_readable(api_module, tmp_path, monkeypatch):
    reviews = _stub_collection(api_module, monkeypatch)
    events = []
    queued_job = []

    def dispatch(scan_id, payload):
        folder = tmp_path / scan_id
        assert json.loads((folder / "scan.json").read_text())["analysis_status"] == "queued"
        assert json.loads((folder / "reviews.json").read_text()) == reviews
        assert events[-1] == "commit:queued"
        queued_job.append((scan_id, payload))

    def commit():
        statuses = [json.loads(path.read_text())["analysis_status"] for path in tmp_path.glob("*/scan.json")]
        events.append(f"commit:{statuses[-1]}")

    api_module.scan_runtime.configure(reload_volume=lambda: events.append("reload"), commit_volume=commit, dispatch=dispatch)
    client = TestClient(api_module.app)
    created = client.post("/api/scans", json={"app_id": "123", "mode": "top"})
    assert created.status_code == 201
    scan_id = created.json()["scan_id"]
    assert client.get(f"/api/scans/{scan_id}").json()["analysis_status"] == "not_started"
    queued = client.post(f"/api/scans/{scan_id}/analyze", json={})
    assert queued.status_code == 202
    assert queued.json() == {"scan_id": scan_id, "analysis_status": "queued"}
    assert client.get(f"/api/scans/{scan_id}").json()["analysis_status"] == "queued"
    assert client.post(f"/api/scans/{scan_id}/analyze", json={}).status_code == 409

    def pipeline(rows, folder, **kwargs):
        assert rows == reviews
        assert json.loads((folder / "scan.json").read_text())["analysis_status"] == "running"
        assert "commit:running" in events
        kwargs["progress_callback"]("issue_extraction", 0.5, 1, 2)
        running = client.get(f"/api/scans/{scan_id}").json()
        assert running["progress"]["percent"] >= 25
        assert running["progress"]["completed"] == 1
        api_module._write_json(folder / "nlp_metrics.json", {"processed": 1})

    monkeypatch.setattr(api_module, "run_full_pipeline", pipeline)
    monkeypatch.setattr(api_module, "run_saved_insights", lambda folder, **_kwargs: api_module._write_json(folder / "insights.json", {"summary": "Done"}))
    api_module._run_analysis_job(scan_id, api_module.AnalyzeRequest.model_validate(queued_job[0][1]))
    assert events[-1] == "commit:completed"
    completed = client.get(f"/api/scans/{scan_id}").json()
    assert completed["analysis_status"] == "completed"
    assert completed["progress"]["percent"] == 100
    assert client.get(f"/api/scans/{scan_id}/metrics").json()["nlp"] == {"processed": 1}
    assert client.get(f"/api/scans/{scan_id}/insights").json() == {"summary": "Done"}
    assert client.get(f"/api/scans/{scan_id}/reviews/download").json() == reviews


def test_worker_failure_persists_failed_state(api_module, tmp_path, monkeypatch):
    _stub_collection(api_module, monkeypatch)
    commits = []
    api_module.scan_runtime.configure(commit_volume=lambda: commits.append(True), dispatch=lambda *_: None)
    client = TestClient(api_module.app)
    scan_id = client.post("/api/scans", json={"app_id": "123"}).json()["scan_id"]
    assert client.post(f"/api/scans/{scan_id}/analyze", json={}).status_code == 202

    def fail(*_args, **_kwargs):
        raise RuntimeError("pipeline stopped")

    monkeypatch.setattr(api_module, "run_full_pipeline", fail)
    api_module._run_analysis_job(scan_id, api_module.AnalyzeRequest())
    scan = client.get(f"/api/scans/{scan_id}").json()
    assert scan["analysis_status"] == "failed"
    assert scan["error"] == "RuntimeError: pipeline stopped"
    assert len(commits) == 4  # creation, queued, running, failed


def test_local_background_tasks_still_process_analysis(api_module, monkeypatch):
    _stub_collection(api_module, monkeypatch)
    monkeypatch.setattr(api_module, "run_full_pipeline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_module, "run_saved_insights", lambda *_args, **_kwargs: None)
    client = TestClient(api_module.app)
    scan_id = client.post("/api/scans", json={"app_id": "123"}).json()["scan_id"]
    assert client.post(f"/api/scans/{scan_id}/analyze", json={}).status_code == 202
    assert client.get(f"/api/scans/{scan_id}").json()["analysis_status"] == "completed"


def test_queued_scan_can_be_stopped_and_resumed(api_module, monkeypatch, tmp_path):
    _stub_collection(api_module, monkeypatch)
    dispatched = []
    api_module.scan_runtime.configure(dispatch=lambda scan_id, payload: dispatched.append((scan_id, payload)))
    client = TestClient(api_module.app)
    scan_id = client.post("/api/scans", json={"app_id": "123"}).json()["scan_id"]
    assert client.post(f"/api/scans/{scan_id}/analyze", json={}).status_code == 202
    assert client.post(f"/api/scans/{scan_id}/stop").json()["analysis_status"] == "cancelling"
    assert client.get(f"/api/scans/{scan_id}").json()["analysis_status"] == "cancelling"
    monkeypatch.setattr(api_module, "run_full_pipeline", lambda *_args, **_kwargs: pytest.fail("queued scan ran"))
    api_module._run_analysis_job(scan_id, api_module.AnalyzeRequest())
    stopped = client.get(f"/api/scans/{scan_id}").json()
    assert stopped["analysis_status"] == "interrupted"
    assert stopped["progress"]["percent"] == 0
    assert client.post(f"/api/scans/{scan_id}/stop").status_code == 409
    assert client.post(f"/api/scans/{scan_id}/analyze", json={}).status_code == 202
    assert not (tmp_path / scan_id / "stop.requested").exists()


def test_running_scan_stop_preserves_partial_results(api_module, monkeypatch, tmp_path):
    _stub_collection(api_module, monkeypatch)
    api_module.scan_runtime.configure(dispatch=lambda *_args: None)
    client = TestClient(api_module.app)
    scan_id = client.post("/api/scans", json={"app_id": "123"}).json()["scan_id"]
    client.post(f"/api/scans/{scan_id}/analyze", json={})

    def pipeline(_reviews, folder, **kwargs):
        api_module._write_json(folder / "review_analyses.json", {
            "r1": {"review_id": "r1", "status": "success", "aspects": []}})
        assert client.post(f"/api/scans/{scan_id}/stop").status_code == 202
        kwargs["progress_callback"]("issue_extraction", 0.5, 1, 2)
        pytest.fail("worker continued after stop")

    monkeypatch.setattr(api_module, "run_full_pipeline", pipeline)
    monkeypatch.setattr(api_module, "run_saved_insights", lambda *_args, **_kwargs: pytest.fail("insights ran"))
    api_module._run_analysis_job(scan_id, api_module.AnalyzeRequest())
    stopped = client.get(f"/api/scans/{scan_id}").json()
    assert stopped["analysis_status"] == "interrupted"
    assert stopped["progress"]["percent"] >= 25
    assert stopped["available_results"]["extracted_reviews"] == 1
    assert client.get(f"/api/scans/{scan_id}/reviews").json()["items"][0]["issue_analysis"]["status"] == "success"
    assert client.get(f"/api/scans/{scan_id}/report/download").status_code == 200
