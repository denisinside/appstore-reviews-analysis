"""HTTP regression checks for API input errors."""

import json
from concurrent.futures import ThreadPoolExecutor

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

    def pipeline(rows, folder, **_kwargs):
        assert rows == reviews
        assert json.loads((folder / "scan.json").read_text())["analysis_status"] == "running"
        assert events[-1] == "commit:running"
        api_module._write_json(folder / "nlp_metrics.json", {"processed": 1})

    monkeypatch.setattr(api_module, "run_full_pipeline", pipeline)
    monkeypatch.setattr(api_module, "run_saved_insights", lambda folder, **_kwargs: api_module._write_json(folder / "insights.json", {"summary": "Done"}))
    api_module._run_analysis_job(scan_id, api_module.AnalyzeRequest.model_validate(queued_job[0][1]))
    assert events[-1] == "commit:completed"
    assert client.get(f"/api/scans/{scan_id}").json()["analysis_status"] == "completed"
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
