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
    return api


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
