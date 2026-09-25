import asyncio
import json
from pathlib import Path

import httpx
import pytest

from appstore_reviews.issue_aspects import IssueAspectExtractor, EXTRACTION_SCHEMA
from appstore_reviews.normalization import _cached_completion, normalize_signals
from appstore_reviews.nlp_metrics import calculate_nlp_metrics
from appstore_reviews.openrouter_client import OpenRouterClient, OpenRouterError
from appstore_reviews.analysis_pipeline import analyze_full_pipeline, recalculate_saved_metrics
from appstore_reviews.openrouter_client import split_input_records


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    async def json_completion(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def aspect(category, name, sentiment, evidence, issues=(), features=()):
    return {"category": category, "aspect": name, "sentiment": sentiment,
            "evidence": evidence,
            "issues": [{"description": d, "evidence": e} for d, e in issues],
            "feature_requests": [{"description": d, "evidence": e} for d, e in features]}


@pytest.mark.parametrize("review,aspects", [
    ({"review_id":"1", "title":"Great UI", "text":"I love the interface"},
     [aspect("ui_ux", "interface", "positive", "I love the interface")]),
    ({"review_id":"2", "title":"Mixed", "text":"Beautiful design, but I cannot cancel my subscription."},
     [aspect("ui_ux", "design", "positive", "Beautiful design"), aspect("pricing_subscriptions", "subscription cancellation", "negative", "I cannot cancel my subscription.", [("Cannot cancel subscription", "I cannot cancel my subscription.")])]),
    ({"review_id":"3", "title":"App", "text":"The search is fast but login keeps crashing."},
     [aspect("performance", "search", "positive", "search is fast"), aspect("stability", "login", "negative", "login keeps crashing", [("Login crashes", "login keeps crashing")])]),
    ({"review_id":"4", "title":"Request", "text":"Please add dark mode."},
     [aspect("features", "appearance", "neutral", "add dark mode", features=[("Add dark mode", "Please add dark mode")])]),
    ({"review_id":"5", "title":"Bad", "text":"This app is terrible and disappointing."}, []),
    ({"review_id":"6", "title":"Підписка", "text":"Не можу скасувати підписку після оплати."},
     [aspect("pricing_subscriptions", "скасування підписки", "negative", "Не можу скасувати підписку після оплати.", [("Cannot cancel subscription after payment", "Не можу скасувати підписку після оплати.")])]),
])
def test_extraction_scenarios(review, aspects):
    async def run():
        c = FakeClient([{"results":[{"review_id":review["review_id"], "aspects":aspects}]}])
        result = await IssueAspectExtractor(c, batch_size=8).extract_reviews([review])
        row = result[review["review_id"]]
        assert row["status"] == "success"
        assert row["aspects"] == [{**a, "evidence_verified": True,
            "issues": [{**s, "evidence_verified": True, "canonical_id": None} for s in a["issues"]],
            "feature_requests": [{**s, "evidence_verified": True, "canonical_id": None} for s in a["feature_requests"]]} for a in aspects]
    asyncio.run(run())


def test_evidence_unverified_is_not_rewritten():
    async def run():
        r = {"review_id":"x", "title":"Title", "text":"Actual words"}
        response = {"results":[{"review_id":"x", "aspects":[aspect("other","claim","negative","invented proof", [("Invented", "not present")])]}]}
        out = await IssueAspectExtractor(FakeClient([response])).extract_reviews([r])
        item = out["x"]["aspects"][0]
        assert item["evidence"] == "invented proof" and not item["evidence_verified"]
        assert item["issues"][0]["evidence"] == "not present" and not item["issues"][0]["evidence_verified"]
    asyncio.run(run())


def test_failed_review_and_omitted_ids_only_are_retried():
    async def run():
        reviews = [{"review_id":x,"text":x} for x in "abc"]
        ok = lambda rid: {"review_id":rid,"aspects":[]}
        c = FakeClient([
            {"results":[ok("a"), ok("b")]},
            {"results":[ok("c")]},
            OpenRouterError("temporary"),
            {"results":[ok("c")]},
        ])
        first = await IssueAspectExtractor(c, batch_size=3).extract_reviews(reviews)
        assert [x["status"] for x in first.values()] == ["success", "success", "success"]
        # first response omitted c (retries c only); then simulate a resumed run where only failed d is sent.
        reviews2 = reviews + [{"review_id":"d","text":"d"}]
        c2 = FakeClient([{"results":[ok("d")]}])
        resumed = await IssueAspectExtractor(c2, batch_size=4).extract_reviews(reviews2, existing=first)
        assert len(c2.calls) == 1
        payloads = [json.loads(call["user"])["reviews"] for call in c2.calls]
        assert payloads == [[{"review_id":"d","title":"","text":"d"}]]
        assert all(resumed[x]["status"] == "success" for x in "abcd")
    asyncio.run(run())


def test_missing_id_retries_only_missing_review():
    async def run():
        reviews = [{"review_id":x,"text":x} for x in "abc"]
        def ok(rid): return {"review_id":rid,"aspects":[]}
        c = FakeClient([{"results":[ok("a"),ok("b")]}, {"results":[ok("c")]}])
        await IssueAspectExtractor(c, batch_size=3).extract_reviews(reviews)
        assert json.loads(c.calls[1]["user"])["reviews"] == [{"review_id":"c","title":"","text":"c"}]
    asyncio.run(run())


def test_extraction_batches_can_run_concurrently():
    async def run():
        class ConcurrentClient:
            active = 0
            peak = 0

            async def json_completion(self, **kwargs):
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(0)
                self.active -= 1
                reviews = json.loads(kwargs["user"])["reviews"]
                return {"results": [{"review_id": row["review_id"], "aspects": []} for row in reviews]}

        client = ConcurrentClient()
        reviews = [{"review_id": str(i), "text": "Review text"} for i in range(24)]
        result = await IssueAspectExtractor(client, batch_size=12).extract_reviews(reviews)
        assert client.peak == 2
        assert len(result) == 24

    asyncio.run(run())


def test_extraction_token_cap_and_large_review_batches_split():
    class RecordingClient:
        def __init__(self):
            self.calls = []

        async def json_completion(self, **kwargs):
            self.calls.append(kwargs)
            rows = json.loads(kwargs["user"])["reviews"]
            return {"results": [{"review_id": row["review_id"], "aspects": []} for row in rows]}

    client = RecordingClient()
    reviews = [{"review_id": str(i), "title": ":)", "text": "x" * 27_000} for i in range(2)]
    result = asyncio.run(IssueAspectExtractor(client).extract_reviews(reviews))
    assert len(client.calls) == 2 and len(result) == 2
    assert all(call["max_tokens"] == 25_000 and call["reasoning_max_tokens"] == 20_000
               for call in client.calls)
    assert {row["review_id"] for call in client.calls
            for row in json.loads(call["user"])["reviews"]} == {"0", "1"}


def test_openrouter_hard_caps_and_reasoning_fallback():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(400, text="reasoning.max_tokens unsupported")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"results":[]}'}}]})

    async def run():
        client = OpenRouterClient(api_key="fake", max_retries=0, transport=httpx.MockTransport(handler))
        try:
            return await client.json_completion(schema_name="test", schema=EXTRACTION_SCHEMA,
                                                system="", user='{"title": ":)", "text": ":)"}')
        finally:
            await client.http.aclose()

    assert asyncio.run(run()) == {"results": []}
    assert all(call["max_tokens"] == 25_000 for call in calls)
    assert calls[0]["reasoning"] == {"max_tokens": 20_000}
    assert "reasoning" not in calls[1]


def test_normalization_token_cap_and_input_chunking(tmp_path):
    analyses = {"one": {"status": "success", "aspects": [
        aspect("ui_ux", "screen", "negative", "slow", [("Slow screen", "slow")])]}}
    client = FakeClient([{"groups": []}])
    # Stub source IDs to keep this test focused on request options.
    import appstore_reviews.normalization as n
    sources = {"src_one": {"id": "src_one", "category": "ui_ux", "aspect": "screen",
                           "description": "Slow screen", "descriptions": ["Slow screen"],
                           "evidence": ["slow"], "review_ids": ["one"]}}
    original = n._collect
    n._collect = lambda *_: (sources, {("one", 0, 0): "src_one"})
    client.responses = [{"groups": [{"canonical_name": "Slow screen", "source_ids": ["src_one"]}]}]
    try:
        asyncio.run(normalize_signals("issues", analyses, client, tmp_path))
    finally:
        n._collect = original
    assert client.calls[0]["max_tokens"] == 100_000
    assert client.calls[0]["reasoning_max_tokens"] == 80_000
    long_records = [{"id": str(i), "description": "x" * 130_000} for i in range(2)]
    chunks = split_input_records(long_records, key="items", system="", limit=100_000, max_records=30)
    assert [len(chunk) for chunk in chunks] == [1, 1]


def test_insights_token_cap(tmp_path, monkeypatch):
    import appstore_reviews.insights as insights

    for name in ("reviews.json", "review_analyses.json", "nlp_metrics.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(insights, "build_insights_input", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(insights, "_validate_generated_ids", lambda *_args: None)
    monkeypatch.setattr(insights, "_merge_deterministic_metrics", lambda *_args: {})
    client = FakeClient([{}])
    asyncio.run(insights.generate_saved_insights(tmp_path, client=client))
    assert client.calls[0]["max_tokens"] == 100_000
    assert client.calls[0]["reasoning_max_tokens"] == 50_000


def test_local_nlp_overlaps_issue_extraction(tmp_path):
    import threading

    extraction_started = threading.Event()

    class Sentiment:
        def analyze_reviews(self, reviews):
            assert extraction_started.wait(timeout=2)
            return [{"review_id": row["review_id"], "sentiment": "positive", "error": None} for row in reviews]

    def handler(request):
        extraction_started.set()
        reviews = json.loads(json.loads(request.content)["messages"][1]["content"])["reviews"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "results": [{"review_id": row["review_id"], "aspects": []} for row in reviews]
        })}}]})

    async def run():
        client = OpenRouterClient(api_key="fake", transport=httpx.MockTransport(handler))
        try:
            result = await analyze_full_pipeline(
                [{"review_id": "one", "text": "Useful app", "rating": 5}], tmp_path,
                client=client, sentiment_analyzer=Sentiment(),
            )
            assert result["nlp_metrics"]["coverage_metrics"]["successful_review_count"] == 1
        finally:
            await client.http.aclose()

    asyncio.run(run())


def test_openrouter_rejects_malformed_and_incomplete_json_locally():
    async def exercise(body):
        def handler(request):
            return httpx.Response(200, json={"choices":[{"message":{"content":body}}]})
        client = OpenRouterClient(api_key="fake", max_retries=0, transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(OpenRouterError):
                await client.json_completion(schema_name="test", schema=EXTRACTION_SCHEMA, system="", user="")
        finally:
            await client.http.aclose()
    asyncio.run(exercise("not json"))
    asyncio.run(exercise('{"results":[{}]}'))


def test_openrouter_provider_preferences_from_environment(monkeypatch):
    async def request_provider_payload():
        captured = []
        def handler(request):
            captured.append(json.loads(request.content)["provider"])
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"results":[]}'}}]})
        client = OpenRouterClient(api_key="fake", max_retries=0, transport=httpx.MockTransport(handler))
        try:
            await client.json_completion(schema_name="test", schema=EXTRACTION_SCHEMA, system="", user="")
        finally:
            await client.http.aclose()
        return captured[0]

    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    assert asyncio.run(request_provider_payload()) == {
        "order": ["parasail", "baseten", "together"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", " ")
    assert asyncio.run(request_provider_payload()) == {"require_parameters": True}
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "foo, bar")
    assert asyncio.run(request_provider_payload()) == {
        "order": ["foo", "bar"], "allow_fallbacks": False, "require_parameters": True,
    }


def test_openrouter_retries_429_and_falls_back_from_unsupported_schema():
    async def run():
        calls = []
        def handler(request):
            body = json.loads(request.content)
            calls.append(body["response_format"]["type"])
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            if len(calls) == 2:
                return httpx.Response(400, text="response_format json_schema unsupported")
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"results":[]}'}}],
                                             "usage": {"prompt_tokens": 1, "completion_tokens": 2,
                                                       "total_tokens": 3, "cost": 0.0001}})
        client = OpenRouterClient(api_key="fake", max_retries=2, transport=httpx.MockTransport(handler))
        try:
            result = await client.json_completion(schema_name="test", schema=EXTRACTION_SCHEMA,
                                                  system="", user="")
            assert result == {"results": []}
            assert calls == ["json_schema", "json_schema", "json_object"]
            assert client.usage.requests == 3 and client.usage.cost_usd == 0.0001
        finally:
            await client.http.aclose()
    asyncio.run(run())


def test_normalization_groups_equivalent_but_keeps_distinct_and_preserves_forms(tmp_path):
    analyses = {}
    for rid, desc, evidence in [
        ("r1", "Cannot cancel my subscription", "cannot cancel my subscription"),
        ("r2", "Unable to cancel Premium", "unable to cancel Premium"),
        ("r3", "Charged after cancelling Premium", "Charged after cancelling Premium"),
    ]:
        analyses[rid] = {"status":"success", "aspects":[aspect("pricing_subscriptions", "subscription", "negative", evidence, [(desc,evidence)])]}
    ids = ["src_one", "src_two", "src_three"]
    fake = FakeClient([{"groups":[{"canonical_name":"Unable to cancel subscription", "source_ids":ids[:2]}, {"canonical_name":"Charged after cancellation", "source_ids":[ids[2]]}]}])
    # Stub source IDs while exercising public normalization through deterministic collection.
    import appstore_reviews.normalization as n
    original_collect = n._collect
    sources = {
        ids[0]: {"id":ids[0],"category":"pricing_subscriptions","aspect":"subscription","description":"Cannot cancel my subscription","descriptions":["Cannot cancel my subscription"],"evidence":["cannot cancel my subscription"],"review_ids":["r1"]},
        ids[1]: {"id":ids[1],"category":"pricing_subscriptions","aspect":"subscription","description":"Unable to cancel Premium","descriptions":["Unable to cancel Premium"],"evidence":["unable to cancel Premium"],"review_ids":["r2"]},
        ids[2]: {"id":ids[2],"category":"pricing_subscriptions","aspect":"subscription","description":"Charged after cancelling Premium","descriptions":["Charged after cancelling Premium"],"evidence":["Charged after cancelling Premium"],"review_ids":["r3"]},
    }
    n._collect = lambda kind, data: (sources, {})
    try:
        updated, catalog = asyncio.run(normalize_signals("issues", analyses, fake, tmp_path))
    finally:
        n._collect = original_collect
    assert len(catalog) == 2
    merged = next(x for x in catalog if x["canonical_name"] == "Unable to cancel subscription")
    assert {s["description"] for s in merged["source_issues"]} == {"Cannot cancel my subscription", "Unable to cancel Premium"}
    assert merged["review_ids"] == ["r1", "r2"]
    assert all(x["canonical_id"].startswith("issue_") for x in catalog)


def test_normalization_retries_omitted_source_id_and_caches_result(tmp_path):
    records = [{"id": "src_one", "category": "stability", "aspect": "startup", "description": "App crashes on launch"},
               {"id": "src_two", "category": "stability", "aspect": "login", "description": "Login freezes"}]
    incomplete = {"groups": [{"canonical_name": "App crashes on launch", "source_ids": ["src_one"]}]}
    complete = {"groups": [*incomplete["groups"],
                           {"canonical_name": "Login freezes", "source_ids": ["src_two"]}]}
    client = FakeClient([incomplete, complete])
    groups = asyncio.run(_cached_completion(client, "issues", records, tmp_path, stage="batch_0"))
    assert len(client.calls) == 2
    assert "src_two" in client.calls[1]["system"]
    assert {sid for group in groups for sid in group["source_ids"]} == {"src_one", "src_two"}
    assert asyncio.run(_cached_completion(client, "issues", records, tmp_path, stage="batch_0")) == groups
    assert len(client.calls) == 2


def test_normalization_keeps_valid_groups_if_retries_still_omit_id(tmp_path):
    records = [{"id": "src_one", "category": "stability", "aspect": "startup", "description": "App crashes on launch"},
               {"id": "src_two", "category": "stability", "aspect": "login", "description": "Login freezes"}]
    incomplete = {"groups": [{"canonical_name": "App crashes on launch", "source_ids": ["src_one"]}]}
    client = FakeClient([incomplete, incomplete, incomplete])
    groups = asyncio.run(_cached_completion(client, "issues", records, tmp_path, stage="batch_0"))
    assert len(client.calls) == 3
    assert groups == [{"canonical_name": "App crashes on launch", "source_ids": ["src_one"], "category": "stability"},
                      {"canonical_name": "Login freezes", "source_ids": ["src_two"], "category": "stability"}]


def test_normalization_retries_duplicate_source_id(tmp_path):
    records = [{"id": "src_one", "category": "stability", "aspect": "startup", "description": "Crash"},
               {"id": "src_two", "category": "stability", "aspect": "login", "description": "Freeze"}]
    duplicate = {"groups": [{"canonical_name": "Crash", "source_ids": ["src_one"]},
                            {"canonical_name": "Freeze", "source_ids": ["src_one", "src_two"]}]}
    corrected = {"groups": [{"canonical_name": "Crash", "source_ids": ["src_one"]},
                            {"canonical_name": "Freeze", "source_ids": ["src_two"]}]}
    client = FakeClient([duplicate, corrected])
    result = asyncio.run(_cached_completion(client, "issues", records, tmp_path, stage="batch_0"))
    assert len(client.calls) == 2
    assert "src_one" in client.calls[1]["system"]
    assert [sid for group in result for sid in group["source_ids"]] == ["src_one", "src_two"]


@pytest.mark.parametrize("invalid", [
    {"groups": [{"canonical_name": "Crash", "source_ids": ["src_one"]},
                {"canonical_name": "Freeze", "source_ids": ["src_one", "src_two"]}]},
    {"groups": [{"canonical_name": "Crash", "source_ids": ["src_one", "invented"]}]},
    {"groups": [{"canonical_name": "General issue", "source_ids": ["src_one", "src_three"]}]},
    {"groups": [{"canonical_name": "Same name", "source_ids": ["src_one"]},
                {"canonical_name": "Same name", "source_ids": ["src_two"]}]},
])
def test_normalization_repairs_invalid_groups_without_losing_ids(tmp_path, invalid):
    records = [{"id": "src_one", "category": "stability", "aspect": "startup", "description": "Crash"},
               {"id": "src_two", "category": "stability", "aspect": "login", "description": "Freeze"},
               {"id": "src_three", "category": "performance", "aspect": "loading", "description": "Slow"}]
    client = FakeClient([invalid, invalid, invalid])
    result = asyncio.run(_cached_completion(client, "issues", records, tmp_path, stage="batch_0"))
    ids = [sid for group in result for sid in group["source_ids"]]
    assert len(client.calls) == 3
    assert sorted(ids) == sorted(record["id"] for record in records)
    assert len(ids) == len(set(ids))
    assert len({(group["category"], group["canonical_name"].casefold()) for group in result}) == len(result)


def test_metrics_dedupe_review_ids_and_saved_recalculation(tmp_path):
    reviews = [{"review_id":"a","rating":1,"country":"US"}, {"review_id":"b","rating":5,"country":"US"}, {"review_id":"c","rating":2}]
    analysis = {
        "a":{"status":"success","aspects":[aspect("stability","crash","negative","x",[("Crash","x")]), aspect("stability","crash","negative","x",[("Crash","x")])]},
        "b":{"status":"success","aspects":[aspect("stability","crash","positive","y",[("Crash","y")])]},
        "c":{"status":"error","aspects":[]},
    }
    catalog = [{"canonical_id":"issue_crash","canonical_name":"Crash","category":"stability","review_ids":["a","b"]}]
    m = calculate_nlp_metrics(reviews, analysis, catalog, [], [{"review_id":"a","sentiment":"negative"},{"review_id":"b","sentiment":"positive"}])
    im = m["issue_metrics"]["issues"][0]
    assert im["review_count"] == 2 and im["review_ids"] == ["a","b"]
    assert im["average_rating"] == 3.0
    assert m["aspect_metrics"][0]["review_count"] == 2
    assert m["coverage_metrics"]["successful_review_count"] == 2
    (tmp_path/"reviews.json").write_text(json.dumps(reviews))
    (tmp_path/"review_analyses.json").write_text(json.dumps(analysis))
    (tmp_path/"issue_catalog.json").write_text(json.dumps(catalog))
    (tmp_path/"feature_request_catalog.json").write_text("[]")
    (tmp_path/"sentiment_results.json").write_text(json.dumps([{"review_id":"a","sentiment":"negative"},{"review_id":"b","sentiment":"positive"}]))
    saved = recalculate_saved_metrics(tmp_path)
    assert saved == m
    assert (tmp_path/"nlp_metrics.json").exists()


def test_real_source_mapping_and_cached_normalization(tmp_path):
    analyses = {}
    descriptions = [
        ("a", "Cannot cancel my subscription", "I cannot cancel my subscription"),
        ("b", "Unable to cancel Premium", "I am unable to cancel Premium"),
        ("c", "Charged after cancelling Premium", "I was charged after cancelling Premium"),
    ]
    for rid, description, evidence in descriptions:
        item = aspect("pricing_subscriptions", "subscription", "negative", evidence,
                      [(description, evidence)])
        item["evidence_verified"] = True
        item["issues"][0]["evidence_verified"] = True
        analyses[rid] = {"status": "success", "aspects": [item]}

    class GroupingClient:
        def __init__(self): self.calls = 0
        async def json_completion(self, **kwargs):
            self.calls += 1
            items = json.loads(kwargs["user"])["items"]
            cancel = [x["id"] for x in items if "cancel" in x["description"].lower() and "charged" not in x["description"].lower()]
            charged = [x["id"] for x in items if "charged" in x["description"].lower()]
            return {"groups": [{"canonical_name": "Unable to cancel subscription", "source_ids": cancel},
                               {"canonical_name": "Charged after cancellation", "source_ids": charged}]}

    client = GroupingClient()
    updated, catalog = asyncio.run(normalize_signals("issues", analyses, client, tmp_path))
    assert client.calls == 1
    ids = {rid: updated[rid]["aspects"][0]["issues"][0]["canonical_id"] for rid in "abc"}
    assert ids["a"] == ids["b"] != ids["c"]
    merged = next(x for x in catalog if x["canonical_id"] == ids["a"])
    assert {s["description"] for s in merged["source_issues"]} == {descriptions[0][1], descriptions[1][1]}
    assert merged["review_ids"] == ["a", "b"]
    updated2, catalog2 = asyncio.run(normalize_signals("issues", updated, client, tmp_path))
    assert client.calls == 1
    assert updated2 == updated and catalog2 == catalog


def test_mixed_aspect_and_breakdown_denominators():
    reviews = [
        {"review_id": "a", "rating": 1, "country": "us", "observed_countries": ["us", "ca"], "app_version": "1.0", "updated_at": "2026-01-02T00:00:00Z"},
        {"review_id": "b", "rating": None, "country": "us", "app_version": "2.0", "updated_at": "2026-02-02T00:00:00Z"},
    ]
    def mk(sentiment):
        row = aspect("features", "playlist", sentiment, "proof", [("Playlist broken", "proof")])
        row["evidence_verified"] = True
        row["issues"][0].update({"canonical_id": "issue_x", "evidence_verified": True})
        return row
    analyses = {"a": {"status": "success", "aspects": [mk("positive"), mk("negative")]},
                "b": {"status": "success", "aspects": [mk("neutral")]}}
    catalog = [{"canonical_id": "issue_x", "canonical_name": "Playlist broken", "category": "features", "review_ids": ["a", "b"]}]
    metrics = calculate_nlp_metrics(reviews, analyses, catalog, [])
    aspect_metric = metrics["aspect_metrics"][0]
    assert aspect_metric["mixed_review_count"] == 1
    assert aspect_metric["sentiment_distribution"]["negative"]["review_count"] == 1
    issue = metrics["issue_metrics"]["issues"][0]
    assert issue["rating_sample_size"] == 1 and issue["negative_rating_share"] == 1.0
    assert metrics["country_breakdown"]["ca"]["issue_share"]["issue_x"] == 1.0
    assert metrics["time_breakdown"]["2026-01"]["issue_share"]["issue_x"] == 1.0


def test_pipeline_sends_all_reviews_to_llm_and_resumes_without_api(tmp_path):
    reviews = [{"review_id": "p", "title": "Great", "text": "Love the interface", "rating": 5},
               {"review_id": "n", "title": "Problem", "text": "Cannot cancel subscription", "rating": 1}]
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        user = json.loads(body["messages"][1]["content"])
        if "reviews" in user:
            results = [
                {"review_id": "p", "aspects": [aspect("ui_ux", "interface", "positive", "Love the interface")]},
                {"review_id": "n", "aspects": [aspect("pricing_subscriptions", "cancellation", "negative",
                        "Cannot cancel subscription", [("Cannot cancel subscription", "Cannot cancel subscription")])]},
            ]
            content = {"results": results}
        else:
            content = {"groups": [{"canonical_name": "Cannot cancel subscription",
                                   "source_ids": [item["id"] for item in user["items"]]}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                                                   "total_tokens": 30, "cost": 0.001}})
    class Sentiment:
        def analyze_reviews(self, batch):
            return [{"review_id": r["review_id"], "sentiment": "negative" if r["review_id"] == "n" else "positive",
                     "error": None} for r in batch]
    class Keywords:
        calls = []
        def extract_reviews(self, batch):
            self.calls.append([r["review_id"] for r in batch])
            return [{"review_id": r["review_id"], "keywords": [], "error": None} for r in batch]
    async def run():
        client = OpenRouterClient(api_key="fake", transport=httpx.MockTransport(handler))
        keyword_model = Keywords()
        try:
            first = await analyze_full_pipeline(reviews, tmp_path, client=client,
                sentiment_analyzer=Sentiment(), keyword_extractor=keyword_model)
            assert keyword_model.calls == [["n"]]
            assert {item["review_id"] for item in json.loads(requests[0]["messages"][1]["content"])["reviews"]} == {"p", "n"}
            assert first["nlp_metrics"]["coverage_metrics"]["successful_review_count"] == 2
            assert first["api_usage"]["extraction"]["cost_usd"] == 0.001
            second = await analyze_full_pipeline(reviews, tmp_path, client=client,
                sentiment_analyzer=Sentiment(), keyword_extractor=keyword_model)
            assert len(requests) == 2
            assert second["nlp_metrics"] == first["nlp_metrics"]
        finally:
            await client.http.aclose()
    asyncio.run(run())


def test_cross_category_normalization_falls_back_to_category_batches(tmp_path):
    analyses = {}
    for rid, cat, description in [("a", "features", "Playlist is missing"),
                                  ("b", "payments_billing", "Payment failed")]:
        row = aspect(cat, description, "negative", description, [(description, description)])
        row["evidence_verified"] = True
        row["issues"][0]["evidence_verified"] = True
        analyses[rid] = {"status": "success", "aspects": [row]}
    class BadThenGood:
        def __init__(self): self.calls = 0
        async def json_completion(self, **kwargs):
            self.calls += 1
            records = json.loads(kwargs["user"])["items"]
            if self.calls == 1:
                return {"groups": [{"canonical_name": "Problems", "source_ids": [r["id"] for r in records]}]}
            return {"groups": [{"canonical_name": records[0]["description"], "source_ids": [records[0]["id"]]}]}
    client = BadThenGood()
    _, catalog = asyncio.run(normalize_signals("issues", analyses, client, tmp_path))
    assert client.calls == 3
    assert {x["category"] for x in catalog} == {"features", "payments_billing"}


def test_cross_category_equivalent_issues_get_one_canonical_id(tmp_path):
    analyses = {}
    for rid, category in [("a", "content"), ("b", "ui_ux")]:
        row = aspect(category, "ads", "negative", "Too many ads", [("Too many ads", "Too many ads")])
        row["evidence_verified"] = True
        row["issues"][0]["evidence_verified"] = True
        analyses[rid] = {"status": "success", "aspects": [row]}
    class Reconciler:
        def __init__(self): self.calls = 0
        async def json_completion(self, **kwargs):
            self.calls += 1
            body = json.loads(kwargs["user"])
            if "items" in body:
                return {"groups": [{"canonical_name": "Too many ads", "source_ids": [r["id"]]}
                                   for r in body["items"]]}
            return {"groups": [{"canonical_name": "Too many ads", "category": "ui_ux",
                               "source_ids": [r["id"] for r in body["groups"]],
                               "justification": "Both describe the same excessive advertisement load."}]}
    client = Reconciler()
    updated, catalog = asyncio.run(normalize_signals("issues", analyses, client, tmp_path))
    assert client.calls == 2
    assert len(catalog) == 1 and catalog[0]["category"] == "ui_ux"
    assert catalog[0]["review_ids"] == ["a", "b"]
    assert catalog[0]["category_reconciliation"]
    assert updated["a"]["aspects"][0]["issues"][0]["canonical_id"] == updated["b"]["aspects"][0]["issues"][0]["canonical_id"]

