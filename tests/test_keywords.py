"""Small deterministic checks; real checkpoint smoke test is documented separately."""

import hashlib
import json

import numpy as np
import pytest

from appstore_reviews.keywords import KeywordExtractor, aggregate_keywords, generate_candidates, normalize_phrase
from appstore_reviews.pipeline import analyze_negative_keywords
from models_arena.keyword_extraction.benchmark import DATA, LABEL_SHA256, matched_count, phrase_match, tfidf_baseline


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([float("playlist" in lower), float("crash" in lower), float("не" in lower)])
        matrix = np.asarray(vectors, dtype=float)
        lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
        lengths[lengths == 0] = 1
        return matrix / lengths


def test_unicode_candidate_extraction_preserves_negation_and_source_text():
    text = "Не можу увійти в акаунт. Плейлист постійно зникає!"
    phrases = generate_candidates(text, "uk")
    assert "Не можу увійти" in phrases
    assert any("Плейлист" in phrase for phrase in phrases)
    assert all(phrase in text for phrase in phrases)
    assert normalize_phrase("  Акаунт…  ") == "акаунт"


def test_keyword_extractor_single_batch_empty_and_reuses_model():
    extractor = KeywordExtractor(max_candidates=20)
    fake = FakeEncoder()
    extractor._model = fake
    reviews = [
        {"review_id": "a", "title": "Playlist crashes", "text": "My playlist crashes again", "language": "en", "rating": 1},
        {"review_id": "b", "title": "", "text": "", "language": "uk", "rating": 5},
    ]
    result = extractor.extract_reviews(reviews)
    assert len(result) == 2 and result[0]["keywords"] and result[1]["keywords"] == []
    assert result[0]["review_id"] == "a" and len(result[0]["keywords"]) <= 5
    assert fake.calls == 1
    extractor.extract_review(reviews[0])
    assert fake.calls == 2 and extractor._model is fake
    reviews[0]["rating"] = 5
    assert result[0]["keywords"] == extractor.extract_review(reviews[0])["keywords"]
    assert extractor.extract_review({"review_id": "c", "title": "Playlist", "content": "Playlist crashes", "language": "en"})["keywords"]
    assert extractor.extract_review({"review_id": "d", "title": "", "text": "", "content": "Playlist crashes", "language": "en"})["keywords"]


def test_invalid_input_and_no_model_needed_for_empty_text():
    extractor = KeywordExtractor()
    assert extractor.extract_review({"review_id": "x", "title": None, "text": None})["keywords"] == []
    with pytest.raises(ValueError):
        extractor.extract_review({"title": "text"})
    with pytest.raises(TypeError):
        extractor.extract_reviews("text")
    assert generate_candidates("cannot log in", "en")
    bad = extractor.extract_reviews([{"review_id": "bad", "title": 123, "text": "body"},
                                     {"review_id": "empty", "title": "", "text": ""}])
    assert bad[0]["error"].startswith("invalid review text") and bad[1]["error"] is None


def test_long_review_candidate_limit_and_sentence_boundary():
    text = "Playlist crashes. Cannot log in! " + "Songs disappear from playlist. " * 100
    phrases = generate_candidates(text, "en", limit=40)
    assert 0 < len(phrases) <= 40
    assert "crashes Cannot" not in phrases
    assert any("Songs disappear" in phrase for phrase in phrases)


def test_one_to_one_matching_and_negation_guard():
    assert matched_count(["cannot log in"], ["cannot log in", "cannot log in"]) == 1
    assert phrase_match("can log in", "cannot log in") == 0
    assert phrase_match("playlist crashes", "crashes playlist") == 1
    assert phrase_match("playlist fails", "playlist crashes") == 0


def test_aggregation_counts_unique_reviews_and_separates_languages():
    rows = [
        {"review_id": "1", "language": "en", "keywords": [{"text": "Playlist crashes", "score": 0.8}, {"text": "playlist crashes", "score": 0.9}], "error": None},
        {"review_id": "2", "language": "en", "keywords": [{"text": "playlist crashes", "score": 0.6}], "error": None},
        {"review_id": "3", "language": "uk", "keywords": [{"text": "плейлист зникає", "score": 0.7}], "error": None},
    ]
    result = aggregate_keywords(rows)
    assert result["en"][0]["review_count"] == 2
    assert result["en"][0]["share"] == 1
    assert result["en"][0]["average_score"] == 0.7
    assert result["uk"][0]["review_count"] == 1


def test_pipeline_filters_negative_outside_extractor():
    class Sentiment:
        def analyze_reviews(self, reviews):
            return [{"review_id": review["review_id"], "sentiment": sentiment, "error": None}
                    for review, sentiment in zip(reviews, ("negative", "positive", "negative"), strict=True)]

    class Keywords:
        def extract_reviews(self, reviews):
            return [{"review_id": row["review_id"], "language": row["language"],
                     "keywords": [{"text": "ads", "score": 0.6}], "error": None} for row in reviews]

    reviews = [{"review_id": rid, "language": "en", "title": "Ads", "text": "Many ads"} for rid in ("a", "b", "c")]
    result = analyze_negative_keywords(reviews, sentiment_analyzer=Sentiment(), keyword_extractor=Keywords())
    assert [r["review_id"] for r in result["keyword_results"]] == ["a", "c"]
    assert result["negative_review_count"] == 2
    assert result["common_keywords_by_language"]["en"][0]["review_count"] == 2


def test_gold_source_spans_and_frozen_hash():
    path = DATA / "keyword_labels.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LABEL_SHA256
    labels = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    reviews = {row["review_id"]: row for row in [json.loads(line) for line in (DATA / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]}
    assert len(labels) == 255
    for row in labels:
        review = reviews[row["reviewId"]]
        for keyword in row["keywords"]:
            field = keyword["sourceField"]
            assert review[field][keyword["sourceStart"]:keyword["sourceEnd"]] == keyword["sourceSpan"]
            assert keyword["text"] in keyword["sourceSpan"]


def test_tfidf_handles_empty_corpus():
    predictions, _ = tfidf_baseline([{"review_id": "empty", "title": "", "text": "", "language": "uk", "evaluation_sets": ["ukrainian"]}])
    assert predictions[0]["keywords"] == []
