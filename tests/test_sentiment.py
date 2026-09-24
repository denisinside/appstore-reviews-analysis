"""Offline tests for the optional production sentiment interface."""

import pytest

from appstore_reviews.sentiment import (
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_SENTIMENT_REVISION,
    SentimentAnalyzer,
    _combine_scores,
)


def review(review_id: str, title: str | None, text: str | None, rating: int) -> dict:
    return {"review_id": review_id, "title": title, "text": text, "rating": rating}


def test_default_checkpoint_matches_arena():
    analyzer = SentimentAnalyzer()
    assert analyzer.model_id == DEFAULT_SENTIMENT_MODEL == "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    assert analyzer.revision == DEFAULT_SENTIMENT_REVISION == "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"


def test_combine_five_classes():
    scores = _combine_scores(
        [0.1, 0.2, 0.3, 0.25, 0.15],
        {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"},
    )
    assert scores == pytest.approx({"negative": 0.3, "neutral": 0.3, "positive": 0.4})


def test_single_batch_and_text_only(monkeypatch):
    analyzer = SentimentAnalyzer(batch_size=2)
    calls = []

    def infer(texts):
        calls.append(texts)
        return [
            {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
            {"negative": 0.8, "neutral": 0.1, "positive": 0.1},
        ][:len(texts)]

    monkeypatch.setattr(analyzer, "_classify_texts", infer)
    input_rows = [
        review("en", "Nice", "Works  well", 1),
        review("uk", "Погано", "Не працює", 5),
    ]
    result = analyzer.analyze_reviews(input_rows)
    assert result[0]["sentiment"] == "positive"
    assert result[1]["sentiment"] == "negative"
    assert result[0]["sentiment_scores"]["positive"] == 0.9
    assert result[0]["sentiment_model"]["id"] == analyzer.model_id
    assert calls == [["Nice\nWorks well", "Погано\nНе працює"]]
    assert input_rows[0]["text"] == "Works  well"
    assert analyzer.analyze_review(review("other", "Good", None, 2))["sentiment"] == "positive"
    assert len(calls) == 2


def test_empty_text_does_not_load(monkeypatch):
    analyzer = SentimentAnalyzer()
    monkeypatch.setattr(analyzer, "_classify_texts", lambda _: pytest.fail("model should not load"))
    assert analyzer.analyze_review(review("empty", "  ", None, 3))["error"] == "empty text"


def test_batch_error_isolated_to_one_review(monkeypatch):
    analyzer = SentimentAnalyzer(batch_size=2)

    def infer(texts):
        if len(texts) > 1 or "bad" in texts[0]:
            raise RuntimeError("inference failed")
        return [{"negative": 0.1, "neutral": 0.1, "positive": 0.8}]

    monkeypatch.setattr(analyzer, "_classify_texts", infer)
    result = analyzer.analyze_reviews([review("good", "good", None, 3), review("bad", "bad", None, 3)])
    assert result[0]["sentiment"] == "positive"
    assert result[1]["sentiment"] is None
    assert result[1]["error"] == "inference failed"


def test_invalid_input():
    with pytest.raises(ValueError, match="batch_size"):
        SentimentAnalyzer(batch_size=0)
    with pytest.raises(ValueError, match="review_id"):
        SentimentAnalyzer().analyze_reviews([{"text": "hello"}])
