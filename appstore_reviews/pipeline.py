"""Sentiment -> negative review keywords -> per-language frequency summary."""

from __future__ import annotations

from collections.abc import Mapping

from .keywords import KeywordExtractor, aggregate_keywords
from .sentiment import SentimentAnalyzer


def analyze_negative_keywords(
    reviews: list[Mapping[str, object]],
    *,
    sentiment_analyzer: SentimentAnalyzer | None = None,
    keyword_extractor: KeywordExtractor | None = None,
) -> dict:
    """Analyze collected reviews without changing the original records.

    The keyword extractor remains independently usable on reviews of any
    sentiment; only this orchestration function filters for negative reviews.
    """
    if not isinstance(reviews, list):
        raise TypeError("reviews must be a list")
    sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
    keyword_extractor = keyword_extractor or KeywordExtractor()
    sentiments = sentiment_analyzer.analyze_reviews(reviews)
    if len(sentiments) != len(reviews):
        raise ValueError("sentiment analyzer returned an unexpected number of results")
    negatives = [review for review, result in zip(reviews, sentiments, strict=True)
                 if result["sentiment"] == "negative" and not result.get("error")]
    keywords = keyword_extractor.extract_reviews(negatives)
    if len(keywords) != len(negatives):
        raise ValueError("keyword extractor returned an unexpected number of results")
    return {
        "review_count": len(reviews),
        "negative_review_count": len(negatives),
        "sentiment_results": sentiments,
        "keyword_results": keywords,
        "common_keywords_by_language": aggregate_keywords(keywords),
    }
