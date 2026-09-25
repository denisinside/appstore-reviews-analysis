"""Small deterministic checks; real checkpoint smoke test is documented separately."""

import hashlib
import json

import numpy as np
import pytest

from appstore_reviews.keywords import DEFAULT_KEYWORD_MODEL, DEFAULT_KEYWORD_REVISION, KeywordExtractor, aggregate_keywords, generate_candidates, generate_candidate_records, normalize_phrase, select_keywords
from appstore_reviews.pipeline import analyze_negative_keywords
from models_arena.keyword_extraction.benchmark import DATA, LABEL_SHA256, matched_count, phrase_match, tfidf_baseline
from models_arena.keyword_extraction.benchmark_v2 import DATASETS, HELDOUT_SETUP_EXCLUSIONS, load_dataset


class FakeEncoder:
    def __init__(self):
        self.calls = 0
        self.texts = []

    def encode(self, texts, **kwargs):
        self.calls += 1
        self.texts.append(list(texts))
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


def test_keyword_extractor_batches_embeddings_across_reviews():
    extractor = KeywordExtractor(max_candidates=20)
    fake = FakeEncoder()
    extractor._model = fake
    reviews = [
        {"review_id": "a", "title": "Playlist crashes", "text": "My playlist crashes again", "language": "en"},
        {"review_id": "b", "title": "Songs disappear", "text": "Songs disappear from playlist", "language": "en"},
    ]

    result = extractor.extract_reviews(reviews)

    assert fake.calls == 1
    assert len(fake.texts[0]) == sum(1 + len(generate_candidate_records(
        review["title"] + "\n" + review["text"], review["language"], limit=20)) for review in reviews)
    assert [row["review_id"] for row in result] == ["a", "b"]
    assert all(row["keywords"] and row["error"] is None for row in result)


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


def test_candidate_generation_preserves_quantity_negation_and_longer_complaints():
    text = "Only 6 skips in an hour. There are 2 minutes of ads and 3 times a day it logs me out. I can't play downloaded songs."
    phrases = generate_candidates(text, "en")
    assert "Only 6 skips" in phrases
    assert "Only 6 skips in" not in phrases
    assert "2 minutes of ads" in phrases
    assert "3 times a day" in phrases
    assert any("can't play downloaded songs" in phrase for phrase in phrases)
    assert all("ads and 3" not in phrase for phrase in phrases)


@pytest.mark.parametrize("text, expected", [("Price is 4,99 zł", "4,99 zł"), ("Storage is 2.5 GB", "2.5 GB")])
def test_decimal_quantities_with_units_keep_exact_candidate_and_offsets(text, expected):
    records = generate_candidate_records(text, "en")
    matching = [row for row in records if row["text"] == expected]
    assert matching
    record = matching[0]
    assert text[record["source_start"]:record["source_end"]] == expected
    assert not any(row["text"].endswith(expected.split()[0]) for row in records)
    polish = generate_candidates("aplikacja wymaga zakupu subskrypcji 4,99 zł", "pl")
    assert any("4,99 zł" in phrase for phrase in polish)
    assert not any(phrase.endswith("4,99") for phrase in polish)


def test_negation_and_numeric_restrictions_remain_in_candidates():
    text = "Only 4 skips are allowed and it is not working."
    phrases = generate_candidates(text, "en")
    assert any("Only 4 skips" in phrase for phrase in phrases)
    assert any("not working" in phrase for phrase in phrases)
    ukrainian = generate_candidates("Перестали працювати кружки або працюють через раз", "uk")
    assert any("працюють через раз" in phrase for phrase in ukrainian)
    assert "через" not in ukrainian


@pytest.mark.parametrize("text, language, positive, complaint", [
    ("Негативний відгук: налаштування працюють.", "uk", "налаштування працюють", "не працюють"),
    ("Negative review: settings work fine.", "en", "settings work fine", "not working"),
    ("The bots make it somewhat bearable.", "en", "bots make it somewhat bearable", "not working"),
])
def test_positive_aspects_in_negative_reviews_are_not_complaint_candidates(text, language, positive, complaint):
    phrases = generate_candidates(text, language)
    assert not any(positive in phrase for phrase in phrases)
    assert any(complaint in phrase for phrase in generate_candidates(complaint, language))


def test_positive_clauses_do_not_emit_orphaned_fragments():
    uk = generate_candidates("Я не можу відкрити чати, хоча дзвінки, налаштування працюють", "uk")
    assert any("не можу відкрити чати" in phrase for phrase in uk)
    assert not any("налаштування" in phrase for phrase in uk)
    en = generate_candidates("There are so many ads. Only the bots make it somewhat bearable.", "en")
    assert any("ads" in phrase for phrase in en)
    assert not any("bots" in phrase for phrase in en)


def test_production_e5_defaults_and_query_prefix():
    extractor = KeywordExtractor()
    assert extractor.model_id == DEFAULT_KEYWORD_MODEL == "intfloat/multilingual-e5-small"
    assert extractor.revision == DEFAULT_KEYWORD_REVISION == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert extractor.prefix == "query: "
    fake = FakeEncoder()
    extractor._model = fake
    extractor.extract_review({"review_id": "e5", "title": "Playlist crashes", "text": "Playlist crashes", "language": "en"})
    assert fake.texts and all(value.startswith("query: ") for value in fake.texts[0])


def test_ukrainian_colloquial_candidates_and_evidence():
    text = "Ну три реклами після двох пісень вже троха за дохєра. Не можу увійти в акаунт."
    records = generate_candidate_records(text, "uk")
    phrases = [row["text"] for row in records]
    assert "три реклами" in phrases
    assert "після двох пісень" in phrases
    assert "Не можу увійти в акаунт" in phrases
    assert all(row["text"] in text[row["source_start"]:row["source_end"]] for row in records)
    assert all(row["text"] in row["evidence_span"] for row in records)


def test_meaning_guards_for_mixed_praise_irony_and_external_negation():
    mixed = generate_candidates("Great interface, but every update logs me out.", "en")
    assert "Great interface" not in mixed
    assert any("logs me out" in phrase for phrase in mixed)
    irony = generate_candidates("Great, another crash after update.", "en")
    assert "Great" not in irony
    assert any("crash after update" in phrase for phrase in irony)
    lithuanian = generate_candidates("Nepasakyčiau, kad Spotify yra puiki programa. Persukti nevisada galima.", "lt")
    assert not any("Spotify yra" in phrase for phrase in lithuanian)
    assert any("nevisada" in phrase for phrase in lithuanian)
    ukrainian = generate_candidates("Якщо це авторскі права, то чому у інших ці пісні є?", "uk")
    assert not any("права, то" in phrase for phrase in ukrainian)


def test_ranking_removes_nested_variants_but_preserves_distinct_limits():
    records = [{"text": phrase, "evidence_span": "Songs disappear from playlist", "source_start": 0, "source_end": len(phrase)}
               for phrase in ("songs disappear", "songs disappear from playlist", "only 6 skips", "only 3 skips", "app crashes")]
    chosen = select_keywords(records, [0.9, 0.92, 0.8, 0.79, 0.7], top_k=5)
    texts = [row["text"] for row in chosen]
    assert "songs disappear from playlist" in texts and "songs disappear" not in texts
    assert "only 6 skips" in texts and "only 3 skips" in texts
    assert all(row["evidence_span"] for row in chosen)


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


def test_v2_gold_freeze_and_independent_heldout_spans():
    for name in ("old", "heldout"):
        directory, label_file, digest, _ = DATASETS[name]
        assert hashlib.sha256((directory / label_file).read_bytes()).hexdigest() == digest
        all_reviews = {row["review_id"]: row for row in map(json.loads, (directory / "reviews.jsonl").read_text(encoding="utf-8").splitlines())}
        for label in map(json.loads, (directory / label_file).read_text(encoding="utf-8").splitlines()):
            review = all_reviews[label["reviewId"]]
            for keyword in label["keywords"]:
                assert review[keyword["sourceField"]][keyword["sourceStart"]:keyword["sourceEnd"]] == keyword["sourceSpan"]
                assert keyword["text"] in keyword["sourceSpan"]
    old, _ = load_dataset("old")
    heldout, _ = load_dataset("heldout")
    assert len(old) == 255 and len(heldout) == 195
    assert {row["review_id"] for row in old}.isdisjoint(row["review_id"] for row in heldout)
    assert HELDOUT_SETUP_EXCLUSIONS.isdisjoint(row["review_id"] for row in heldout)
