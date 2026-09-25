"""Frozen v2 keyword comparison. Old Spotify is regression; heldout is final."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

from appstore_reviews.keywords import KeywordExtractor, MAX_CANDIDATES, generate_candidate_records, normalize_phrase, select_keywords
from appstore_reviews.preprocessing import prepare_review_text
from .benchmark import MODELS, _rss_mb, read_jsonl, summarize, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASETS = {
    "old": (ROOT / "models_arena/datasets/spotify_appstore_v1", "keyword_labels_v2.jsonl",
            "44cb6c1f5ce5ca9396ef3399ea0bab49afe078d53958df17e1a33e7d88159465", "results_v2"),
    "heldout": (ROOT / "models_arena/datasets/heldout_crossapp_v1", "keyword_labels.jsonl",
                "3329280c786fff85ccbf37ced681c54d198214f1e77a0a119b255e0ace589307", "results_heldout_v1"),
}
HELDOUT_SETUP_EXCLUSIONS = {"14583149356", "14130625981"}


def load_dataset(name: str) -> tuple[list[dict], dict[str, dict]]:
    directory, label_file, digest, _ = DATASETS[name]
    path = directory / label_file
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Gold labels changed after freezing"
    labels = {row["reviewId"]: row for row in read_jsonl(path)}
    reviews = [row for row in read_jsonl(directory / "reviews.jsonl") if row["review_id"] in labels]
    if name == "heldout":
        reviews = [row for row in reviews if labels[row["review_id"]]["sentiment"] == "negative"
                   and row["review_id"] not in HELDOUT_SETUP_EXCLUSIONS]
        for label in labels.values():
            label.setdefault("sentimentNeedsManualReview", False)
    assert len(reviews) == (195 if name == "heldout" else 255)
    return reviews, labels


def tfidf_predictions(reviews: list[dict]) -> tuple[list[dict], dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    started = time.perf_counter()
    documents = [prepare_review_text(row) for row in reviews]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)[^\W_]+(?:['’][^\W_]+)?", ngram_range=(1, 6),
                                 lowercase=True, sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(documents)
    vocabulary = vectorizer.vocabulary_
    rows = []
    for index, review in enumerate(reviews):
        candidates = generate_candidate_records(documents[index], review.get("language"), limit=MAX_CANDIDATES)
        scores = []
        for candidate in candidates:
            column = vocabulary.get(normalize_phrase(candidate["text"]))
            scores.append(float(matrix[index, column]) if column is not None else 0.0)
        rows.append({"review_id": review["review_id"], "language": review.get("language") or "und",
                     "evaluation_sets": review["evaluation_sets"],
                     "keywords": select_keywords(candidates, scores, top_k=5), "error": None})
    total = time.perf_counter() - started
    return rows, {"load_seconds": 0.0, "total_seconds": round(total, 3),
                  "mean_seconds_per_review": round(total / len(reviews), 6), "rss_mb": _rss_mb()}


def embedding_predictions(name: str, reviews: list[dict], folder: Path) -> tuple[list[dict], dict]:
    model_id, revision, prefix = MODELS[name]
    extractor = KeywordExtractor(model_id, revision, batch_size=32, max_candidates=MAX_CANDIDATES, prefix=prefix)
    partial_path = folder / "predictions.partial.jsonl"
    existing = read_jsonl(partial_path) if partial_path.exists() else []
    assert [row["review_id"] for row in existing] == [row["review_id"] for row in reviews[:len(existing)]]
    resumed_from = len(existing)
    started = time.perf_counter()
    extractor._ensure_loaded()
    load_seconds = time.perf_counter() - started
    rows = existing
    for begin in range(resumed_from, len(reviews), 10):
        batch = extractor.extract_reviews(reviews[begin:begin + 10])
        for row, review in zip(batch, reviews[begin:begin + 10], strict=True):
            row["evaluation_sets"] = review["evaluation_sets"]
        rows.extend(batch)
        write_jsonl(partial_path, rows)
        print(f"{name} {len(rows)}/{len(reviews)}", flush=True)
    total = time.perf_counter() - started
    perf = {"load_seconds": round(load_seconds, 3), "total_seconds": round(total, 3),
            "mean_seconds_per_review": round((total-load_seconds) / max(1, len(reviews)-resumed_from), 6),
            "rss_mb": _rss_mb(), "resumed_from": resumed_from}
    del extractor
    gc.collect()
    return rows, perf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["tfidf", *MODELS], required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    reviews, labels = load_dataset(args.dataset)
    folder = HERE / DATASETS[args.dataset][3] / args.model
    folder.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        rows = read_jsonl(folder / "predictions.jsonl")
        perf = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["performance"]
    else:
        rows, perf = (tfidf_predictions(reviews) if args.model == "tfidf" else
                      embedding_predictions(args.model, reviews, folder))
    assert [row["review_id"] for row in rows] == [row["review_id"] for row in reviews]
    metrics = summarize(rows, reviews, labels)
    if not args.evaluate_only:
        write_jsonl(folder / "predictions.jsonl", rows)
    (folder / "metrics.json").write_text(json.dumps({"performance": perf, "metrics": metrics},
                                            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.dataset, args.model, json.dumps({"all": metrics["all"],
          "ukrainian": metrics.get("ukrainian"), "english": metrics.get("english"),
          "performance": perf}, ensure_ascii=False), flush=True)
    assert hashlib.sha256((DATASETS[args.dataset][0] / DATASETS[args.dataset][1]).read_bytes()).hexdigest() == DATASETS[args.dataset][2]


if __name__ == "__main__":
    main()
