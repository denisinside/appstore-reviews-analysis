"""Frozen Spotify keyword benchmark. Run with `python -m models_arena.keyword_extraction.benchmark`."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from appstore_reviews.keywords import KeywordExtractor, generate_candidates, normalize_phrase
from appstore_reviews.preprocessing import prepare_review_text

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "models_arena/datasets/spotify_appstore_v1"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LABEL_SHA256 = "dd1cae6c84e5bbc6b23eb4950652be79e48161b5aece015cacbd536333985e5a"
MODELS = {
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42", ""),
    "e5_small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "query: "),
    "bge_m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", ""),
}


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / 2**20, 1)
NEGATION = {"not", "no", "never", "cannot", "can't", "dont", "don't", "не", "ні", "немає", "без", "нет", "nicht", "kein", "pas", "sans", "sin", "non", "nie", "yok", "değil"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def load_dataset() -> tuple[list[dict], dict[str, dict]]:
    assert hashlib.sha256((DATA / "keyword_labels.jsonl").read_bytes()).hexdigest() == LABEL_SHA256, "Gold labels changed after freezing"
    labels = {row["reviewId"]: row for row in read_jsonl(DATA / "keyword_labels.jsonl")}
    reviews = [row for row in read_jsonl(DATA / "reviews.jsonl") if row["review_id"] in labels]
    assert len(reviews) == len(labels) == 255
    return reviews, labels


def _tokens(phrase: str) -> set[str]:
    return set(normalize_phrase(phrase).split())


def phrase_match(prediction: str, gold: str) -> float:
    """Conservative exact or near-exact token matching with negation guard."""
    a, b = _tokens(prediction), _tokens(gold)
    if not a or not b:
        return 0.0
    if (a & NEGATION) != (b & NEGATION):
        return 0.0
    if normalize_phrase(prediction) == normalize_phrase(gold):
        return 1.0
    overlap = len(a & b) / len(a | b)
    return overlap if len(a & b) >= 2 and overlap >= 0.8 else 0.0


def matched_count(predicted: list[str], gold: list[str]) -> int:
    edges = [[j for j, g in enumerate(gold) if phrase_match(p, g)] for p in predicted]
    owner = {}

    def visit(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or visit(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(visit(i, set()) for i in range(len(predicted)))


def review_metrics(prediction: dict, label: dict) -> dict:
    pred = [x["text"] for x in prediction["keywords"][:5]]
    gold = [x["text"] for x in label["keywords"]]
    if not pred and not gold:
        return {"precision_at_5": 1.0, "recall_at_5": 1.0, "f1_at_5": 1.0, "matches": 0}
    matches = matched_count(pred, gold)
    precision = matches / len(pred) if pred else 0.0
    recall = matches / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision_at_5": precision, "recall_at_5": recall, "f1_at_5": f1, "matches": matches}


def summarize(predictions: list[dict], reviews: list[dict], labels: dict[str, dict]) -> dict:
    review_map = {r["review_id"]: r for r in reviews}
    groups = defaultdict(list)
    for row in predictions:
        rid = row["review_id"]
        review = review_map[rid]
        metrics = review_metrics(row, labels[rid])
        for subset in review["evaluation_sets"]:
            groups[subset].append((row, metrics, labels[rid]))
            groups[subset + "/lang/" + str(review["language"])].append((row, metrics, labels[rid]))
        groups["all"].append((row, metrics, labels[rid]))
        if not labels[rid]["sentimentNeedsManualReview"]:
            groups["all_excluding_uncertain_sentiment"].append((row, metrics, labels[rid]))
            for subset in review["evaluation_sets"]:
                groups[subset + "_excluding_uncertain_sentiment"].append((row, metrics, labels[rid]))
    result = {}
    for name, rows in groups.items():
        result[name] = {"n": len(rows), "processed": sum(not row[0]["error"] for row in rows),
                        "gold_empty": sum(not row[2]["keywords"] for row in rows),
                        "uncertain_keyword_labels": sum(row[2]["needsHumanReview"] for row in rows)}
        for metric in ("precision_at_5", "recall_at_5", "f1_at_5"):
            result[name][metric] = round(sum(row[1][metric] for row in rows) / len(rows), 6)
    return result


def tfidf_baseline(reviews: list[dict]) -> tuple[list[dict], dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    started = time.perf_counter()
    documents = [prepare_review_text(review) for review in reviews]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)[^\W_]+(?:['’][^\W_]+)?", ngram_range=(1, 3),
                                 lowercase=True, sublinear_tf=True, min_df=1)
    try:
        matrix = vectorizer.fit_transform(documents)
        vocab = vectorizer.vocabulary_
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        matrix, vocab = None, {}
    loading = time.perf_counter() - started
    rows = []
    for i, review in enumerate(reviews):
        phrases = generate_candidates(documents[i], review.get("language"))
        ranked = []
        for phrase in phrases:
            key = normalize_phrase(phrase)
            column = vocab.get(key)
            score = float(matrix[i, column]) if matrix is not None and column is not None else 0.0
            if score > 0:
                ranked.append((phrase, score))
        ranked.sort(key=lambda item: (-item[1], -len(item[0].split()), item[0].casefold()))
        # Ignore near-identical candidate variants after a phrase was selected.
        chosen = []
        for phrase, score in ranked:
            if any(len(_tokens(phrase) & _tokens(previous)) / max(1, len(_tokens(phrase) | _tokens(previous))) >= 0.8
                   for previous, _ in chosen):
                continue
            chosen.append((phrase, score))
            if len(chosen) == 5:
                break
        rows.append({"review_id": review["review_id"], "language": review.get("language") or "und",
                     "evaluation_sets": review["evaluation_sets"],
                     "keywords": [{"text": phrase, "score": round(score, 6)} for phrase, score in chosen],
                     "error": None})
    total = time.perf_counter() - started
    return rows, {"load_seconds": round(loading, 3), "total_seconds": round(total, 3),
                  "mean_seconds_per_review": round(total / len(reviews), 6),
                  "rss_mb": _rss_mb()}


def model_predictions(name: str, reviews: list[dict]) -> tuple[list[dict], dict]:

    model_id, revision, prefix = MODELS[name]
    extractor = KeywordExtractor(model_id, revision, batch_size=32, max_candidates=80, prefix=prefix)
    start = time.perf_counter()
    extractor._ensure_loaded()
    loading = time.perf_counter() - start
    rows = extractor.extract_reviews(reviews)
    for row, review in zip(rows, reviews, strict=True):
        row["evaluation_sets"] = review["evaluation_sets"]
    total = time.perf_counter() - start
    perf = {"load_seconds": round(loading, 3), "total_seconds": round(total, 3),
            "mean_seconds_per_review": round((total-loading) / len(reviews), 6),
            "rss_mb": _rss_mb()}
    del extractor
    gc.collect()
    return rows, perf


def main():
    reviews, labels = load_dataset()
    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = os.environ.get("KEYWORD_ARENA_MODEL")
    evaluate_only = os.environ.get("KEYWORD_ARENA_EVALUATE_ONLY") == "1"
    names = [selected] if selected else ["tfidf", *MODELS]
    comparison = {}
    for name in names:
        if name not in {"tfidf", *MODELS}:
            raise ValueError(name)
        print(f"{'Evaluating' if evaluate_only else 'Running'} {name} on {len(reviews)} reviews", flush=True)
        folder = RESULTS / name
        if evaluate_only:
            rows = read_jsonl(folder / "predictions.jsonl")
            performance = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["performance"]
        else:
            rows, performance = tfidf_baseline(reviews) if name == "tfidf" else model_predictions(name, reviews)
        assert len(rows) == len(reviews) and [r["review_id"] for r in rows] == [r["review_id"] for r in reviews]
        metrics = summarize(rows, reviews, labels)
        if not evaluate_only:
            write_jsonl(folder / "predictions.jsonl", rows)
        (folder / "metrics.json").write_text(json.dumps({"performance": performance, "metrics": metrics},
                                                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        comparison[name] = {"performance": performance, "metrics": metrics}
        print(f"{name}: F1 all={metrics['all']['f1_at_5']}, uk={metrics['ukrainian']['f1_at_5']}, en={metrics['english']['f1_at_5']}", flush=True)
    if not selected:
        (RESULTS / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert hashlib.sha256((DATA / "keyword_labels.jsonl").read_bytes()).hexdigest() == LABEL_SHA256


if __name__ == "__main__":
    main()
