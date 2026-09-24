"""Reproducible sentiment benchmark over the frozen Spotify App Store dataset.

Run with ``python -m models_arena.sentiment.benchmark --model xlm_roberta``.
Weights are fetched separately by ``download_models.py`` and kept in HF cache.
"""

import argparse
import json
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from appstore_reviews.preprocessing import prepare_review_text

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "datasets" / "spotify_appstore_v1"
LABELS = ("negative", "neutral", "positive")
MODELS = {
    "xlm_roberta": ("cardiffnlp/twitter-xlm-roberta-base-sentiment", "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"),
    "tabularis_multilingual": ("tabularisai/multilingual-sentiment-analysis", "eea032081f8d247b4303ef3565e7cec1b6f201c9"),
    "multilingual_distilbert": ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", "cf991100d706c13c0a080c097134c05b7f436c45"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def class_metrics(rows: list[dict]) -> dict:
    expected = [row for row in rows if row["prediction"] in LABELS]
    matrix = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for row in expected:
        matrix[row["gold"]][row["prediction"]] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "count": len(rows),
        "successful_predictions": len(expected),
        "coverage": len(expected) / len(rows) if rows else None,
        "accuracy": sum(row["gold"] == row["prediction"] for row in expected) / len(expected) if expected else None,
        "macro_f1": sum(per_class[label]["f1"] for label in LABELS) / 3 if expected else None,
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "rows": matrix},
    }


def summarize(rows: list[dict]) -> dict:
    sections = {
        set_name: class_metrics([row for row in rows if set_name in row["evaluation_sets"]])
        for set_name in ("multilingual", "ukrainian", "english")
    }
    multilingual = [row for row in rows if "multilingual" in row["evaluation_sets"]]
    by_language = {
        lang: class_metrics([row for row in multilingual if row["language"] == lang])
        for lang in sorted({row["language"] for row in multilingual})
    }
    sections["multilingual"]["mean_macro_f1_across_languages"] = (
        sum(value["macro_f1"] for value in by_language.values() if value["macro_f1"] is not None) / len(by_language)
        if by_language else None
    )
    return {"datasets": sections, "languages": by_language}


def three_scores(probabilities: list[float], id2label: dict) -> dict[str, float]:
    scores = dict.fromkeys(LABELS, 0.0)
    for idx, probability in enumerate(probabilities):
        label = str(id2label[idx]).lower().replace("_", " ").strip()
        if label in ("very negative", "negative"):
            target = "negative"
        elif label in ("very positive", "positive"):
            target = "positive"
        elif label == "neutral":
            target = "neutral"
        else:
            raise ValueError(f"unknown model label {idx}: {id2label[idx]}")
        scores[target] += probability
    return scores


def run(slug: str, batch_size: int = 8, max_length: int = 256) -> dict:
    model_id, revision = MODELS[slug]
    reviews = read_jsonl(DATASET / "reviews.jsonl")
    labels = {row["reviewId"]: row for row in read_jsonl(DATASET / "sentiment_labels.jsonl")}
    assert len(reviews) == len(labels) == 391
    torch.set_num_threads(4)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True,
        use_fast=slug != "xlm_roberta",  # Cardiff ships SentencePiece, not a fast tokenizer artifact.
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, local_files_only=True)
    model.eval()
    load_seconds = time.perf_counter() - load_start
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    text_by_id = {row["review_id"]: prepare_review_text(row) for row in reviews}
    # Warmup uses the same input for every model and is excluded from timing.
    warmup = tokenizer([text_by_id[reviews[0]["review_id"]]], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.inference_mode():
        model(**warmup)
    predictions = {}
    inference_seconds = 0.0
    errors = []
    for offset in range(0, len(reviews), batch_size):
        batch = reviews[offset:offset + batch_size]
        texts = [text_by_id[row["review_id"]] for row in batch]
        started = time.perf_counter()
        try:
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits
            raw = torch.softmax(logits, dim=-1).tolist()
            elapsed = time.perf_counter() - started
            inference_seconds += elapsed
            for row, probs in zip(batch, raw, strict=True):
                scores = three_scores(probs, id2label)
                predictions[row["review_id"]] = {"prediction": max(LABELS, key=scores.get), "scores": scores, "error": None}
        except Exception as exc:
            elapsed = time.perf_counter() - started
            inference_seconds += elapsed
            for row in batch:
                predictions[row["review_id"]] = {"prediction": None, "scores": None, "error": str(exc)}
                errors.append({"review_id": row["review_id"], "error": str(exc)})
    result_rows = [
        {
            "review_id": row["review_id"],
            "language": row["language"],
            "evaluation_sets": row["evaluation_sets"],
            "gold": labels[row["review_id"]]["sentiment"],
            **predictions[row["review_id"]],
        }
        for row in reviews
    ]
    results = summarize(result_rows)
    results.update({
        "model_id": model_id,
        "revision": revision,
        "id2label": id2label,
        "library_versions": {name: version(name) for name in ("torch", "transformers", "huggingface-hub", "tokenizers")},
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "device": "cpu", "torch_threads": torch.get_num_threads()},
        "inference": {
            "batch_size": batch_size,
            "max_length_tokens": max_length,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "average_inference_seconds_per_review": inference_seconds / len(reviews),
            "attempted_reviews": len(reviews),
            "successful_predictions": len(reviews) - len(errors),
            "errors": errors,
        },
        "gold_distribution": dict(Counter(row["gold"] for row in result_rows)),
    })
    out = HERE / "results" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows) + "\n", encoding="utf-8")
    write_json(out / "metrics.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    result = run(args.model, args.batch_size, args.max_length)
    print(json.dumps({"model_id": result["model_id"], "datasets": result["datasets"], "inference": result["inference"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
