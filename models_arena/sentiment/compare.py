"""Combine completed model outputs and extract reviewable disagreement examples."""

import json
from pathlib import Path

from appstore_reviews.preprocessing import prepare_review_text

from .benchmark import DATASET, HERE, MODELS, read_jsonl, write_json


def main() -> None:
    reviews = read_jsonl(DATASET / "reviews.jsonl")
    labels = {row["reviewId"]: row for row in read_jsonl(DATASET / "sentiment_labels.jsonl")}
    metrics = {slug: json.loads((HERE / "results" / slug / "metrics.json").read_text(encoding="utf-8")) for slug in MODELS}
    predictions = {
        slug: {row["review_id"]: row for row in read_jsonl(HERE / "results" / slug / "predictions.jsonl")}
        for slug in MODELS
    }
    comparison = {
        "dataset_id": "spotify_appstore_v1",
        "model_ids": {slug: data["model_id"] for slug, data in metrics.items()},
        "multilingual": {slug: data["datasets"]["multilingual"] for slug, data in metrics.items()},
        "ukrainian": {slug: data["datasets"]["ukrainian"] for slug, data in metrics.items()},
        "english": {slug: data["datasets"]["english"] for slug, data in metrics.items()},
        "per_language": {
            language: {slug: data["languages"].get(language) for slug, data in metrics.items()}
            for language in sorted({row["language"] for row in reviews if "multilingual" in row["evaluation_sets"]})
        },
        "timing": {slug: data["inference"] for slug, data in metrics.items()},
    }
    write_json(HERE / "results" / "comparison.json", comparison)

    examples = []
    for review in reviews:
        review_id = review["review_id"]
        gold = labels[review_id]["sentiment"]
        predicted = {slug: predictions[slug][review_id]["prediction"] for slug in MODELS}
        wrong = [slug for slug, value in predicted.items() if value != gold]
        if not wrong:
            continue
        examples.append({
            "review_id": review_id,
            "language": review["language"],
            "evaluation_sets": review["evaluation_sets"],
            "title": review["title"],
            "text": review["text"],
            "analysis_text": prepare_review_text(review),
            "gold": gold,
            "gold_ambiguous": labels[review_id]["ambiguous"],
            "needs_manual_review": labels[review_id]["needs_manual_review"],
            "predictions": predicted,
            "wrong_models": wrong,
            "all_three_wrong": len(wrong) == 3,
            "negative_missed": gold == "negative" and bool(wrong),
        })
    examples.sort(key=lambda item: (
        not item["all_three_wrong"],
        not item["negative_missed"],
        item["language"] not in ("uk", "en"),
        item["review_id"],
    ))
    write_json(HERE / "results" / "error_examples.json", {
        "total_reviews_with_any_disagreement": len(examples),
        "all_three_wrong": sum(row["all_three_wrong"] for row in examples),
        "negative_missed_by_at_least_one": sum(row["negative_missed"] for row in examples),
        "examples": examples,
    })
    print(json.dumps({"disagreements": len(examples), "all_three_wrong": sum(row["all_three_wrong"] for row in examples)}))


if __name__ == "__main__":
    main()
