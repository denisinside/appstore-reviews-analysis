"""Recompute and save v2 comparisons from frozen prediction files."""

import json
import random
from pathlib import Path

from .benchmark import MODELS, read_jsonl, review_metrics, summarize
from .benchmark_v2 import DATASETS, HERE, load_dataset

METHODS = ("tfidf", *MODELS)


def subset_f1(rows: list[dict], labels: dict[str, dict], keep) -> dict:
    selected = [review_metrics(row, labels[row["review_id"]])["f1_at_5"] for row in rows if keep(row)]
    return {"n": len(selected), "f1_at_5": round(sum(selected) / len(selected), 6) if selected else None}


def paired_bootstrap_difference(first: list[dict], second: list[dict], labels: dict[str, dict], *, seed=20260924):
    """95% percentile interval for paired mean F1(first − second)."""
    a = [review_metrics(row, labels[row["review_id"]])["f1_at_5"] for row in first]
    b = [review_metrics(row, labels[row["review_id"]])["f1_at_5"] for row in second]
    differences = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    draws = sorted(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
                   for _ in range(5000))
    return [round(draws[124], 6), round(draws[4874], 6)]


def compare(name: str) -> dict:
    reviews, labels = load_dataset(name)
    root = HERE / DATASETS[name][3]
    predictions = {}
    comparison = {}
    for method in METHODS:
        folder = root / method
        rows = read_jsonl(folder / "predictions.jsonl")
        assert [r["review_id"] for r in rows] == [r["review_id"] for r in reviews]
        predictions[method] = rows
        performance = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["performance"]
        comparison[method] = {"performance": performance, "metrics": summarize(rows, reviews, labels),
                              "gold_confident": subset_f1(rows, labels, lambda row: not labels[row["review_id"]]["needsHumanReview"])}
        if name == "heldout":
            review_map = {review["review_id"]: review for review in reviews}
            comparison[method]["by_app"] = {app: subset_f1(rows, labels, lambda row, app=app: review_map[row["review_id"]]["app_name"] == app)
                                            for app in ("Duolingo", "Telegram")}
    if name == "old":
        comparison["v1_rescored_on_v2_gold"] = {}
        for method in METHODS:
            rows = read_jsonl(HERE / "results" / method / "predictions.jsonl")
            comparison["v1_rescored_on_v2_gold"][method] = summarize(rows, reviews, labels)
    else:
        for first, second in (("e5_small", "minilm"), ("bge_m3", "e5_small"), ("bge_m3", "minilm")):
            comparison.setdefault("paired_bootstrap_f1_difference_ci95", {})[f"{first}_minus_{second}"] = paired_bootstrap_difference(
                predictions[first], predictions[second], labels)
    (root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return comparison


if __name__ == "__main__":
    for dataset in ("old", "heldout"):
        result = compare(dataset)
        print(dataset)
        for method in METHODS:
            metric = result[method]["metrics"]
            print(method, {key: metric.get(key, {}).get("f1_at_5") for key in ("all", "ukrainian", "english", "multilingual", "multilingual_other")})
        if dataset == "heldout":
            print("paired bootstrap", result["paired_bootstrap_f1_difference_ci95"])
