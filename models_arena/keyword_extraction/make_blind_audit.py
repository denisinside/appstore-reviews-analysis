"""Deterministically anonymize a held-out sample for independent semantic audit."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .benchmark import read_jsonl, write_jsonl
from .benchmark_v2 import HELDOUT_SETUP_EXCLUSIONS

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets/heldout_crossapp_v1"
MODELS = ("tfidf", "minilm", "e5_small", "bge_m3")
SEED = 20260924


def sample_ids():
    reviews = read_jsonl(DATA / "reviews.jsonl")
    labels = {row["reviewId"]: row for row in read_jsonl(DATA / "keyword_labels.jsonl")}
    by_language = defaultdict(list)
    for row in reviews:
        if labels[row["review_id"]]["sentiment"] == "negative" and row["review_id"] not in HELDOUT_SETUP_EXCLUSIONS:
            by_language[row["language"]].append(row["review_id"])
    rng = random.Random(SEED)
    ids = []
    for language in ("uk", "en", "pl", "ru"):
        ids.extend(rng.sample(by_language[language], 10))
    return ids


def main():
    ids = sample_ids()
    (DATA / "blind_audit_sample_ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reviews = {row["review_id"]: row for row in read_jsonl(DATA / "reviews.jsonl")}
    predictions = {name: {row["review_id"]: row for row in read_jsonl(HERE / "results_heldout_v1" / name / "predictions.jsonl")}
                   for name in MODELS}
    rng = random.Random(SEED + 1)
    blinded = []
    mapping = []
    for rid in ids:
        review = reviews[rid]
        names = list(MODELS)
        rng.shuffle(names)
        aliases = dict(zip("ABCD", names, strict=True))
        blinded.append({"reviewId": rid, "language": review["language"], "appName": review["app_name"],
                        "title": review["title"], "text": review["text"],
                        "candidateSets": {alias: [{"text": item["text"], "evidenceSpan": item.get("evidence_span")}
                                                  for item in predictions[name][rid]["keywords"]]
                                          for alias, name in aliases.items()}})
        mapping.append({"reviewId": rid, "aliases": aliases})
    write_jsonl(DATA / "blind_audit_input.jsonl", blinded)
    write_jsonl(DATA / "blind_audit_mapping.jsonl", mapping)
    for name in ("blind_audit_input.jsonl", "blind_audit_mapping.jsonl", "blind_audit_sample_ids.json"):
        print(name, hashlib.sha256((DATA / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
