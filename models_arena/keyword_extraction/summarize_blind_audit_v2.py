"""Unblind and summarize independent held-out semantic judgments."""

import json
from collections import defaultdict
from pathlib import Path

from .benchmark import read_jsonl

DATA = Path(__file__).resolve().parents[1] / "datasets/heldout_crossapp_v1"
RESULTS = Path(__file__).resolve().parent / "results_heldout_v1"


def main():
    inputs = {row["reviewId"]: row for row in read_jsonl(DATA / "blind_audit_input.jsonl")}
    maps = {row["reviewId"]: row["aliases"] for row in read_jsonl(DATA / "blind_audit_mapping.jsonl")}
    judgments = read_jsonl(DATA / "blind_audit_judgments.jsonl")
    assert len(inputs) == len(maps) == len(judgments) == 40
    assert {row["reviewId"] for row in judgments} == set(inputs)
    totals = defaultdict(lambda: {"reviews": 0, "utility_sum": 0, "phrases": 0, "useful": 0,
                                "meaning_errors": 0, "negation_or_number_loss": 0,
                                "generic_or_duplicate": 0, "missed_major_complaints": 0,
                                "utility_by_language": defaultdict(list)})
    for row in judgments:
        rid = row["reviewId"]
        assert set(row["sets"]) == set("ABCD")
        for alias, judgment in row["sets"].items():
            model = maps[rid][alias]
            phrases = inputs[rid]["candidateSets"][alias]
            size = len(phrases)
            assert isinstance(judgment["utility"], int) and 0 <= judgment["utility"] <= 5
            for field in ("useful_indices", "meaning_error_indices", "negation_or_number_loss_indices", "generic_or_duplicate_indices"):
                indices = judgment[field]
                assert len(set(indices)) == len(indices) and all(isinstance(i, int) and 0 <= i < size for i in indices), (rid, alias, field)
            assert set(judgment["negation_or_number_loss_indices"]).issubset(judgment["meaning_error_indices"])
            item = totals[model]
            item["reviews"] += 1
            item["utility_sum"] += judgment["utility"]
            item["phrases"] += size
            item["useful"] += len(judgment["useful_indices"])
            item["meaning_errors"] += len(judgment["meaning_error_indices"])
            item["negation_or_number_loss"] += len(judgment["negation_or_number_loss_indices"])
            item["generic_or_duplicate"] += len(judgment["generic_or_duplicate_indices"])
            item["missed_major_complaints"] += len(judgment["missed_major_complaints"])
            item["utility_by_language"][inputs[rid]["language"]].append(judgment["utility"])
    output = {}
    for model, item in totals.items():
        assert item["reviews"] == 40
        output[model] = {"reviews": item["reviews"], "phrases": item["phrases"],
                         "mean_utility_0_to_5": round(item["utility_sum"] / 40, 4),
                         "useful_phrase_share": round(item["useful"] / item["phrases"], 4) if item["phrases"] else None,
                         "meaning_error_count": item["meaning_errors"],
                         "negation_or_number_loss_count": item["negation_or_number_loss"],
                         "generic_or_duplicate_count": item["generic_or_duplicate"],
                         "missed_major_complaint_count": item["missed_major_complaints"],
                         "utility_by_language": {language: round(sum(scores) / len(scores), 4)
                                                 for language, scores in item["utility_by_language"].items()}}
    (RESULTS / "blind_semantic_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
