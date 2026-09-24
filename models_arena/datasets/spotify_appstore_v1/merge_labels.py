"""Merge the three independent text-only annotation batches; refuse ID mismatches."""

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    reviews = read_jsonl(HERE / "reviews.jsonl")
    batches = [
        read_jsonl(HERE / f"sentiment_labels_{start:03d}_{end:03d}.jsonl")
        for start, end in [(1, 130), (131, 260), (261, 391)]
    ]
    labels = [row for batch in batches for row in batch]
    assert len(reviews) == len(labels) == 391
    assert len({row["review_id"] for row in reviews}) == 391
    assert len({row["reviewId"] for row in labels}) == 391
    for review, label in zip(reviews, labels, strict=True):
        assert review["review_id"] == label["reviewId"]
        assert label["sentiment"] in {"positive", "neutral", "negative"}
        assert label["confidence"] in {"high", "medium", "low"}
        assert isinstance(label["ambiguous"], bool)
        assert isinstance(label["needs_manual_review"], bool)
    output = HERE / "sentiment_labels.jsonl"
    if output.exists():
        raise FileExistsError(output)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in labels) + "\n", encoding="utf-8")
    metadata_path = HERE / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["labeler"] = {
        "model": "GPT-6 Luna",
        "instructions_version": "text-only-sentiment-v1",
        "label_distribution": dict(Counter(row["sentiment"] for row in labels)),
        "ambiguous_count": sum(row["ambiguous"] for row in labels),
        "needs_manual_review_count": sum(row["needs_manual_review"] for row in labels),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["labeler"], indent=2))


if __name__ == "__main__":
    main()
