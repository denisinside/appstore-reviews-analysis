"""Freeze a deterministic Spotify evaluation set from saved scraper outputs."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from appstore_reviews import prepare_review_text


SELECTION_VERSION = "spotify-selection-v1"
PREPROCESSING_VERSION = "nfc-whitespace-v1"
SETS = ("multilingual", "ukrainian", "english")


def fingerprint(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def valid_review(review: dict) -> bool:
    text = prepare_review_text(review)
    return (
        review.get("language") not in (None, "und")
        and isinstance(review.get("language_confidence"), (int, float))
        and review["language_confidence"] >= 0.5
        and len(text) >= 12
        and sum(char.isalpha() for char in text) >= 5
        and isinstance(review.get("review_id"), str)
        and bool(review["review_id"])
    )


def length_band(text: str) -> str:
    length = len(text)
    return "short" if length < 80 else "medium" if length < 250 else "long"


def near_duplicate(text: str, selected_texts: list[str]) -> bool:
    for previous in selected_texts:
        if abs(len(text) - len(previous)) > max(len(text), len(previous)) * 0.2:
            continue
        if SequenceMatcher(None, text.casefold(), previous.casefold()).ratio() >= 0.92:
            return True
    return False


def choose(candidates: list[dict], limit: int, used_ids: set, used_fingerprints: set) -> list[dict]:
    selected = []
    texts = []
    rating_counts = Counter()
    country_counts = Counter()
    length_counts = Counter()
    available = [
        (review, prepare_review_text(review), fingerprint(prepare_review_text(review)))
        for review in sorted(candidates, key=lambda item: item["review_id"])
    ]
    while len(selected) < limit:
        eligible = []
        for review, text, normalized in available:
            review_id = review["review_id"]
            if review_id in used_ids or normalized in used_fingerprints:
                continue
            score = (
                rating_counts[review["rating"]] + country_counts[review["country"]] + length_counts[length_band(text)],
                rating_counts[review["rating"]],
                country_counts[review["country"]],
                length_counts[length_band(text)],
                -float(review["language_confidence"]),
                hashlib.sha256(review_id.encode("utf-8")).hexdigest(),
            )
            eligible.append((score, review, text, normalized))
        if not eligible:
            break
        ranked = sorted(eligible, key=lambda item: item[0])
        chosen = next((item for item in ranked if not near_duplicate(item[2], texts)), None)
        if chosen is None:
            break
        _, review, text, normalized = chosen
        selected.append(review)
        texts.append(text)
        used_ids.add(review["review_id"])
        used_fingerprints.add(normalized)
        rating_counts[review["rating"]] += 1
        country_counts[review["country"]] += 1
        length_counts[length_band(text)] += 1
        available = [candidate for candidate in available if candidate[0]["review_id"] != review["review_id"]]
    return selected


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.dataset_dir
    output = directory / "reviews.jsonl"
    if output.exists():
        raise SystemExit(f"Frozen dataset already exists: {output}")

    top_path = directory / "source_top10.json"
    ua_path = directory / "source_ua.json"
    discovery_path = directory / "discovery.json"
    top = read_json(top_path)
    ua = read_json(ua_path)
    discovery = read_json(discovery_path)
    expected_countries = [row["country"] for row in discovery["countries"] if row["status"] == "success"][:10]
    actual_countries = [row["country"] for row in top["countries"]]
    if expected_countries != actual_countries or top["status"] != "success" or ua["status"] != "success":
        raise SystemExit("Sources do not match complete Discovery top ten and Ukrainian collections")
    if top["app_id"] != "324684580" or ua["app_id"] != "324684580":
        raise SystemExit("Sources must contain Spotify App Store ID 324684580")

    top_reviews = [review for review in top["reviews"] if valid_review(review)]
    ua_reviews = [review for review in ua["reviews"] if valid_review(review)]
    detected_languages = sorted({review["language"] for review in top["reviews"] if review["language"] != "und"})
    by_language = defaultdict(list)
    for review in top_reviews:
        by_language[review["language"]].append(review)
    # The top ten contain fewer than 20 Ukrainian texts. UA is a supplemental
    # supported storefront for the explicit Ukrainian evaluation requirement.
    by_language["uk"].extend(review for review in ua_reviews if review["language"] == "uk")

    used_ids = set()
    used_fingerprints = set()
    membership = defaultdict(set)
    selected_by_id = {}
    shortages = {}
    for language in detected_languages:
        chosen = choose(by_language[language], 20, used_ids, used_fingerprints)
        for review in chosen:
            membership[review["review_id"]].add("multilingual")
            selected_by_id[review["review_id"]] = review
        if len(chosen) < 20:
            shortages[language] = {"target": 20, "selected": len(chosen)}

    for language, set_name in (("uk", "ukrainian"), ("en", "english")):
        pool = [review for review in top_reviews + ua_reviews if review["language"] == language]
        chosen = choose(pool, 50, used_ids, used_fingerprints)
        for review in chosen:
            membership[review["review_id"]].add(set_name)
            selected_by_id[review["review_id"]] = review
        if len(chosen) < 50:
            shortages[set_name] = {"target": 50, "selected": len(chosen)}

    top_ids = {review["review_id"] for review in top["reviews"]}
    rows = []
    for review_id in sorted(selected_by_id):
        review = selected_by_id[review_id]
        row = dict(review)
        row["evaluation_sets"] = [name for name in SETS if name in membership[review_id]]
        row["source"] = {
            "collection": "discovery_top10" if review_id in top_ids else "supplemental_ua",
            "storefront": review["country"],
            "rss_page_range": "1-10",
        }
        rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    set_counts = Counter(name for row in rows for name in row["evaluation_sets"])
    language_counts = Counter(row["language"] for row in rows)
    country_counts = Counter(row["country"] for row in rows)
    metadata = {
        "dataset_id": "spotify_appstore_v1",
        "app_id": "324684580",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_version": SELECTION_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_countries_discovery_rank": expected_countries,
        "supplemental_country": "ua",
        "source_files_sha256": {
            "source_top10.json": sha256(top_path),
            "source_ua.json": sha256(ua_path),
            "discovery.json": sha256(discovery_path),
        },
        "source_review_counts": {"top10": top["review_count"], "ua": ua["review_count"]},
        "detected_languages_top10": detected_languages,
        "valid_candidates_top10_by_language": dict(sorted(Counter(review["language"] for review in top_reviews).items())),
        "selected_review_count": len(rows),
        "evaluation_set_counts": dict(set_counts),
        "language_counts": dict(sorted(language_counts.items())),
        "country_counts": dict(sorted(country_counts.items())),
        "shortages": shortages,
        "reviews_jsonl_sha256": sha256(output),
        "labeler": None,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("selected_review_count", "evaluation_set_counts", "language_counts", "shortages")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
