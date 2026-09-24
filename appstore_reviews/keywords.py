"""Local multilingual keyphrase extraction from review title and body.

The method ranks source n-grams by similarity to the whole review. It does not
generate phrases, and never uses rating or sentiment as an input feature.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from .preprocessing import prepare_review_text

DEFAULT_KEYWORD_MODEL = "BAAI/bge-m3"
DEFAULT_KEYWORD_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MAX_CANDIDATES = 80
GENERIC_WORDS = {"app", "application", "spotify", "music", "bad", "good", "problem", "problems"}
NEGATIONS = {"not", "no", "never", "cannot", "can't", "dont", "don't", "не", "ні", "немає", "без", "нет", "ни", "kein", "keine", "nicht", "pas", "sans", "sin", "non", "nie", "yok", "değil"}
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
BREAK_RE = re.compile(r"[.!?;:\n\r]|https?://|www\.", re.IGNORECASE)


class KeywordModelError(RuntimeError):
    """The optional embedding model could not be loaded."""


def normalize_phrase(phrase: str) -> str:
    """Conservative language-independent surface normalization."""
    phrase = unicodedata.normalize("NFC", phrase).casefold()
    return " ".join(TOKEN_RE.findall(phrase))


def _stopwords(language: str | None) -> set[str]:
    try:
        import stopwordsiso

        return set(stopwordsiso.stopwords((language or "").split("-")[0]) or ())
    except ImportError as exc:
        raise KeywordModelError("Install keyword dependencies with `python -m pip install -e '.[keywords]'`.") from exc


def generate_candidates(text: str, language: str | None, *, limit: int = MAX_CANDIDATES) -> list[str]:
    """Extract source spans of 1–3 Unicode word tokens, preserving negation."""
    if not isinstance(text, str) or not text.strip():
        return []
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    tokens = list(TOKEN_RE.finditer(text))
    stops = _stopwords(language)
    scored: dict[str, tuple[float, int, str]] = {}
    for i, first in enumerate(tokens):
        for n in (1, 2, 3):
            if i + n > len(tokens):
                continue
            chunk = tokens[i:i+n]
            if any(BREAK_RE.search(text[a.end():b.start()]) for a, b in zip(chunk, chunk[1:])):
                continue
            words = [m.group().casefold() for m in chunk]
            if any(word.isdecimal() for word in words):
                continue
            if all(word in stops or word in GENERIC_WORDS for word in words):
                continue
            if n > 1 and (words[0] in stops or words[-1] in stops) and not any(w in NEGATIONS for w in words):
                continue
            phrase = text[first.start():chunk[-1].end()]
            key = normalize_phrase(phrase)
            if not key or len(key) < 3:
                continue
            # Fixed before evaluation: favor substantive phrases, while keeping
            # early and late parts of long reviews in the candidate pool.
            informative = sum(w not in stops and w not in GENERIC_WORDS for w in words)
            score = informative + 0.2 * (n - 1)
            value = (score, first.start(), phrase)
            if key not in scored or value[:2] > scored[key][:2]:
                scored[key] = value
    if len(scored) <= limit:
        return [v[2] for v in sorted(scored.values(), key=lambda v: v[1])]
    # Round-robin position buckets prevent long reviews from losing later topics.
    buckets: list[list[tuple[float, int, str]]] = [[], [], [], []]
    for value in scored.values():
        bucket = min(3, 4 * value[1] // max(1, len(text)))
        buckets[bucket].append(value)
    for bucket in buckets:
        bucket.sort(key=lambda v: (-v[0], v[1], v[2].casefold()))
    chosen = []
    while len(chosen) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(chosen) < limit:
                chosen.append(bucket.pop(0))
    return [v[2] for v in sorted(chosen, key=lambda v: v[1])]


def _rank(phrases: list[str], vectors, *, top_k: int) -> list[dict]:
    """Cosine similarity with fixed MMR diversity (normalized vectors)."""
    import numpy as np

    if not phrases:
        return []
    document = vectors[0]
    candidates = vectors[1:]
    relevance = candidates @ document
    selected: list[int] = []
    remaining = set(range(len(phrases)))
    while remaining and len(selected) < top_k:
        def score(index: int) -> tuple[float, float, int]:
            redundancy = max((float(candidates[index] @ candidates[j]) for j in selected), default=0.0)
            return (0.75 * float(relevance[index]) - 0.25 * redundancy, float(relevance[index]), -index)
        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return [{"text": phrases[i], "score": round(float(relevance[i]), 6)} for i in selected]


class KeywordExtractor:
    """Reusable batch extractor; the checkpoint is loaded once per instance."""

    def __init__(self, model_id: str = DEFAULT_KEYWORD_MODEL, revision: str = DEFAULT_KEYWORD_REVISION,
                 *, batch_size: int = 32, max_keywords: int = 5, max_candidates: int = MAX_CANDIDATES,
                 device: str = "cpu", prefix: str = "") -> None:
        if any(not isinstance(x, int) or x < 1 for x in (batch_size, max_keywords, max_candidates)):
            raise ValueError("batch_size, max_keywords and max_candidates must be positive integers")
        self.model_id, self.revision = model_id, revision
        self.batch_size, self.max_keywords, self.max_candidates = batch_size, min(max_keywords, 5), max_candidates
        self.device, self.prefix = device, prefix
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_id, revision=self.revision, device=self.device)
            except Exception as exc:
                raise KeywordModelError(
                    "Could not load the keyword model. Install `python -m pip install -e '.[keywords]'` "
                    "and check the pinned checkpoint. " + str(exc)
                ) from exc

    def extract_reviews(self, reviews: list[Mapping[str, object]]) -> list[dict]:
        if not isinstance(reviews, list):
            raise TypeError("reviews must be a list")
        output, work = [], []
        for index, review in enumerate(reviews):
            if not isinstance(review, Mapping):
                raise ValueError(f"review {index}: expected a dictionary")
            review_id = review.get("review_id")
            if not isinstance(review_id, str) or not review_id:
                raise ValueError(f"review {index}: missing review_id")
            # Collected reviews use `text`; accept `content` from callers using
            # Apple's feed naming without changing the shared preprocessing.
            language = review.get("language")
            source = review if review.get("text") not in (None, "") else {**review, "text": review.get("content")}
            try:
                text = prepare_review_text(source)
            except (TypeError, ValueError) as exc:
                output.append({"review_id": review_id, "language": language or "und", "keywords": [],
                               "keyword_model": {"id": self.model_id, "revision": self.revision},
                               "error": f"invalid review text: {exc}"})
                continue
            phrases = generate_candidates(text, language if isinstance(language, str) else None,
                                          limit=self.max_candidates)
            output.append({"review_id": review_id, "language": language or "und", "keywords": [],
                           "keyword_model": {"id": self.model_id, "revision": self.revision}, "error": None})
            if phrases:
                work.append((index, text, phrases))
        if not work:
            return output
        self._ensure_loaded()
        for index, text, phrases in work:
            try:
                vectors = self._model.encode([self.prefix + text] + [self.prefix + p for p in phrases],
                                             batch_size=self.batch_size, normalize_embeddings=True,
                                             show_progress_bar=False)
                output[index]["keywords"] = _rank(phrases, vectors, top_k=self.max_keywords)
            except KeywordModelError:
                raise
            except Exception as exc:
                output[index]["error"] = str(exc)
        return output

    def extract_review(self, review: Mapping[str, object]) -> dict:
        return self.extract_reviews([review])[0]


def aggregate_keywords(extractions: Sequence[Mapping[str, object]], *, example_limit: int = 5) -> dict[str, list[dict]]:
    """Count one occurrence per review and language; keep conservative surface forms."""
    if not isinstance(example_limit, int) or example_limit < 1:
        raise ValueError("example_limit must be positive")
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    denominator: dict[str, set[str]] = defaultdict(set)
    for record in extractions:
        if record.get("error"):
            continue
        review_id = record.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            continue
        language = str(record.get("language") or "und")
        denominator[language].add(review_id)
        for item in record.get("keywords") or []:
            phrase = str(item.get("text") or "")
            key = normalize_phrase(phrase)
            if not key:
                continue
            group = groups[language].setdefault(key, {"phrase": phrase, "review_ids": set(), "scores": []})
            if review_id in group["review_ids"]:
                continue
            group["review_ids"].add(review_id)
            score = item.get("score")
            if isinstance(score, (int, float)) and math.isfinite(score):
                group["scores"].append(float(score))
    result = {}
    for language, phrases in groups.items():
        rows = []
        for key, group in phrases.items():
            count = len(group["review_ids"])
            rows.append({"phrase": group["phrase"], "normalized_phrase": key,
                         "review_count": count, "share": round(count / len(denominator[language]), 6),
                         "average_score": round(sum(group["scores"]) / len(group["scores"]), 6) if group["scores"] else None,
                         "example_review_ids": sorted(group["review_ids"])[:example_limit]})
        result[language] = sorted(rows, key=lambda row: (-row["review_count"], row["normalized_phrase"]))
    return result
