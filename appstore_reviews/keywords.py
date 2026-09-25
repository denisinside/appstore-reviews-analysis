"""Source-grounded multilingual keyphrases from a review title and body.

Only review language and text are inputs. Ratings and sentiment are deliberately
outside this extractor; callers may filter negative reviews before calling it.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from .preprocessing import prepare_review_text

DEFAULT_KEYWORD_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_KEYWORD_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_CANDIDATES = 120
GENERIC_WORDS = {"app", "application", "spotify", "music", "bad", "good", "problem", "problems"}
NEGATIONS = {"not", "no", "never", "cannot", "can't", "dont", "don't", "didn't", "doesn't", "won't", "without", "не", "ні", "немає", "без", "нет", "ни", "kein", "keine", "nicht", "pas", "sans", "sin", "non", "nie", "yok", "değil", "ne", "negalima", "neįmanoma", "nevisada"}
TOKEN_RE = re.compile(r"\d+(?:[,.]\d+)+|[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[,.]\d+)*\Z", re.UNICODE)
BREAK_RE = re.compile(r"[.!?;:\n\r,—–]|https?://|www\.", re.IGNORECASE)
PREFIXES = NEGATIONS | {"only", "just", "too", "much", "many", "very", "really", "still", "even", "same", "last", "after", "before", "every", "all", "always", "constantly", "almost", "often", "again", "were", "was", "are", "is", "лише", "тільки", "занадто", "після", "кожного", "кожної", "постійно", "всього", "вже", "дуже", "все", "багато", "завжди", "хоча", "по", "po", "beveik", "tik", "daug", "za", "zbyt", "tylko", "после", "слишком"}
END_CONNECTORS = {"and", "or", "but", "if", "that", "which", "the", "a", "an", "of", "to", "for", "with", "in", "on", "at", "by", "from", "і", "й", "та", "або", "але", "що", "якщо", "це", "у", "в", "з", "на", "до", "за", "через", "то", "и", "или", "но", "как", "если", "что", "de", "la", "le", "et", "que", "der", "die", "das", "und", "po", "su", "ir", "kad", "bei"}
QUANTIFIERS = {"only", "just", "too", "much", "many", "very", "really", "still", "even", "every", "all", "almost", "лише", "тільки", "кожного", "кожної", "після", "всього", "багато", "завжди", "тик", "tik", "beveik", "po", "tylko", "после"}
COORDINATORS = {"and", "but", "or", "і", "й", "але", "або", "и", "но", "или", "ir", "bet", "und", "et", "y", "pero", "oraz"}
PHRASAL_ENDINGS = {"out", "off", "up", "down", "in"}
PHRASAL_VERBS = {"log", "logs", "logged", "sign", "signed", "check", "checked", "cuts", "cut", "turn", "turns", "turned", "shuts", "shut", "drops", "drop"}
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "first", "second", "once", "twice", "один", "одна", "два", "дві", "двох", "три", "чотири", "п'ять", "п’ять", "раз", "одного", "одну", "однієї", "два", "три", "пять", "двух", "один", "jedna", "dwa", "trzy", "du", "deux", "trois"}
MEASUREMENT_UNITS = {"zł", "pln", "usd", "eur", "uah", "грн", "gb", "mb", "kb", "tb", "гб", "мб", "хв", "min", "mins", "minutes", "hours", "days"}
DISCOURSE_START = {"ну", "then", "well", "so", "anyway", "якщо", "if", "а", "and", "but", "or", "але", "і", "й", "або", "и", "но", "или", "ir", "bet", "und", "et", "y", "pero", "oraz"}
EVIDENCE_BREAK_RE = re.compile(r"[.!?;:\n\r,—–]|\b(?:and|but|or|але|або|і|й|и|но|или|ir|bet|und|et|pero|oraz)\b", re.IGNORECASE)
PRAISE = {"great", "good", "amazing", "excellent", "love", "nice", "perfect", "impressive", "cool", "bearable", "чудовий", "чудова", "гарний", "гарна", "подобається", "класний", "класна", "супер"}
POSITIVE_PREDICATES = {"work", "works", "working", "працюють", "работают", "funciona", "funktioniert"}
RESTRICTIVE_WORDS = NEGATIONS | {"only", "barely", "hardly", "poorly", "badly", "sometimes", "rarely", "лише", "тільки", "погано", "іноді", "редко", "только", "schlecht"}
NEGATED_SUBCLAUSE_RE = re.compile(r"(?:nepasakyčiau|wouldn't say|не скажу|не думаю)\s*,?\s*(?:kad|that|що)\s+", re.IGNORECASE)


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


def _is_negation(word: str) -> bool:
    return word in NEGATIONS or (word.startswith("ne") and word in {"nepasakyčiau", "negalima", "neįmanoma", "nevisada"})


def _is_number(word: str) -> bool:
    return bool(NUMBER_RE.fullmatch(word)) or word in NUMBER_WORDS


def _clause_span(text: str, start: int, end: int) -> str:
    left = start
    right = len(text)
    for match in EVIDENCE_BREAK_RE.finditer(text):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return text[left:right].strip()


def _externally_negated(text: str, start: int) -> bool:
    """Conservative guard for 'wouldn't say that X is good' fragments."""
    for match in NEGATED_SUBCLAUSE_RE.finditer(text):
        if start < match.end():
            continue
        tail = text[match.end():start]
        if not BREAK_RE.search(tail):
            return True
    return False


def generate_candidate_records(text: str, language: str | None, *, limit: int = MAX_CANDIDATES) -> list[dict]:
    """Generate short, exact source spans without crossing punctuation.

    Prefix guards keep a nearby negation or numeric restriction attached to its
    subject. This is intentionally conservative: unclear snippets are excluded.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    tokens = list(TOKEN_RE.finditer(text))
    stops = _stopwords(language)
    scored: dict[str, tuple[float, int, dict]] = {}
    for i, first in enumerate(tokens):
        for n in range(1, 7):
            if i + n > len(tokens):
                continue
            chunk = tokens[i:i+n]
            if any(BREAK_RE.search(text[a.end():b.start()]) for a, b in zip(chunk, chunk[1:])):
                continue
            words = [m.group().casefold() for m in chunk]
            if words[0] in DISCOURSE_START or any(word in COORDINATORS for word in words[1:-1]):
                continue
            # A price or size is incomplete without the unit that immediately
            # follows it in the source: prefer a shorter window containing both.
            if _is_number(words[-1]) and i + n < len(tokens):
                next_token = tokens[i+n]
                if (next_token.group().casefold() in MEASUREMENT_UNITS
                        and not BREAK_RE.search(text[chunk[-1].end():next_token.start()])):
                    continue
            if _externally_negated(text, first.start()) and not any(_is_negation(w) for w in words):
                continue
            evidence = _clause_span(text, first.start(), chunk[-1].end())
            evidence_words = [word.casefold() for word in TOKEN_RE.findall(evidence)]
            if "bearable" in evidence_words and not any(_is_negation(w) for w in evidence_words):
                continue
            evidence_restricted = any(w in RESTRICTIVE_WORDS for w in evidence_words) or any(
                pair == ("через", "раз") for pair in zip(evidence_words, evidence_words[1:]))
            if (any(w in POSITIVE_PREDICATES for w in evidence_words)
                    and not evidence_restricted):
                continue
            if any(word in PRAISE for word in words) and not any(_is_negation(w) for w in words):
                continue
            # Suppress only plainly working aspects; preserve restricted operation
            # such as "not working" and "працюють через раз".
            restricted_operation = any(word in POSITIVE_PREDICATES for word in words) and (
                any(word in RESTRICTIVE_WORDS for word in words)
                or any(pair == ("через", "раз") for pair in zip(words, words[1:]))
            )
            if any(word in POSITIVE_PREDICATES for word in words) and not restricted_operation:
                continue
            if all((word in stops and word not in MEASUREMENT_UNITS) or word in GENERIC_WORDS or _is_number(word) for word in words) and not restricted_operation:
                continue
            phrasal_end = words[-1] in PHRASAL_ENDINGS and any(w in PHRASAL_VERBS for w in words[:-1])
            if words[-1] in END_CONNECTORS and not phrasal_end:
                continue
            if words[-1] in stops and words[-1] not in MEASUREMENT_UNITS and not phrasal_end and not _is_negation(words[-1]) and not restricted_operation:
                continue
            signal = any(_is_negation(w) or _is_number(w) or w in QUANTIFIERS for w in words)
            if words[0] in stops and words[0] not in PREFIXES and not (words[0] in {"i", "я", "you", "ми"} and any(_is_negation(w) for w in words)):
                continue
            if n == 1 and (len(words[0]) < 4 and words[0] != "ads" or words[0] in GENERIC_WORDS or words[0] in stops):
                continue
            # A nearby negator or numeric limit changes the meaning of a phrase.
            # Do not emit an affirmative-looking fragment from that scope.
            previous = []
            for prior in reversed(tokens[max(0, i-3):i]):
                if BREAK_RE.search(text[prior.end():first.start()]):
                    break
                previous.insert(0, prior.group().casefold())
            if any(_is_negation(w) for w in previous) and not any(_is_negation(w) for w in words):
                continue
            if any(_is_number(w) for w in previous[-2:]) and not any(_is_number(w) for w in words):
                continue
            if previous and previous[-1] in QUANTIFIERS and previous[-1] not in words:
                continue
            phrase = text[first.start():chunk[-1].end()]
            key = normalize_phrase(phrase)
            if not key or len(key) < 3:
                continue
            informative = sum((w not in stops or w in MEASUREMENT_UNITS) and w not in GENERIC_WORDS and not _is_number(w) for w in words)
            if n > 1 and informative < 2 and not signal and words[-1] not in PHRASAL_ENDINGS:
                continue
            priority = informative + 0.24 * min(n, 4) + (0.8 if signal else 0.0) - 0.16 * max(0, n - 4)
            record = {"text": phrase, "source_start": first.start(), "source_end": chunk[-1].end(),
                      "evidence_span": evidence,
                      "priority": round(priority, 4)}
            value = (priority, first.start(), record)
            if key not in scored or value[0] > scored[key][0]:
                scored[key] = value
    if len(scored) <= limit:
        return [v[2] for v in sorted(scored.values(), key=lambda v: v[1])]
    # Keep early and late topics in long reviews. Reserve informative clauses.
    buckets: list[list[tuple[float, int, dict]]] = [[], [], [], []]
    for value in scored.values():
        bucket = min(3, 4 * value[1] // max(1, len(text)))
        buckets[bucket].append(value)
    for bucket in buckets:
        bucket.sort(key=lambda v: (-v[0], v[1], v[2]["text"].casefold()))
    chosen = []
    while len(chosen) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(chosen) < limit:
                chosen.append(bucket.pop(0))
    return [v[2] for v in sorted(chosen, key=lambda v: v[1])]


def generate_candidates(text: str, language: str | None, *, limit: int = MAX_CANDIDATES) -> list[str]:
    """Compatibility helper returning only candidate strings."""
    return [row["text"] for row in generate_candidate_records(text, language, limit=limit)]


def select_keywords(records: list[dict], relevance, *, top_k: int, vectors=None) -> list[dict]:
    """Rank grounded spans and suppress redundant keyphrase variants."""
    if not records:
        return []
    selected: list[int] = []
    remaining = set(range(len(records)))
    token_sets = [set(normalize_phrase(row["text"]).split()) for row in records]
    qualifiers = [{w for w in tokens if _is_negation(w) or _is_number(w) or w in QUANTIFIERS} for tokens in token_sets]
    while remaining and len(selected) < top_k:
        def score(index: int) -> tuple[float, float, int]:
            tokens = token_sets[index]
            length = len(tokens)
            specificity = min(4, length) / 4 - 0.08 * max(0, length - 4)
            signal = bool(qualifiers[index])
            merit = float(relevance[index]) + 0.055 * specificity + (0.035 if signal else 0.0)
            merit -= 0.055 * sum(records[index]["evidence_span"] == records[j]["evidence_span"] for j in selected)
            redundancy = 0.0
            for j in selected:
                if qualifiers[index] != qualifiers[j]:
                    continue
                overlap = len(tokens & token_sets[j]) / max(1, min(len(tokens), len(token_sets[j])))
                if overlap >= 0.75:
                    redundancy = max(redundancy, 0.33 * overlap)
                elif vectors is not None and overlap >= 0.4:
                    redundancy = max(redundancy, 0.08 * max(0.0, float(vectors[index] @ vectors[j])))
            return (merit - redundancy, float(relevance[index]), -index)
        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
        for i in tuple(remaining):
            if qualifiers[i] != qualifiers[best]:
                continue
            overlap = len(token_sets[i] & token_sets[best]) / max(1, min(len(token_sets[i]), len(token_sets[best])))
            if overlap >= 0.75:
                remaining.remove(i)
    return [{"text": records[i]["text"], "score": round(float(relevance[i]), 6),
             "evidence_span": records[i]["evidence_span"],
             "source_start": records[i]["source_start"], "source_end": records[i]["source_end"]} for i in selected]


def _rank(records: list[dict], vectors, *, top_k: int) -> list[dict]:
    if not records:
        return []
    candidates = vectors[1:]
    return select_keywords(records, candidates @ vectors[0], top_k=top_k, vectors=candidates)


class KeywordExtractor:
    """Reusable batch extractor; the checkpoint is loaded once per instance."""

    def __init__(self, model_id: str = DEFAULT_KEYWORD_MODEL, revision: str = DEFAULT_KEYWORD_REVISION,
                 *, batch_size: int = 32, max_keywords: int = 5, max_candidates: int = MAX_CANDIDATES,
                 device: str = "cpu", prefix: str | None = None) -> None:
        if any(not isinstance(x, int) or x < 1 for x in (batch_size, max_keywords, max_candidates)):
            raise ValueError("batch_size, max_keywords and max_candidates must be positive integers")
        self.model_id, self.revision = model_id, revision
        self.batch_size, self.max_keywords, self.max_candidates = batch_size, min(max_keywords, 5), max_candidates
        self.device = device
        self.prefix = ("query: " if model_id == "intfloat/multilingual-e5-small" else "") if prefix is None else prefix
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
            phrases = generate_candidate_records(text, language if isinstance(language, str) else None,
                                                 limit=self.max_candidates)
            output.append({"review_id": review_id, "language": language or "und", "keywords": [],
                           "keyword_model": {"id": self.model_id, "revision": self.revision}, "error": None})
            if phrases:
                work.append((index, text, phrases))
        if not work:
            return output
        self._ensure_loaded()
        # Encode the query and candidate phrases for all reviews together. Keep
        # each review's slice so ranking remains review-local and unchanged.
        flattened, spans = [], []
        for index, text, phrases in work:
            start = len(flattened)
            flattened.extend([self.prefix + text] + [self.prefix + p["text"] for p in phrases])
            spans.append((index, phrases, start, len(flattened)))
        try:
            vectors = self._model.encode(flattened, batch_size=self.batch_size,
                                         normalize_embeddings=True, show_progress_bar=False)
            for index, phrases, start, end in spans:
                output[index]["keywords"] = _rank(phrases, vectors[start:end], top_k=self.max_keywords)
        except KeywordModelError:
            raise
        except Exception:
            # Preserve the previous per-review error isolation if one combined
            # inference call fails for a model/runtime-specific reason.
            for index, text, phrases in work:
                try:
                    vectors = self._model.encode([self.prefix + text] + [self.prefix + p["text"] for p in phrases],
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
