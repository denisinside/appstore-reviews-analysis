"""Optional batch sentiment inference for collected App Store reviews."""

import threading
from collections.abc import Mapping

from .preprocessing import prepare_review_text

DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEFAULT_SENTIMENT_REVISION = "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"
CLASSES = ("negative", "neutral", "positive")


class SentimentModelError(RuntimeError):
    """The optional pretrained sentiment model could not be loaded."""


def _combine_scores(probabilities: list[float], id2label: Mapping[int, str]) -> dict[str, float]:
    scores = dict.fromkeys(CLASSES, 0.0)
    for index, probability in enumerate(probabilities):
        label = str(id2label[index]).lower().replace("_", " ").strip()
        if label in ("very negative", "negative"):
            sentiment = "negative"
        elif label in ("very positive", "positive"):
            sentiment = "positive"
        elif label == "neutral":
            sentiment = "neutral"
        else:
            raise ValueError(f"Unknown sentiment label {index}: {label}")
        scores[sentiment] += float(probability)
    return scores


class SentimentAnalyzer:
    """Load one pinned checkpoint lazily and classify reviews in batches.

    ``analyze_reviews`` returns separate result records and leaves the reviews
    unchanged. The rating and other metadata never enter model inference.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_SENTIMENT_MODEL,
        revision: str = DEFAULT_SENTIMENT_REVISION,
        *,
        batch_size: int = 8,
        max_length: int = 256,
        device: str = "auto",
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    # This pinned Cardiff checkpoint ships SentencePiece, not tokenizer.json.
                    use_fast=self.model_id != DEFAULT_SENTIMENT_MODEL,
                )
                model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=self.revision)
                model.to(device).eval()
            except Exception as exc:
                raise SentimentModelError(
                    "Could not load the sentiment model. Install the optional sentiment "
                    "dependencies with `python -m pip install -e '.[sentiment]'` "
                    "and check access to the Hugging Face checkpoint. "
                    f"Details: {exc}"
                ) from exc
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self.device = device

    def _classify_texts(self, texts: list[str]) -> list[dict[str, float]]:
        self._ensure_loaded()
        inputs = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
        id2label = {int(index): label for index, label in self._model.config.id2label.items()}
        return [_combine_scores(row, id2label) for row in probabilities]

    def analyze_reviews(self, reviews: list[Mapping[str, object]]) -> list[dict]:
        if not isinstance(reviews, list):
            raise TypeError("reviews must be a list")
        output = []
        pending = []
        for index, review in enumerate(reviews):
            if not isinstance(review, Mapping):
                raise ValueError(f"review {index}: expected a dictionary")
            review_id = review.get("review_id")
            if not isinstance(review_id, str) or not review_id:
                raise ValueError(f"review {index}: missing review_id")
            text = prepare_review_text(review)
            output.append({
                "review_id": review_id,
                "sentiment": None,
                "sentiment_scores": None,
                "sentiment_model": {"id": self.model_id, "revision": self.revision},
                "error": "empty text" if not text else None,
            })
            if text:
                pending.append((index, text))
        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset:offset + self.batch_size]
            try:
                scores_list = self._classify_texts([text for _, text in batch])
                if len(scores_list) != len(batch):
                    raise ValueError("model returned an unexpected number of predictions")
                for (index, _), scores in zip(batch, scores_list, strict=True):
                    output[index]["sentiment_scores"] = scores
                    output[index]["sentiment"] = max(CLASSES, key=scores.get)
            except SentimentModelError:
                raise
            except Exception:
                # A single malformed review must not discard the rest of a batch.
                for index, text in batch:
                    try:
                        scores = self._classify_texts([text])[0]
                        output[index]["sentiment_scores"] = scores
                        output[index]["sentiment"] = max(CLASSES, key=scores.get)
                    except Exception as exc:
                        output[index]["error"] = str(exc)
        return output

    def analyze_review(self, review: Mapping[str, object]) -> dict:
        return self.analyze_reviews([review])[0]
