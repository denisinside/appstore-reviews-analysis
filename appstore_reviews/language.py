"""Lazy, once-per-instance fastText language identification."""

import logging
import math
import os
import threading
from pathlib import Path

from .config import MIN_LANGUAGE_CHARS, MIN_LANGUAGE_CONFIDENCE

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "lid.176.ftz"


class ModelLoadError(RuntimeError):
    pass


class LanguageDetector:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or os.getenv("FASTTEXT_MODEL_PATH") or DEFAULT_MODEL)
        self._model = None
        self._lock = threading.Lock()

    def ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path.is_file():
                raise ModelLoadError(
                    f"fastText model missing: {self.model_path}. "
                    "Download lid.176.ftz as described in README.md."
                )
            try:
                import fasttext

                self._model = fasttext.load_model(str(self.model_path))
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load fastText model {self.model_path}: {exc}. "
                    "Install the dependencies and download lid.176.ftz as described in README.md."
                ) from exc

    def detect(self, title: str | None, text: str | None):
        combined = " ".join(part for part in (title, text) if part)
        prediction_text = " ".join(combined.split())
        if len(prediction_text) < MIN_LANGUAGE_CHARS:
            return "und", None
        self.ensure_loaded()
        try:
            labels, scores = self._model.predict(prediction_text, k=1)
            confidence = float(scores[0])
            if not math.isfinite(confidence):
                return "und", None
            confidence = min(1.0, max(0.0, confidence))
            if confidence < MIN_LANGUAGE_CONFIDENCE:
                return "und", confidence
            return str(labels[0]).removeprefix("__label__"), confidence
        except (IndexError, TypeError, ValueError, RuntimeError) as exc:
            LOGGER.warning("Language detection failed: %s", exc)
            return "und", None
