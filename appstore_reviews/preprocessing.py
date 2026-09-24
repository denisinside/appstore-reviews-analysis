"""Prepare a review's text for downstream analysis without changing the review."""

import re
import unicodedata
from collections.abc import Mapping


def normalize_text(text: str | None) -> str:
    """Use Unicode NFC and collapse whitespace in an analysis copy of text."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def prepare_review_text(review: Mapping[str, str | None]) -> str:
    """Join normalized title and body, keeping their boundary as a newline."""
    parts = (normalize_text(review.get("title")), normalize_text(review.get("text")))
    return "\n".join(part for part in parts if part)
