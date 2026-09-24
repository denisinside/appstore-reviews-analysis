"""Descriptive statistics for the review dictionaries returned by the scraper."""

from collections.abc import Mapping


def _optional_text(value: object, field: str, index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"review {index}: {field} must be a string or null")
    return value or None


def _add(groups: dict[str | None, tuple[int, int]], key: str | None, rating: int) -> None:
    count, total = groups.get(key, (0, 0))
    groups[key] = (count + 1, total + rating)


def _summarize(groups: dict[str | None, tuple[int, int]], field: str) -> list[dict]:
    rows = [
        {field: key, "review_count": count, "average_rating": total / count}
        for key, (count, total) in groups.items()
    ]
    rows.sort(key=lambda row: (-row["review_count"], row[field] is None, row[field] or ""))
    return rows


def calculate_metrics(reviews: list[Mapping[str, object]]) -> dict:
    """Calculate metrics for a list of already deduplicated reviews.

    Shares are fractions from 0 to 1. A review observed in several countries
    contributes to each country's statistics, but only once to global metrics.
    A missing country, language, or version is grouped under JSON null.
    Invalid ratings fail clearly instead of silently changing denominators.
    """
    if not isinstance(reviews, list):
        raise TypeError("reviews must be a list of review dictionaries")

    rating_counts = {rating: 0 for rating in range(1, 6)}
    countries: dict[str | None, tuple[int, int]] = {}
    languages: dict[str | None, tuple[int, int]] = {}
    versions: dict[str | None, tuple[int, int]] = {}
    rating_total = 0

    for index, review in enumerate(reviews):
        if not isinstance(review, Mapping):
            raise ValueError(f"review {index}: expected a dictionary")
        rating = review.get("rating")
        if type(rating) is not int or not 1 <= rating <= 5:
            raise ValueError(f"review {index}: rating must be an integer from 1 to 5")
        rating_counts[rating] += 1
        rating_total += rating

        origin = _optional_text(review.get("country"), "country", index)
        observed = review.get("observed_countries")
        if observed is not None and not isinstance(observed, list):
            raise ValueError(f"review {index}: observed_countries must be an array or null")
        country_codes = [origin] if origin else []
        for value in observed or []:
            code = _optional_text(value, "observed_countries item", index)
            if code and code not in country_codes:
                country_codes.append(code)
        for country in country_codes or [None]:
            _add(countries, country, rating)

        language = _optional_text(review.get("language"), "language", index)
        version = _optional_text(review.get("app_version"), "app_version", index)
        _add(languages, language, rating)
        _add(versions, version, rating)

    count = len(reviews)
    return {
        "review_count": count,
        "average_rating": rating_total / count if count else None,
        "rating_distribution": [
            {"rating": rating, "count": rating_counts[rating], "share": rating_counts[rating] / count if count else None}
            for rating in range(1, 6)
        ],
        "negative_rating_share": (rating_counts[1] + rating_counts[2]) / count if count else None,
        "country_statistics": _summarize(countries, "country"),
        "language_statistics": _summarize(languages, "language"),
        "rating_by_version": _summarize(versions, "app_version"),
    }
