"""Small, fixed settings for the public App Store RSS feed."""

EASTERN_EUROPE = (
    "ua", "pl", "cz", "ro", "md", "sk", "hu",
    "lt", "lv", "ee", "bg", "kz", "ge", "tr",
)
WESTERN_EUROPE = (
    "gb", "de", "fr", "es", "it", "nl", "be",
    "ch", "at", "se", "no", "dk", "fi", "ie", "pt",
)
NORTH_AMERICA = ("us", "ca", "mx")
SUPPORTED_COUNTRIES = EASTERN_EUROPE + WESTERN_EUROPE + NORTH_AMERICA

MAX_RSS_PAGES = 10
MAX_CONCURRENT_REQUESTS = 3
HTTP_TIMEOUT_SECONDS = 10.0
HTTP_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.3
CACHE_TTL_SECONDS = 21600
EMPTY_PAGE_TTL_SECONDS = 300
MIN_LANGUAGE_CHARS = 10
MIN_LANGUAGE_CONFIDENCE = 0.5
