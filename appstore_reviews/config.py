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

COUNTRY_NAMES = {
    "ua": "Ukraine", "pl": "Poland", "cz": "Czechia", "ro": "Romania",
    "md": "Moldova", "sk": "Slovakia", "hu": "Hungary", "lt": "Lithuania",
    "lv": "Latvia", "ee": "Estonia", "bg": "Bulgaria", "kz": "Kazakhstan",
    "ge": "Georgia", "tr": "Turkey", "gb": "United Kingdom", "de": "Germany",
    "fr": "France", "es": "Spain", "it": "Italy", "nl": "Netherlands",
    "be": "Belgium", "ch": "Switzerland", "at": "Austria", "se": "Sweden",
    "no": "Norway", "dk": "Denmark", "fi": "Finland", "ie": "Ireland",
    "pt": "Portugal", "us": "United States", "ca": "Canada", "mx": "Mexico",
}

MAX_RSS_PAGES = 10
MAX_CONCURRENT_REQUESTS = 3
HTTP_TIMEOUT_SECONDS = 10.0
HTTP_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.3
CACHE_TTL_SECONDS = 21600
EMPTY_PAGE_TTL_SECONDS = 300
MIN_LANGUAGE_CHARS = 10
MIN_LANGUAGE_CONFIDENCE = 0.5

# Stable two-level taxonomy for LLM issue/aspect analysis. The second level
# (specific aspect) is extracted dynamically from each original review.
ASPECT_TAXONOMY_VERSION = "1.1"
ASPECT_CATEGORIES = (
    "stability", "performance", "ui_ux", "accounts_auth", "payments_billing",
    "pricing_subscriptions", "features", "content", "notifications",
    "privacy_security", "support", "localization_access", "ads_monetization", "other",
)
