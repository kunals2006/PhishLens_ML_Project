"""URL-only lexical feature extraction for the PhishLens prototype."""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd
import tldextract

_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_SUSPICIOUS = ("login", "verify", "update", "secure", "account", "confirm", "bank", "password", "signin", "webscr")


def normalize_url(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        value = "http://" + value
    return value


def registrable_domain(value: str) -> str:
    parsed = urlsplit(normalize_url(value))
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        return _extract(host).top_domain_under_public_suffix or host
    except Exception:
        return host


def _is_ip(host: str) -> int:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return 1
    except ValueError:
        return 0


def extract_one(value: str) -> dict:
    url = normalize_url(value)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path, query = parsed.path or "", parsed.query or ""
    domain = registrable_domain(url)
    ext = _extract(host)
    try:
        has_port = int(parsed.port is not None) if host else 0
    except ValueError:
        has_port = 0
    letters = [c for c in url.lower() if c.isalpha()]
    counts = Counter(url)
    entropy = -sum((n / max(len(url), 1)) * math.log2(n / max(len(url), 1)) for n in counts.values())
    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "count_dots": url.count("."),
        "count_hyphens": url.count("-"),
        "count_at": url.count("@"),
        "count_question": url.count("?"),
        "count_ampersand": url.count("&"),
        "count_equal": url.count("="),
        "count_digits": sum(c.isdigit() for c in url),
        "count_subdomains": max(0, len(host.split(".")) - len(domain.split("."))) if domain else len(host.split(".")),
        "has_ip_address": _is_ip(host),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_port": has_port,
        "has_punycode": int("xn--" in host),
        "has_suspicious_word": int(any(word in url.lower() for word in _SUSPICIOUS)),
        "suspicious_word_count": sum(url.lower().count(word) for word in _SUSPICIOUS),
        "has_percent_encoding": int("%" in url),
        "has_double_slash_in_path": int("//" in path),
        "tld_length": len(ext.suffix),
        "letter_ratio": sum(c.isalpha() for c in url) / max(len(url), 1),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "url_entropy": entropy,
    }


def extract_many(values) -> pd.DataFrame:
    return pd.DataFrame([extract_one(value) for value in values])
