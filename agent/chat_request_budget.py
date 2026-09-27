"""Explicit per-endpoint/model request budgets for authenticated dashboard sessions.

A model's advertised context window does not establish the account's request quota.
No provider limit is guessed; absent entries retain normal model discovery.
"""
from urllib.parse import urlsplit


def dashboard_request_budget(config, model, base_url):
    section = config.get("dashboard", {}) if isinstance(config, dict) else {}
    limits = section.get("request_token_limits", {}) if isinstance(section, dict) else {}
    try:
        host = urlsplit(base_url or "").hostname
    except (ValueError, TypeError):
        return None
    endpoint = limits.get(host, {}) if isinstance(limits, dict) else {}
    value = endpoint.get(model) if isinstance(endpoint, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and 1024 <= value <= 10_000_000:
        return value
    return None
