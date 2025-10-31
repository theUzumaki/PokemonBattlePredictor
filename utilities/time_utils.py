from datetime import datetime, timezone


def utc_iso_now() -> str:
    """Return current UTC time as an ISO-8601 string with a trailing Z.

    Uses a timezone-aware datetime under the hood to avoid deprecation warnings
    for datetime.utcnow().
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
