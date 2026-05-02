"""Append-only ticket processing logs for the hackathon contract."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re


SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^'\"\s,;]+"),
    re.compile(r"(?i)\b(bearer)\s+[a-z0-9._~+/=-]{12,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"),
)


def log_path() -> Path:
    """Return the required shared hackathon log path."""

    return Path.home() / "hackerrank_orchestrate" / "log.txt"


def ensure_log_file(path: Path | None = None) -> Path:
    """Create the parent directory and log file if missing."""

    target = path or log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=True)
    return target


def redact_secrets(text: str | None) -> str:
    """Redact obvious secrets before writing ticket content to logs."""

    if text is None:
        return ""

    redacted = str(text)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    return redacted


def _redact_match(match: re.Match[str]) -> str:
    value = match.group(0)
    prefix_match = re.match(r"(?i)^([^:=\s]+(?:\s*[:=]\s*| \s*))", value)
    if prefix_match:
        return f"{prefix_match.group(1)}[REDACTED]"
    return "[REDACTED]"


def _summarize_response(response: str, limit: int = 180) -> str:
    clean = " ".join(redact_secrets(response).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def append_ticket_log(
    *,
    subject: str,
    company: str | None,
    issue: str,
    request_type: str,
    product_area: str,
    status: str,
    response: str,
    path: Path | None = None,
) -> None:
    """Append one processed-ticket entry to the shared log file."""

    target = ensure_log_file(path)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    entry = "\n".join(
        (
            f"## [{timestamp}] TICKET PROCESSED",
            "",
            f"Subject: {redact_secrets(subject)}",
            f"Company: {redact_secrets(company or '')}",
            "Issue:",
            redact_secrets(issue),
            "",
            f"Request Type: {request_type}",
            f"Product Area: {product_area}",
            f"Status: {status}",
            f"Response Summary: {_summarize_response(response)}",
            "",
        )
    )

    with target.open("a", encoding="utf-8", newline="\n") as file:
        file.write(entry)
