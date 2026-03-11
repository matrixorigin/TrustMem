"""Sensitivity filter — tiered PII/credential handling for long-term memory.

Three tiers:
  HIGH   (passwords, API keys, private keys) → block entire memory
  MEDIUM (email, phone, SSN, credit card)    → redact in-place, keep memory structure
  LOW    (usernames)                          → allow through unchanged

Audit: blocked/redacted content is logged with content_hash (no raw content in logs).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SensitivityTier(str, Enum):
    HIGH = "high"      # block
    MEDIUM = "medium"  # redact
    LOW = "low"        # allow


# (label, tier, pattern, redact_replacement)
_PATTERNS: list[tuple[str, SensitivityTier, re.Pattern, str]] = [
    # HIGH — block
    ("aws_key",            SensitivityTier.HIGH,   re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}"), ""),
    ("private_key",        SensitivityTier.HIGH,   re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"), ""),
    ("bearer_token",       SensitivityTier.HIGH,   re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), ""),
    ("password_assignment",SensitivityTier.HIGH,   re.compile(r"(?:password|passwd|secret)\s*[:=]\s*\S+", re.IGNORECASE), ""),
    # MEDIUM — redact
    ("email",              SensitivityTier.MEDIUM, re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"), "[email]"),
    ("phone",              SensitivityTier.MEDIUM, re.compile(r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b"), "[phone]"),
    ("ssn",                SensitivityTier.MEDIUM, re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ssn]"),
    ("credit_card",        SensitivityTier.MEDIUM, re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[card]"),
]


@dataclass
class SensitivityResult:
    blocked: bool
    redacted_content: str | None  # None if not redacted; set to cleaned text if MEDIUM hits
    matched_labels: list[str] = field(default_factory=list)


def check_sensitivity(text_: str) -> SensitivityResult:
    """Classify and handle PII/credentials.

    Returns:
        SensitivityResult with:
          blocked=True           → caller must discard the memory
          redacted_content=str   → caller should use this cleaned text instead
          redacted_content=None  → content is safe as-is
    """
    content_hash = hashlib.sha256(text_.encode()).hexdigest()[:16]

    # Check HIGH tier first — any match blocks immediately
    for label, tier, pat, _ in _PATTERNS:
        if tier == SensitivityTier.HIGH and pat.search(text_):
            logger.warning(
                "sensitivity_blocked",
                extra={"label": label, "content_hash": content_hash},
            )
            return SensitivityResult(blocked=True, redacted_content=None, matched_labels=[label])

    # Check MEDIUM tier — redact all matches, keep memory
    redacted = text_
    medium_hits: list[str] = []
    for label, tier, pat, replacement in _PATTERNS:
        if tier == SensitivityTier.MEDIUM:
            new_text, n = pat.subn(replacement, redacted)
            if n:
                medium_hits.append(label)
                redacted = new_text

    if medium_hits:
        logger.info(
            "sensitivity_redacted",
            extra={"labels": medium_hits, "content_hash": content_hash},
        )
        return SensitivityResult(blocked=False, redacted_content=redacted, matched_labels=medium_hits)

    return SensitivityResult(blocked=False, redacted_content=None, matched_labels=[])
