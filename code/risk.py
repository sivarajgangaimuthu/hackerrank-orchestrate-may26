"""Risk assessment and escalation policy for support tickets."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


REPLIED = "replied"
ESCALATED = "escalated"


@dataclass(frozen=True)
class RiskRule:
    """Escalation rule for sensitive or high-risk support tickets."""

    category: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RiskDecision:
    """Status decision with matched risk reasons."""

    status: str
    reasons: tuple[str, ...]

    @property
    def should_escalate(self) -> bool:
        return self.status == ESCALATED


RISK_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "fraud",
        (
            "fraud",
            "fraudulent",
            "scam",
            "phishing",
            "identity theft",
            "stolen card",
        ),
    ),
    RiskRule(
        "hacked or stolen account",
        (
            "hacked",
            "compromised",
            "stolen account",
            "account stolen",
            "taken over",
            "takeover",
        ),
    ),
    RiskRule(
        "unauthorized access",
        (
            "unauthorized",
            "unauthorised",
            "someone accessed",
            "not me",
            "without permission",
            "unknown login",
            "suspicious login",
        ),
    ),
    RiskRule(
        "payment dispute",
        (
            "payment dispute",
            "dispute",
            "chargeback",
            "wrong charge",
            "duplicate charge",
            "refund dispute",
            "unauthorized charge",
        ),
    ),
    RiskRule(
        "legal threat",
        (
            "lawsuit",
            "sue",
            "legal action",
            "attorney",
            "lawyer",
            "court",
            "regulator",
        ),
    ),
    RiskRule(
        "sensitive financial issue",
        (
            "bank account",
            "card number",
            "credit card number",
            "cvv",
            "ssn",
            "tax id",
            "financial information",
            "large unauthorized transaction",
        ),
    ),
)


def normalize_text(*parts: str | None) -> str:
    """Normalize ticket fields for deterministic risk matching."""

    return " ".join(part.strip().lower() for part in parts if part and part.strip())


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def matched_risk_categories(
    ticket_text: str,
    rules: Sequence[RiskRule] = RISK_RULES,
) -> tuple[str, ...]:
    """Return all escalation categories matched by the ticket."""

    normalized = normalize_text(ticket_text)
    matches: list[str] = []

    for rule in rules:
        if any(_contains_keyword(normalized, keyword) for keyword in rule.keywords):
            matches.append(rule.category)

    return tuple(matches)


def assess_risk(ticket_text: str) -> RiskDecision:
    """Decide whether the ticket can be replied to or must be escalated."""

    reasons = matched_risk_categories(ticket_text)
    if reasons:
        return RiskDecision(status=ESCALATED, reasons=reasons)
    return RiskDecision(status=REPLIED, reasons=())


class RiskAssessor:
    """Reusable escalation policy wrapper."""

    def decide(self, ticket_text: str) -> RiskDecision:
        return assess_risk(ticket_text)


if __name__ == "__main__":
    samples = (
        "My Visa payment failed at checkout.",
        "I think my account was hacked and someone accessed it without permission.",
        "I want to dispute a duplicate charge on my card.",
        "If this is not fixed I will sue and contact my attorney.",
        "Can you add dark mode to the assessment dashboard?",
    )

    assessor = RiskAssessor()
    for text in samples:
        decision = assessor.decide(text)
        reason_text = ", ".join(decision.reasons) if decision.reasons else "none"
        print(f"Ticket: {text}")
        print(f"  status={decision.status}")
        print(f"  reasons={reason_text}")
