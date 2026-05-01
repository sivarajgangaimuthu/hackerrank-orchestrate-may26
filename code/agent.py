"""End-to-end support ticket triage agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classifier import TicketClassifier
from corpus import CorpusDocument, load_corpus
from responder import Responder
from retrieval import Retriever
from risk import RiskAssessor


class TicketTriageAgent:
    """Reusable support triage pipeline with cached corpus and retrieval index."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        top_k: int = 3,
        use_gemini: bool = True,
    ) -> None:
        self.top_k = top_k
        self.documents: list[CorpusDocument] = load_corpus(data_dir)
        self.retriever = Retriever(self.documents)
        self.classifier = TicketClassifier()
        self.risk_assessor = RiskAssessor()
        self.responder = Responder(use_gemini=use_gemini)

    def process_ticket(
        self,
        ticket_text: str,
        subject: str = "",
        company: str | None = None,
    ) -> dict[str, str]:
        """Process a support ticket and return the evaluator output fields."""

        combined_text = self._combine_ticket_text(ticket_text, subject)
        normalized_company = self._normalize_company(company)

        retrieved = self.retriever.retrieve(combined_text, top_k=self.top_k)
        retrieved_documents = [result.document for result in retrieved]
        classification = self.classifier.classify(
            combined_text,
            provided_company=normalized_company,
            retrieved_documents=retrieved_documents,
        )
        risk = self.risk_assessor.decide(combined_text)
        response = self.responder.generate(
            combined_text,
            retrieved,
            classification,
            risk,
        )

        return {
            "status": risk.status,
            "product_area": classification.product_area,
            "response": response.response,
            "justification": response.justification,
            "request_type": classification.request_type,
        }

    @staticmethod
    def _combine_ticket_text(ticket_text: str, subject: str = "") -> str:
        subject = subject.strip()
        body = ticket_text.strip()
        if subject and body:
            return f"{subject}\n\n{body}"
        return subject or body

    @staticmethod
    def _normalize_company(company: str | None) -> str | None:
        if company is None:
            return None
        normalized = company.strip()
        if not normalized or normalized.lower() == "none":
            return None
        known_companies = {
            "hackerrank": "HackerRank",
            "claude": "Claude",
            "visa": "Visa",
        }
        return known_companies.get(normalized.lower(), normalized)


def _print_result(ticket: dict[str, Any], result: dict[str, str]) -> None:
    print("=" * 80)
    print(f"Subject: {ticket.get('subject', '')}")
    print(f"Company: {ticket.get('company')}")
    print(f"Issue: {ticket.get('ticket_text', '')}")
    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    samples = (
        {
            "subject": "Payment failed",
            "company": "Visa",
            "ticket_text": "My Visa payment failed at a merchant checkout.",
        },
        {
            "subject": "Account security concern",
            "company": None,
            "ticket_text": "I think my account was hacked and someone accessed it without permission.",
        },
        {
            "subject": "Feature request",
            "company": "HackerRank",
            "ticket_text": "Can you add support for custom assessment branding?",
        },
        {
            "subject": "Conversation error",
            "company": "Claude",
            "ticket_text": "Claude shows an error every time I open an old conversation.",
        },
    )

    agent = TicketTriageAgent()
    for sample in samples:
        output = agent.process_ticket(
            sample["ticket_text"],
            subject=sample["subject"],
            company=sample["company"],
        )
        _print_result(sample, output)
