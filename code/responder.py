"""Grounded response generation for the support triage agent.

The responder can call Gemini through the REST API when ``GEMINI_API_KEY`` is
available. If the key is missing or the API call fails, it falls back to a
deterministic local response so tests and batch runs remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Iterable
from urllib import parse, request
from urllib.error import HTTPError, URLError

from classifier import ClassificationResult, TicketClassifier
from corpus import CorpusDocument, load_corpus
from retrieval import RetrievalResult, Retriever
from risk import ESCALATED, REPLIED, RiskAssessor, RiskDecision


DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_CONTEXT_CHARS_PER_DOC = 900
MAX_FALLBACK_SENTENCES = 3

METADATA_PATTERNS = (
    re.compile(r"\blast\s+(modified|updated)\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}", re.IGNORECASE),
    re.compile(r"\b\d{1,2}:\d{2}\s*(am|pm)\b", re.IGNORECASE),
    re.compile(r"\b(last updated|created|modified)\s*:\s*", re.IGNORECASE),
)


@dataclass(frozen=True)
class ResponseResult:
    """User-facing response and traceable decision justification."""

    response: str
    justification: str


@dataclass(frozen=True)
class GroundingDocument:
    """Normalized document context passed to the responder."""

    company: str
    filepath: str
    content: str
    score: float | None = None


def _document_from_item(item: CorpusDocument | RetrievalResult | GroundingDocument) -> GroundingDocument:
    if isinstance(item, GroundingDocument):
        return item

    if isinstance(item, RetrievalResult):
        return GroundingDocument(
            company=item.document.company,
            filepath=item.document.filepath,
            content=item.document.content,
            score=item.score,
        )

    return GroundingDocument(
        company=item.company,
        filepath=item.filepath,
        content=item.content,
        score=None,
    )


def _normalize_documents(
    retrieved_documents: Iterable[CorpusDocument | RetrievalResult | GroundingDocument],
) -> list[GroundingDocument]:
    return [_document_from_item(item) for item in retrieved_documents]


def _trim_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _clean_support_text(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(
        r"Last modified:\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Last updated:\s*[^.]+(?:ago|\d{4})?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b", " ", text)
    text = re.sub(r"\.\d+Z\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_support_text(text)
    rough_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences: list[str] = []
    for sentence in rough_sentences:
        sentence = sentence.strip(" -")
        if not sentence:
            continue
        if any(pattern.search(sentence) for pattern in METADATA_PATTERNS):
            continue
        if len(sentence.split()) < 5 or len(sentence) > 260:
            continue
        sentences.append(sentence)
    return sentences


def _keywords_from_ticket(ticket_text: str) -> set[str]:
    stop_words = {
        "about",
        "after",
        "again",
        "because",
        "can",
        "could",
        "from",
        "have",
        "help",
        "into",
        "issue",
        "please",
        "support",
        "that",
        "this",
        "with",
        "would",
    }
    words = re.findall(r"[a-z0-9]+", ticket_text.lower())
    return {word for word in words if len(word) > 2 and word not in stop_words}


def _sentence_score(sentence: str, keywords: set[str], index: int) -> tuple[int, int]:
    normalized = sentence.lower()
    overlap = sum(1 for keyword in keywords if keyword in normalized)
    return overlap, -index


def _is_clean_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    blocked_fragments = (
        "last modified",
        "last updated",
        "overview",
        "learn more",
        "where to next",
        "related articles",
        "table of contents",
    )
    if any(fragment in lowered for fragment in blocked_fragments):
        return False
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b|\.?\d+z\b", lowered):
        return False
    return True


def _summarize_documents(
    ticket_text: str,
    documents: list[GroundingDocument],
    classification: ClassificationResult,
) -> str:
    keywords = _keywords_from_ticket(ticket_text)
    candidates: list[tuple[tuple[int, int], str]] = []

    for document in documents:
        for index, sentence in enumerate(_split_sentences(document.content[:2500])):
            if not _is_clean_sentence(sentence):
                continue
            score = _sentence_score(sentence, keywords, index)
            if score[0] > 0:
                candidates.append((score, sentence))

    if not candidates:
        area = classification.product_area.replace(" and ", " or ")
        return (
            f"The available {classification.company} guidance appears most related "
            f"to {area}, but it does not provide enough specific detail to give a confident "
            f"step-by-step answer."
        )

    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in ranked:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        selected.append(sentence.rstrip("."))
        seen.add(normalized)
        if len(selected) == MAX_FALLBACK_SENTENCES:
            break

    return ". ".join(selected) + "."


def _grounded_topic(documents: list[GroundingDocument], classification: ClassificationResult) -> str:
    if not documents:
        return classification.product_area

    path_parts = " ".join(document.filepath.lower().replace("\\", "/") for document in documents[:3])
    if "payment" in path_parts or "transaction" in path_parts or "interchange" in path_parts:
        return "payments and transactions"
    if "assessment" in path_parts or "interview" in path_parts or "candidate" in path_parts:
        return "assessments and interviews"
    if "account" in path_parts or "login" in path_parts or "security" in path_parts:
        return "account access and security"
    if "billing" in path_parts or "invoice" in path_parts or "subscription" in path_parts:
        return "billing and plans"
    if "api" in path_parts or "integration" in path_parts:
        return "api and integrations"
    return classification.product_area


def _clean_fallback_response(
    ticket_text: str,
    documents: list[GroundingDocument],
    classification: ClassificationResult,
) -> str:
    company = classification.company if classification.company != "None" else "the relevant product"
    topic = _grounded_topic(documents, classification)

    if classification.request_type == "feature_request":
        return (
            f"Thanks for the suggestion. Based on the available {company} guidance, this seems related "
            f"to {topic}, but it does not confirm that this feature is currently available. "
            "I will treat this as a feature request rather than promising existing support."
        )

    if "payment" in ticket_text.lower() or topic == "payments and transactions":
        return (
            f"Thanks for reaching out. The available {company} documentation points this "
            "to payments and transactions, but it does not provide a guaranteed fix for this "
            "specific failed payment. Please retry the payment details you control and work "
            "with the merchant or card issuer for transaction-specific help."
        )

    if topic == "account access and security":
        return (
            f"Thanks for reaching out. Based on the available {company} support guidance, "
            "this is an account access or security issue. Please use the product's documented "
            "account recovery or sign-in flow and avoid sharing passwords or verification codes "
            "in the ticket."
        )

    if classification.request_type == "bug":
        return (
            f"Thanks for reaching out. The available documentation suggests this is related "
            f"to {topic}, but it does not confirm a specific outage or root cause for this issue. "
            "Please try the documented product flow again and share the exact error details with "
            "support if it continues."
        )

    return (
        f"Thanks for reaching out. Based on the available {company} support guidance, this is related to "
        f"{topic}, but it does not provide enough detail for a more specific answer. I can only "
        "share guidance that is supported by the provided documentation."
    )


def _format_context(documents: list[GroundingDocument]) -> str:
    if not documents:
        return "No retrieved support documents were available."

    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        score_text = "unknown" if document.score is None else f"{document.score:.4f}"
        snippet = _trim_text(document.content, MAX_CONTEXT_CHARS_PER_DOC)
        parts.append(
            "\n".join(
                (
                    f"[Document {index}]",
                    f"Company: {document.company}",
                    f"Source: {document.filepath}",
                    f"Similarity: {score_text}",
                    f"Content: {snippet}",
                )
            )
        )
    return "\n\n".join(parts)


def _fallback_response(
    ticket_text: str,
    documents: list[GroundingDocument],
    classification: ClassificationResult,
    risk: RiskDecision,
) -> ResponseResult:
    reasons = ", ".join(risk.reasons) if risk.reasons else "none"

    if risk.status == ESCALATED:
        response = (
            "Thanks for reaching out. This issue needs review by a support specialist "
            "because it may involve sensitive account, legal, payment, or financial risk. "
            "Please avoid sharing passwords, full card numbers, CVV codes, or other secrets "
            "in this ticket while the specialist reviews it."
        )
        justification = (
            f"Escalated because the ticket matched risk category: {reasons}. "
            f"Classified as {classification.request_type} in {classification.product_area} "
            f"for {classification.company}."
        )
        return ResponseResult(response=response, justification=justification)

    if not documents:
        response = (
            "Thanks for contacting support. I could not find enough relevant information "
            "in the provided support documentation to answer this confidently, so this "
            "should be reviewed by the support team."
        )
        justification = (
            "No retrieved support documents were available, so a grounded direct answer "
            "could not be produced without risking unsupported claims."
        )
        return ResponseResult(response=response, justification=justification)

    top_doc = documents[0]
    response = _clean_fallback_response(ticket_text, documents, classification)
    if top_doc.score is not None:
        justification = (
            f"Replied because no escalation risk was detected and the response is grounded "
            f"in the top retrieved document for {classification.product_area}. "
            f"Classified as {classification.request_type}; top similarity={top_doc.score:.4f}."
        )
    else:
        justification = (
            f"Replied because no escalation risk was detected and relevant documentation "
            f"was retrieved for {classification.product_area}. Classified as "
            f"{classification.request_type}."
        )
    return ResponseResult(response=response, justification=justification)


class Responder:
    """Generate concise grounded responses and decision justifications."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        use_gemini: bool = True,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", model)
        self.use_gemini = use_gemini
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        ticket_text: str,
        retrieved_documents: Iterable[CorpusDocument | RetrievalResult | GroundingDocument],
        classification: ClassificationResult,
        risk: RiskDecision,
    ) -> ResponseResult:
        """Generate the support response and justification."""

        documents = _normalize_documents(retrieved_documents)
        if not self.use_gemini or not self.api_key:
            return _fallback_response(ticket_text, documents, classification, risk)

        prompt = self._build_prompt(ticket_text, documents, classification, risk)
        try:
            result = self._call_gemini(prompt)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return _fallback_response(ticket_text, documents, classification, risk)

        response = str(result.get("response", "")).strip()
        justification = str(result.get("justification", "")).strip()
        if not response or not justification:
            return _fallback_response(ticket_text, documents, classification, risk)

        return ResponseResult(response=response, justification=justification)

    def _build_prompt(
        self,
        ticket_text: str,
        documents: list[GroundingDocument],
        classification: ClassificationResult,
        risk: RiskDecision,
    ) -> str:
        risk_reasons = ", ".join(risk.reasons) if risk.reasons else "none"
        context = _format_context(documents)

        return f"""You are a support triage agent for HackerRank, Claude, and Visa.

Use only the retrieved support documents as grounding context. Do not use outside knowledge.
If the context does not support a claim, do not make that claim.
Write concise, professional support language.
Prefer natural phrases such as "Based on the available support guidance" or "The available documentation suggests".
Avoid repeatedly saying "retrieved support information" in the customer-facing response.
Do not copy article metadata, timestamps, page headers, navigation labels, or raw extracted text.
Summarize the relevant support information naturally in no more than 3 sentences.

Ticket:
{ticket_text}

Classification:
- company: {classification.company}
- product_area: {classification.product_area}
- request_type: {classification.request_type}

Risk decision:
- status: {risk.status}
- reasons: {risk_reasons}

Retrieved support documents:
{context}

Instructions:
- If status is "{ESCALATED}", produce a safe escalation response. Explain the escalation clearly without giving sensitive procedural advice.
- If status is "{REPLIED}", answer helpfully using only the retrieved documents and sound like a real support agent.
- Return only valid JSON with exactly these keys: "response", "justification".
- Keep "response" under 90 words and 3 sentences. Keep "justification" under 70 words.
"""

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        endpoint = GEMINI_ENDPOINT.format(model=parse.quote(self.model, safe=""))
        url = endpoint
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "topP": 1,
                "topK": 1,
                "candidateCount": 1,
                "maxOutputTokens": 300,
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "justification": {"type": "string"},
                    },
                    "required": ["response", "justification"],
                },
            },
        }
        body = json.dumps(payload).encode("utf-8")
        gemini_request = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key or "",
            },
            method="POST",
        )

        with request.urlopen(gemini_request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")

        response_payload = json.loads(raw)
        text = (
            response_payload.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not text:
            raise ValueError("Gemini response did not contain text")
        return _parse_json_text(text)


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response JSON was not an object")
    return parsed


if __name__ == "__main__":
    samples = (
        ("Visa", "My Visa payment failed at a merchant checkout."),
        (None, "I think my account was hacked and someone accessed it without permission."),
        ("HackerRank", "Can you add support for custom assessment branding?"),
    )

    corpus = load_corpus()
    retriever = Retriever(corpus)
    classifier = TicketClassifier()
    risk_assessor = RiskAssessor()
    responder = Responder()

    for company, ticket in samples:
        retrieved = retriever.retrieve(ticket, top_k=3)
        classification = classifier.classify(
            ticket,
            provided_company=company,
            retrieved_documents=[result.document for result in retrieved],
        )
        risk = risk_assessor.decide(ticket)
        result = responder.generate(ticket, retrieved, classification, risk)

        print(f"Ticket: {ticket}")
        print(f"  status={risk.status}")
        print(f"  response={result.response}")
        print(f"  justification={result.justification}")
