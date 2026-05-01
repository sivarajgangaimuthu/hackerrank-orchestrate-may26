"""Corpus loading utilities for the support triage agent.

This module reads only the local support corpus shipped with the repository.
It is intentionally deterministic: companies and files are processed in a
stable order, and unreadable files are skipped without stopping the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterable


SUPPORTED_EXTENSIONS = {".md", ".txt", ".html"}
COMPANY_DIRS = {
    "HackerRank": "hackerrank",
    "Claude": "claude",
    "Visa": "visa",
}


@dataclass(frozen=True)
class CorpusDocument:
    """A cleaned support document and its source metadata."""

    company: str
    filepath: str
    content: str


class _HTMLTextExtractor(HTMLParser):
    """Minimal deterministic HTML-to-text extractor."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_support_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*"), key=lambda item: str(item).lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _strip_html(raw_text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw_text)
    parser.close()
    return parser.get_text()


def _clean_markdown_text(raw_text: str) -> str:
    text = re.sub(r"\A---\s.*?^---\s*", " ", raw_text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~>#|]", " ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _extract_clean_text(path: Path, raw_text: str) -> str:
    if path.suffix.lower() == ".html":
        text = _strip_html(raw_text)
    else:
        text = _clean_markdown_text(raw_text)
    return _normalize_whitespace(text)


def load_corpus(data_dir: str | Path | None = None) -> list[CorpusDocument]:
    """Load support documents from the local data corpus.

    Args:
        data_dir: Optional path to the repository's data directory. When not
            provided, the function resolves ``../data`` relative to this file.

    Returns:
        A stable list of cleaned support documents with company, filepath, and
        content metadata. Empty or unreadable files are skipped.
    """

    root = Path(data_dir) if data_dir is not None else _repo_root() / "data"
    documents: list[CorpusDocument] = []

    for company, dirname in COMPANY_DIRS.items():
        company_dir = root / dirname
        if not company_dir.exists():
            continue

        for path in _iter_support_files(company_dir):
            try:
                raw_text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            content = _extract_clean_text(path, raw_text)
            if not content:
                continue

            documents.append(
                CorpusDocument(
                    company=company,
                    filepath=str(path),
                    content=content,
                )
            )

    return documents


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"Total documents loaded: {len(corpus)}")
    if corpus:
        sample = corpus[0]
        preview = sample.content[:300]
        print("Sample document preview:")
        print(f"Company: {sample.company}")
        print(f"Filepath: {sample.filepath}")
        print(f"Content: {preview}")
