"""TF-IDF retrieval utilities for the support triage agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from corpus import CorpusDocument, load_corpus


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document and its cosine similarity score."""

    document: CorpusDocument
    score: float


class Retriever:
    """Reusable TF-IDF retriever over a loaded support corpus.

    The document index is built once during initialization and reused for every
    query. Results are deterministic because the corpus order is preserved and
    score ties are resolved by original document position.
    """

    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )

        if self.documents:
            self.document_matrix = self.vectorizer.fit_transform(
                document.content for document in self.documents
            )
        else:
            self.document_matrix = None

    def retrieve(self, ticket_text: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return the most relevant documents for a support ticket.

        Args:
            ticket_text: User ticket subject/body text to search for.
            top_k: Maximum number of documents to return.

        Returns:
            A list of retrieval results ordered by descending cosine similarity.
        """

        if top_k <= 0 or not self.documents or self.document_matrix is None:
            return []

        query = ticket_text.strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_matrix).ravel()

        ranked_indexes = sorted(
            range(len(self.documents)),
            key=lambda index: (-similarities[index], index),
        )

        return [
            RetrievalResult(
                document=self.documents[index],
                score=float(similarities[index]),
            )
            for index in ranked_indexes[: min(top_k, len(ranked_indexes))]
        ]


if __name__ == "__main__":
    corpus = load_corpus()
    retriever = Retriever(corpus)
    results = retriever.retrieve("My Visa payment failed", top_k=3)

    for rank, result in enumerate(results, start=1):
        document = result.document
        preview = document.content[:240].replace("\n", " ")
        print(f"{rank}. score={result.score:.4f}")
        print(f"   company={document.company}")
        print(f"   filepath={document.filepath}")
        print(f"   preview={preview}")
