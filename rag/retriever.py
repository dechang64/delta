"""
rag/retriever.py — Knowledge Base Retriever with Time Cutoff.

CRITICAL: Only retrieves documents dated BEFORE the scoring date.
This prevents look-ahead bias — a key improvement over v1.

Supports:
  - ChromaDB vector store (production)
  - Simple file-based retrieval (development)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional


class KnowledgeRetriever:
    """Retrieve domain-specific knowledge with strict time cutoff."""

    def __init__(
        self,
        kb_dir: str,
        cutoff_date: str = "2099-12-31",
        use_chroma: bool = False,
    ):
        self.kb_dir = Path(kb_dir)
        self.cutoff_date = cutoff_date
        self.use_chroma = use_chroma
        self._collection = None

        if not self.kb_dir.exists():
            print(f"  [WARN] KB directory not found: {kb_dir}")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search knowledge base, returning only documents before cutoff_date.

        Args:
            query: Search query (ticker + date + domain)
            top_k: Number of results to return

        Returns:
            List of document strings, each prefixed with its date
        """
        if self.use_chroma:
            return self._search_chroma(query, top_k)
        else:
            return self._search_files(query, top_k)

    def _search_files(self, query: str, top_k: int) -> List[str]:
        """Simple file-based retrieval for development.

        Reads JSON documents from kb_dir, filters by cutoff_date,
        returns top_k most recent (before cutoff) documents.
        """
        if not self.kb_dir.exists():
            return []

        docs = []
        for json_file in sorted(self.kb_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle both single doc and list of docs
                if isinstance(data, list):
                    for doc in data:
                        self._add_if_valid(doc, docs)
                else:
                    self._add_if_valid(data, docs)

            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by date (most recent first, but before cutoff)
        docs.sort(key=lambda x: x["date"], reverse=True)

        # Format output
        results = []
        for doc in docs[:top_k]:
            results.append(f"[{doc['date']}] {doc.get('title', '')}\n{doc.get('content', '')}")

        return results

    def _add_if_valid(self, doc: dict, docs: list):
        """Add document if it passes time cutoff filter."""
        doc_date = doc.get("date", "")
        if doc_date and doc_date < self.cutoff_date:
            docs.append(doc)

    def _search_chroma(self, query: str, top_k: int) -> List[str]:
        """ChromaDB vector search for production.

        Requires: pip install chromadb sentence-transformers
        """
        try:
            import chromadb
        except ImportError:
            print("  [WARN] chromadb not installed, falling back to file search")
            return self._search_files(query, top_k)

        if self._collection is None:
            client = chromadb.PersistentClient(path=str(self.kb_dir / "chroma"))
            self._collection = client.get_or_create_collection(
                name=self.kb_dir.name,
                metadata={"hnsw:space": "cosine"},
            )

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k * 3,  # Over-retrieve, then filter by date
            where={"date": {"$lt": self.cutoff_date}},
        )

        output = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            date = metadata.get("date", "unknown")
            title = metadata.get("title", "")
            output.append(f"[{date}] {title}\n{doc}")

        return output[:top_k]
