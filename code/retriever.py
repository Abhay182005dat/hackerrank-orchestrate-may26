"""
retriever.py — Corpus loader and BM25 retriever.

Design decisions:
- BM25Okapi chosen over TF-IDF for better term-frequency saturation handling.
- ~400-token chunks with 50-token overlap balance context size vs. granularity.
- Company-biased retrieval: when company is known, we first search that
  company's subset; if scores are low, we fall back to the full corpus.
- All corpus files are .md — we strip markdown formatting for cleaner indexing.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    """A single chunk of text from a corpus document."""
    text: str
    company: str          # hackerrank, claude, visa
    source_file: str      # relative path from data/
    chunk_index: int
    score: float = 0.0    # filled at retrieval time


# --------------------------------------------------------------------------- #
# Module-level state (loaded once at import / init)
# --------------------------------------------------------------------------- #
_chunks: list[Chunk] = []
_tokenized_corpus: list[list[str]] = []
_bm25_full: Optional[BM25Okapi] = None
_bm25_by_company: dict[str, tuple[BM25Okapi, list[int]]] = {}
_initialized: bool = False


# --------------------------------------------------------------------------- #
# Text processing helpers
# --------------------------------------------------------------------------- #
def _strip_markdown(text: str) -> str:
    """Remove common markdown syntax for cleaner indexing."""
    # Remove links but keep text: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    # Remove headers markup but keep text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _simple_tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for BM25."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of approximately `chunk_size` tokens with `overlap`.
    We use word-level splitting as a proxy for tokens (~1 word ≈ 1.3 tokens).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward by (chunk_size - overlap)
    return chunks


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def _detect_company(filepath: str) -> str:
    """Infer company from file path under data/."""
    parts = Path(filepath).parts
    for part in parts:
        lower = part.lower()
        if lower == 'hackerrank':
            return 'hackerrank'
        elif lower == 'claude':
            return 'claude'
        elif lower == 'visa':
            return 'visa'
    return 'unknown'


def init_retriever(data_dir: str) -> None:
    """
    Walk data_dir, read all .md files, chunk them, and build BM25 indices.
    Call this once at startup.
    """
    global _chunks, _tokenized_corpus, _bm25_full, _bm25_by_company, _initialized

    if _initialized:
        return

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Walk all files
    for root, _dirs, files in os.walk(data_path):
        for fname in files:
            # Support .md files (that's all we have, but coded defensively)
            if not fname.endswith(('.md', '.txt', '.html', '.json')):
                continue

            fpath = Path(root) / fname
            try:
                raw = fpath.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue

            # Determine company from path
            rel_path = str(fpath.relative_to(data_path))
            company = _detect_company(rel_path)

            # Clean and chunk
            cleaned = _strip_markdown(raw)
            if len(cleaned.strip()) < 20:
                continue  # skip near-empty files

            text_chunks = _chunk_text(cleaned)
            for idx, chunk_text in enumerate(text_chunks):
                _chunks.append(Chunk(
                    text=chunk_text,
                    company=company,
                    source_file=rel_path,
                    chunk_index=idx,
                ))

    if not _chunks:
        raise RuntimeError("No corpus chunks loaded — check data/ directory")

    # Build full BM25 index
    _tokenized_corpus = [_simple_tokenize(c.text) for c in _chunks]
    _bm25_full = BM25Okapi(_tokenized_corpus)

    # Build per-company indices for biased retrieval
    company_indices: dict[str, list[int]] = {}
    for i, chunk in enumerate(_chunks):
        company_indices.setdefault(chunk.company, []).append(i)

    for company, indices in company_indices.items():
        tokenized_subset = [_tokenized_corpus[i] for i in indices]
        if tokenized_subset:
            _bm25_by_company[company] = (BM25Okapi(tokenized_subset), indices)

    _initialized = True
    print(f"[retriever] Loaded {len(_chunks)} chunks from {data_path} "
          f"({len(company_indices)} companies)")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def retrieve(query: str, company: Optional[str] = None,
             top_k: int = 5) -> list[Chunk]:
    """
    Retrieve the most relevant chunks for a query.

    When company is known, we first search that company's sub-index.
    If the top score is below a threshold, we also search the full corpus
    and merge results.
    """
    if not _initialized:
        raise RuntimeError("Retriever not initialized — call init_retriever() first")

    tokenized_query = _simple_tokenize(query)
    if not tokenized_query:
        return []

    results: list[Chunk] = []

    # Normalize company name
    company_key = None
    if company and company.lower() not in ('none', ''):
        company_key = company.lower()

    # Strategy: search company sub-index first, then full if needed
    if company_key and company_key in _bm25_by_company:
        bm25_sub, indices = _bm25_by_company[company_key]
        scores = bm25_sub.get_scores(tokenized_query)

        scored = [(scores[j], indices[j]) for j in range(len(indices))]
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, idx in scored[:top_k]:
            chunk = Chunk(
                text=_chunks[idx].text,
                company=_chunks[idx].company,
                source_file=_chunks[idx].source_file,
                chunk_index=_chunks[idx].chunk_index,
                score=float(score),
            )
            results.append(chunk)

        # If top score is low, also search full corpus
        top_score = results[0].score if results else 0.0
        if top_score < 2.0:
            full_results = _search_full(tokenized_query, top_k)
            # Merge, dedup by (source_file, chunk_index)
            seen = {(c.source_file, c.chunk_index) for c in results}
            for c in full_results:
                if (c.source_file, c.chunk_index) not in seen:
                    results.append(c)
                    seen.add((c.source_file, c.chunk_index))
            # Re-sort and trim
            results.sort(key=lambda c: c.score, reverse=True)
            results = results[:top_k]
    else:
        # No company specified — search full corpus
        results = _search_full(tokenized_query, top_k)

    return results


def _search_full(tokenized_query: list[str], top_k: int) -> list[Chunk]:
    """Search the full BM25 index."""
    scores = _bm25_full.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        chunk = Chunk(
            text=_chunks[idx].text,
            company=_chunks[idx].company,
            source_file=_chunks[idx].source_file,
            chunk_index=_chunks[idx].chunk_index,
            score=float(scores[idx]),
        )
        results.append(chunk)
    return results


def get_min_score_threshold() -> float:
    """
    Minimum BM25 score to consider a chunk relevant.
    Below this, we consider that no relevant docs were found.
    """
    return 1.0
