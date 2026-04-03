from __future__ import annotations

# handling pdf reading, chunking, and retrieval for the book chat
# uses Moorcheh as the vector store and AI answer engine

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    # using pypdf first because it is usually the cleanest option
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    # keeping a fallback in case pypdf is not available
    from PyPDF2 import PdfReader as PyPDF2Reader
except Exception:
    PyPDF2Reader = None

try:
    from moorcheh_sdk import MoorchehClient
except Exception:
    MoorchehClient = None

# ---------------------------------------------------------------------------
# Moorcheh configuration
# ---------------------------------------------------------------------------

# The namespace name used to store the book in Moorcheh.
# Change this if you want to use a different namespace per deployment.
MOORCHEH_NAMESPACE = "egg-nest-book"

# How many chunks to retrieve before generating the answer.
MOORCHEH_TOP_K = 5

# keeping chunk sizes moderate so retrieval stays focused
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 180

# avoiding very tiny useless chunks
MIN_CHUNK_LENGTH = 120


@dataclass
class BookChunk:
    # storing one chunk of the pdf text plus its page reference
    chunk_id: int
    page_number: int
    text: str


@dataclass
class RetrievalResult:
    # storing one retrieved chunk plus similarity score
    chunk_id: int
    page_number: int
    text: str
    score: float


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

def get_moorcheh_api_key(api_key: Optional[str] = None) -> str:
    """Return the Moorcheh API key.

    Resolution order:
      1. Explicit argument passed by the caller.
      2. MOORCHEH_API_KEY environment variable.
      3. Streamlit secrets (st.secrets["MOORCHEH_API_KEY"]) — only attempted
         when running inside a Streamlit session.

    Raises ValueError if no key is found.
    """
    if api_key:
        return api_key

    env_key = os.environ.get("MOORCHEH_API_KEY", "").strip()
    if env_key:
        return env_key

    # Try Streamlit secrets without hard-importing streamlit at module level
    try:
        import streamlit as st
        secret = st.secrets.get("MOORCHEH_API_KEY", "").strip()
        if secret:
            return secret
    except Exception:
        pass

    raise ValueError(
        "No Moorcheh API key found. "
        "Set the MOORCHEH_API_KEY environment variable, add it to "
        ".streamlit/secrets.toml, or pass it explicitly."
    )


def _make_client(api_key: Optional[str] = None) -> "MoorchehClient": # type: ignore
    if MoorchehClient is None:
        raise ImportError(
            "moorcheh-sdk is not installed. Run: pip install moorcheh-sdk"
        )
    return MoorchehClient(api_key=get_moorcheh_api_key(api_key))


# ---------------------------------------------------------------------------
# PDF reading (unchanged from original)
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_whitespace(text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[=_\-]{3,}", " ", text)
    return normalize_whitespace(text)


def read_pdf_pages(pdf_path: str | Path) -> List[Tuple[int, str]]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"book pdf not found: {pdf_path}")

    reader = None

    if PdfReader is not None:
        reader = PdfReader(str(pdf_path))
    elif PyPDF2Reader is not None:
        reader = PyPDF2Reader(str(pdf_path))
    else:
        raise ImportError("No PDF reader available. Install pypdf or PyPDF2.")

    pages: List[Tuple[int, str]] = []

    for index, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        cleaned = clean_pdf_text(raw_text)
        if cleaned:
            pages.append((index + 1, cleaned))

    return pages


def split_text_into_chunks(
    text: str,
    page_number: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_chunk_id: int = 0,
) -> List[BookChunk]:
    if not text:
        return []

    text = normalize_whitespace(text)
    chunks: List[BookChunk] = []
    start = 0
    current_chunk_id = start_chunk_id
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        if end < text_length:
            last_period = chunk_text.rfind(". ")
            last_newline = chunk_text.rfind("\n")
            best_break = max(last_period, last_newline)

            if best_break > int(chunk_size * 0.6):
                end = start + best_break + 1
                chunk_text = text[start:end]

        chunk_text = normalize_whitespace(chunk_text)

        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunks.append(
                BookChunk(
                    chunk_id=current_chunk_id,
                    page_number=page_number,
                    text=chunk_text,
                )
            )
            current_chunk_id += 1

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_book_chunks(
    pdf_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[BookChunk]:
    pages = read_pdf_pages(pdf_path)
    all_chunks: List[BookChunk] = []
    next_chunk_id = 0

    for page_number, page_text in pages:
        page_chunks = split_text_into_chunks(
            text=page_text,
            page_number=page_number,
            chunk_size=chunk_size,
            overlap=overlap,
            start_chunk_id=next_chunk_id,
        )
        all_chunks.extend(page_chunks)
        next_chunk_id += len(page_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Moorcheh: namespace + document upload
# ---------------------------------------------------------------------------

def _namespace_exists(client: "MoorchehClient", namespace: str) -> bool: # type: ignore
    """Return True if the namespace already exists in Moorcheh."""
    try:
        existing = client.namespaces.list()
        # The SDK returns a list of namespace objects or dicts
        names = [
            (ns["name"] if isinstance(ns, dict) else ns.name)
            for ns in existing
        ]
        return namespace in names
    except Exception:
        return False


def upload_book_to_moorcheh(
    pdf_path: str | Path,
    api_key: Optional[str] = None,
    namespace: str = MOORCHEH_NAMESPACE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Parse the PDF, chunk it, and upload all chunks to a Moorcheh namespace.

    Creates the namespace if it does not already exist. Returns the number
    of chunks uploaded.
    """
    chunks = build_book_chunks(pdf_path, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        raise ValueError("No text chunks could be built from the PDF.")

    with _make_client(api_key) as client:
        try:
            client.namespaces.create(namespace_name=namespace, type="text")
        except Exception as e:
            if "already exists" not in str(e).lower() and "conflict" not in str(e).lower():
                raise

        documents = [
            {
                "id": f"chunk_{chunk.chunk_id}",
                "text": chunk.text,
                "metadata": {"page_number": chunk.page_number},
            }
            for chunk in chunks
        ]

        # Upload in batches of 100 to stay within API limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            client.documents.upload(namespace_name=namespace, documents=batch)

    return len(chunks)


# ---------------------------------------------------------------------------
# build_book_index — public API expected by app.py
# ---------------------------------------------------------------------------

def build_book_index(
    pdf_path: str | Path,
    api_key: Optional[str] = None,
    namespace: str = MOORCHEH_NAMESPACE,
) -> Dict[str, object]:
    """Build (or verify) the Moorcheh namespace for the book.

    Returns a lightweight index dict that answer_book_question can use.
    The heavy lifting (embedding + storage) happens inside Moorcheh.
    Only uploads the PDF chunks if the namespace does not already exist.
    """
    with _make_client(api_key) as client:
        try:
            already_exists = _namespace_exists(client, namespace)
        except Exception:
            already_exists = False

    if not already_exists:
        try:
            upload_book_to_moorcheh(pdf_path, api_key=api_key, namespace=namespace)
        except Exception as e:
            if "already exists" not in str(e).lower() and "conflict" not in str(e).lower():
                raise

    return {
        "pdf_path": str(pdf_path),
        "namespace": namespace,
        "api_key": api_key,  # may be None — get_moorcheh_api_key resolves it later
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    query: str,
    book_index: Dict[str, object],
    top_k: int = MOORCHEH_TOP_K,
) -> List[RetrievalResult]:
    """Run a semantic search against the Moorcheh namespace."""
    namespace: str = book_index["namespace"]
    api_key: Optional[str] = book_index.get("api_key")

    with _make_client(api_key) as client:
        response = client.similarity_search.query(
            namespaces=[namespace],
            query=query,
            top_k=top_k,
        )

    results: List[RetrievalResult] = []
    hits = response.get("results", []) if isinstance(response, dict) else []

    for idx, hit in enumerate(hits):
        text = hit.get("text", "") if isinstance(hit, dict) else str(hit)
        score = float(hit.get("score", 0.0)) if isinstance(hit, dict) else 0.0
        metadata = hit.get("metadata", {}) if isinstance(hit, dict) else {}
        page_number = int(metadata.get("page_number", 0))

        results.append(
            RetrievalResult(
                chunk_id=idx,
                page_number=page_number,
                text=text,
                score=score,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def answer_book_question(
    query: str,
    book_index: Dict[str, object],
    top_k: int = MOORCHEH_TOP_K,
) -> Dict[str, object]:
    """Generate a grounded AI answer using Moorcheh's /answer endpoint.

    Falls back to a snippet-based answer if the generative call fails.
    """
    namespace: str = book_index["namespace"]
    api_key: Optional[str] = book_index.get("api_key")

    # --- try Moorcheh generative answer first ---
    try:
        with _make_client(api_key) as client:
            gen_response = client.answer.generate(
                namespace=namespace,
                query=query,
                top_k=top_k,
            )

        answer_text = (
            gen_response.get("answer", "")
            if isinstance(gen_response, dict)
            else str(gen_response)
        )

        # Also fetch retrieval results for the sources panel in the UI
        results = retrieve_relevant_chunks(query, book_index, top_k=top_k)

        return {
            "query": query,
            "answer": answer_text or _fallback_answer(query, results),
            "results": results,
        }

    except Exception as exc:
        # Degrade gracefully — retrieve chunks and build a snippet answer
        try:
            results = retrieve_relevant_chunks(query, book_index, top_k=top_k)
        except Exception:
            results = []
        return {
            "query": query,
            "answer": _fallback_answer(query, results, error=str(exc)),
            "results": results,
        }


# ---------------------------------------------------------------------------
# Fallback (no LLM)
# ---------------------------------------------------------------------------

def _fallback_answer(
    query: str,
    results: List[RetrievalResult],
    error: Optional[str] = None,
) -> str:
    """Assemble a plain-text snippet answer when the generative call is unavailable."""
    if not results:
        return (
            "I could not find a clear answer in the book for that question. "
            "Try asking with a species name, nest type, egg color, habitat, "
            "or another more specific detail."
        )

    note = f"\n\n_(Generative answer unavailable: {error})_" if error else ""

    intro = f'Here is what I found in the book for your question: "{query}".\n\n'
    parts: List[str] = []

    for index, result in enumerate(results[:3], start=1):
        snippet = result.text.strip()
        if len(snippet) > 700:
            snippet = snippet[:700].rsplit(" ", 1)[0] + "..."
        parts.append(f"{index}. Page {result.page_number}\n{snippet}")

    outro = (
        "\n\nThese passages are the closest matches from the book. "
        "Use the page numbers to verify the original context."
    )

    return intro + "\n\n".join(parts) + outro + note


# ---------------------------------------------------------------------------
# Kept for backward compatibility — no longer used internally
# ---------------------------------------------------------------------------

def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    seen = set()
    unique: List[RetrievalResult] = []
    for result in results:
        key = (result.page_number, result.text.strip())
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique