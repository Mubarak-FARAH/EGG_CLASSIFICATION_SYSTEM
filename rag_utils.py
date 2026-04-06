from __future__ import annotations

#handling pdf reading, chunking, embeddings, and retrieval for the book chat

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    #using pypdf first because it is usually the cleanest option
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    #keeping a fallback in case pypdf is not available
    from PyPDF2 import PdfReader as PyPDF2Reader
except Exception:
    PyPDF2Reader = None

try:
    #using sentence-transformers like the sample idea
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


<<<<<<< HEAD
#keeping model name in one place in case you want to change it later
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
=======
# The namespace name used to store the book in Moorcheh.
# Change this if you want to use a different namespace per deployment.
MOORCHEH_NAMESPACE = "egg-nest-book"
>>>>>>> 44bc637aa98acb88a6fbed95e98040373096def2

#keeping chunk sizes moderate so retrieval stays focused
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 180

#avoiding very tiny useless chunks
MIN_CHUNK_LENGTH = 120


@dataclass
class BookChunk:
    #storing one chunk of the pdf text plus its page reference
    chunk_id: int
    page_number: int
    text: str


@dataclass
class RetrievalResult:
    #storing one retrieved chunk plus similarity score
    chunk_id: int
    page_number: int
    text: str
    score: float


<<<<<<< HEAD
=======
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

>>>>>>> 44bc637aa98acb88a6fbed95e98040373096def2
def normalize_whitespace(text: str) -> str:
    #cleaning repeated spaces and line breaks
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def clean_pdf_text(text: str) -> str:
    #doing light cleanup without being too aggressive
    if not text:
        return ""

    text = normalize_whitespace(text)

    #removing isolated page artifacts when possible
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    #removing weird repeated separators
    text = re.sub(r"[=_\-]{3,}", " ", text)

    return normalize_whitespace(text)


def read_pdf_pages(pdf_path: str | Path) -> List[Tuple[int, str]]:
    #reading text page by page so we preserve page references
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"book pdf not found: {pdf_path}")

    reader = None

    if PdfReader is not None:
        reader = PdfReader(str(pdf_path))
    elif PyPDF2Reader is not None:
        reader = PyPDF2Reader(str(pdf_path))
    else:
        raise ImportError(
            "No PDF reader available. Install pypdf or PyPDF2."
        )

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
    #splitting page text into overlapping chunks for better retrieval
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

        #trying not to cut in the middle of a sentence when possible
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
    #reading the whole pdf and building chunk objects
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


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    #loading the embedding model only when needed
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. Install it before using the book chat."
        )

<<<<<<< HEAD
    return SentenceTransformer(model_name)
=======
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
>>>>>>> 44bc637aa98acb88a6fbed95e98040373096def2


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    #encoding all texts to embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)

    return embeddings.astype(np.float32)


def cosine_search(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    top_k: int = 5,
) -> List[int]:
    #retrieving the most similar chunk indices
    scores = np.dot(document_embeddings, query_embedding)

    if len(scores) == 0:
        return []

    top_k = min(top_k, len(scores))
    best_indices = np.argsort(scores)[::-1][:top_k]
    return best_indices.tolist()


def build_book_index(
    pdf_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, object]:
    #building the complete in-memory index for the book
    chunks = build_book_chunks(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("No text chunks could be built from the pdf.")

    model = get_embedding_model(model_name)
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(chunk_texts, model)

    return {
        "pdf_path": str(pdf_path),
        "model_name": model_name,
        "chunks": chunks,
        "embeddings": embeddings,
        "model": model,
    }


def retrieve_relevant_chunks(
    query: str,
    book_index: Dict[str, object],
    top_k: int = 5,
) -> List[RetrievalResult]:
    #retrieving the best chunks for the user question
    query = normalize_whitespace(query)

    if not query:
        return []

    model: SentenceTransformer = book_index["model"]
    chunks: List[BookChunk] = book_index["chunks"]
    embeddings: np.ndarray = book_index["embeddings"]

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    best_indices = cosine_search(
        query_embedding=query_embedding,
        document_embeddings=embeddings,
        top_k=top_k,
    )

    scores = np.dot(embeddings, query_embedding)

    results: List[RetrievalResult] = []
    for idx in best_indices:
        chunk = chunks[idx]
        results.append(
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                text=chunk.text,
                score=float(scores[idx]),
            )
        )

    return results


def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    #removing duplicate chunks with identical text
    seen = set()
    unique_results: List[RetrievalResult] = []

    for result in results:
        key = (result.page_number, result.text.strip())
        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    return unique_results


def build_grounded_answer(query: str, results: List[RetrievalResult]) -> str:
    #building a grounded response without inventing unsupported details
    if not results:
        return (
            "I could not find a clear answer in the book for that question. "
            "Try asking with a species name, nest type, egg color, habitat, or another more specific detail."
        )

    intro = (
        f"Here is what I found in the book for your question: “{query}”.\n\n"
    )

    body_parts: List[str] = []

    for index, result in enumerate(results[:3], start=1):
        snippet = result.text.strip()
        if len(snippet) > 700:
            snippet = snippet[:700].rsplit(" ", 1)[0] + "..."

        body_parts.append(
            f"{index}. Page {result.page_number}\n{snippet}"
        )

    outro = (
        "\n\nThese passages are the closest matches from the book. "
        "Use the page numbers to verify the original context."
    )

    return intro + "\n\n".join(body_parts) + outro


def answer_book_question(
    query: str,
    book_index: Dict[str, object],
    top_k: int = 5,
) -> Dict[str, object]:
    #running the full retrieval flow for one user question
    raw_results = retrieve_relevant_chunks(
        query=query,
        book_index=book_index,
        top_k=top_k,
    )
    results = deduplicate_results(raw_results)
    answer = build_grounded_answer(query, results)

    return {
        "query": query,
        "answer": answer,
        "results": results,
    }