from __future__ import annotations
import io
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import hashlib

from pypdf import PdfReader
import tiktoken

from openai import OpenAI
from .qdrant_store import ensure_collection, upsert_chunks, get_qdrant_client


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    start_token: int
    end_token: int


def _pdf_to_pages_text(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))

    if getattr(reader, "is_encrypted", False):
        try:
            # many PDFs allow empty password
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF is encrypted and couldn't be decrypted (needs password).")

    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages



def _chunk_text(pages: List[str], *, doc_id: str, chunk_tokens: int = 700, overlap: int = 120) -> List[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")

    chunks: List[Chunk] = []
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text:
            continue

        tokens = enc.encode(page_text)
        i = 0
        c_idx = 0

        while i < len(tokens):
            window = tokens[i : i + chunk_tokens]
            text = enc.decode(window).strip()

            chunk_id = f"{doc_id}:p{page_idx}:c{c_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page_idx,
                    text=text,
                    start_token=i,
                    end_token=min(i + chunk_tokens, len(tokens)),
                )
            )

            c_idx += 1
            i += max(1, chunk_tokens - overlap)

    return chunks


def ingest_pdf_bytes(
    *,
    pdf_bytes: bytes,
    doc_id: str,
    collection: str = "docs",
    chunk_tokens: int = 700,
    overlap: int = 120,
    qdrant_url: str = "http://localhost:6333",
    openai_client: OpenAI,
    embedding_model: str = "text-embedding-3-small",
) -> Dict[str, Any]:
    pages = _pdf_to_pages_text(pdf_bytes)
    chunks = _chunk_text(pages, doc_id=doc_id, chunk_tokens=chunk_tokens, overlap=overlap)

    if not chunks:
        return {"doc_id": doc_id, "chunks": 0, "status": "no_text_extracted"}

    # embeddings
    texts = [c.text for c in chunks]
    emb = openai_client.embeddings.create(model=embedding_model, input=texts)
    vectors = [d.embedding for d in emb.data]

    # payloads
    stable_ids = [c.chunk_id for c in chunks]
    payloads = [
        {
            "stable_id": c.chunk_id,
            "doc_id": c.doc_id,
            "page": c.page,
            "text": c.text,
            "start_token": c.start_token,
            "end_token": c.end_token,
        }
        for c in chunks
    ]

    # qdrant upsert
    qdrant = get_qdrant_client(qdrant_url)
    ensure_collection(qdrant, collection, vector_size=len(vectors[0]))
    upsert_chunks(qdrant, collection, stable_ids=stable_ids, vectors=vectors, payloads=payloads)

    return {"doc_id": doc_id, "chunks": len(chunks), "status": "ok"}
