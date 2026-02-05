from __future__ import annotations
import os
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

from rag.chunking import chunk_text_token_aware
from rag.qdrant_store import get_qdrant_client, ensure_collection, upsert_chunks
from rag.ingest_core import ingest_pdf_bytes


def read_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return pages


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Put it in server/.env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--doc-id", required=False, help="Document ID (defaults to filename)")
    parser.add_argument("--collection", default="docs", help="Qdrant collection name")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--chunk-tokens", type=int, default=700)
    parser.add_argument("--overlap-tokens", type=int, default=120)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()

    pdf_path = args.pdf
    doc_id = args.doc_id or os.path.basename(pdf_path)

    client = OpenAI(api_key=api_key)
    qdrant = get_qdrant_client(args.qdrant_url)

    print(f"[1/5] Reading PDF: {pdf_path}")
    pages = read_pdf_pages(pdf_path)

    print(f"[2/5] Chunking pages (chunk_tokens={args.chunk_tokens}, overlap={args.overlap_tokens})")
    all_chunks = []
    for page_idx, page_text in enumerate(pages, start=1):
        chunks = chunk_text_token_aware(
            doc_id=doc_id,
            page=page_idx,
            text=page_text,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No text extracted from PDF (maybe scanned image PDF).")

    print(f"Total chunks: {len(all_chunks)}")

    print(f"[3/5] Creating embeddings using {args.embedding_model}")
    texts = [c.text for c in all_chunks]

    # Embeddings API call (batch)
    emb = client.embeddings.create(
        model=args.embedding_model,
        input=texts,
    )

    vectors = [d.embedding for d in emb.data]
    vector_size = len(vectors[0])

    print(f"[4/5] Ensuring Qdrant collection '{args.collection}' (dim={vector_size})")
    ensure_collection(qdrant, args.collection, vector_size)

    print("[5/5] Upserting vectors + payloads to Qdrant")
    stable_ids = [c.chunk_id for c in all_chunks]  
    payloads: List[Dict[str, Any]] = [
        {
            "stable_id": c.chunk_id, 
            "doc_id": doc_id,
            "page": c.page,
            "text": c.text,
            "start_token": c.start_token,
            "end_token": c.end_token,
        }
        for c in all_chunks
    ]

    upsert_chunks(qdrant, args.collection, stable_ids=stable_ids, vectors=vectors, payloads=payloads)

    print("✅ Done. Ingested into Qdrant.")
    print(f"Collection: {args.collection}")
    print(f"Doc ID: {doc_id}")
    print(f"Chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()
