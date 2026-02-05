from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import tiktoken


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    start_token: int
    end_token: int


def chunk_text_token_aware(
    *,
    doc_id: str,
    page: int,
    text: str,
    chunk_tokens: int = 700,
    overlap_tokens: int = 120,
    encoding_name: str = "cl100k_base",
) -> List[Chunk]:
    """
    Token-aware chunking using tiktoken.
    - chunk_tokens: target tokens per chunk
    - overlap_tokens: repeated tokens between chunks for continuity
    """
    text = (text or "").strip()
    if not text:
        return []

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < chunk_tokens")

    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_text = enc.decode(tokens[start:end]).strip()

        if chunk_text:
            chunk_id = f"{doc_id}:p{page}:c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page=page,
                    start_token=start,
                    end_token=end,
                )
            )
            idx += 1

        if end == len(tokens):
            break

        # slide window with overlap
        start = max(0, end - overlap_tokens)

    return chunks
