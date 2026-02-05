from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    collection: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[RetrievedChunk]
