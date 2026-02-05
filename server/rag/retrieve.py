from __future__ import annotations
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def _make_filter(doc_id: Optional[str]) -> Optional[qm.Filter]:
    if not doc_id:
        return None
    return qm.Filter(
        must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
    )


def search_chunks(
    client: QdrantClient,
    collection_name: str,
    *,
    query_vector: List[float],
    top_k: int = 6,
    doc_id: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Version-tolerant search wrapper for QdrantClient.

    Returns: [{ "score": float, "payload": {...} }, ...]
    """
    flt = _make_filter(doc_id)

    # Preferred API (newer qdrant-client)
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=flt,
        )
        points = res.points or []
        return [
            {"score": float(p.score or 0.0), "payload": dict(p.payload or {})}
            for p in points
        ]

    # Fallback API (older qdrant-client)
    if hasattr(client, "search"):
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=flt,
        )
        return [{"score": float(h.score), "payload": dict(h.payload or {})} for h in hits]

    raise RuntimeError(
        "Unsupported qdrant-client version: neither query_points nor search exists. "
        "Run: python -m pip install -U qdrant-client"
    )
