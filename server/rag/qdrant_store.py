from __future__ import annotations
from typing import List, Dict, Any, Tuple
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def get_qdrant_client(url: str = "http://localhost:6333") -> QdrantClient:
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def make_uuid_id(stable_key: str) -> str:
    """
    Deterministic UUID from a string key (stable across re-ingests).
    This lets you re-run ingestion and overwrite the same points.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    *,
    stable_ids: List[str],            # your readable ids like doc:p1:c0
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
) -> None:
    points = []
    for i in range(len(stable_ids)):
        qdrant_id = make_uuid_id(stable_ids[i])
        points.append(
            qm.PointStruct(
                id=qdrant_id,
                vector=vectors[i],
                payload=payloads[i],
            )
        )

    client.upsert(collection_name=collection_name, points=points)

def delete_doc(client: QdrantClient, collection_name: str, doc_id: str) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            )
        ),
    )
