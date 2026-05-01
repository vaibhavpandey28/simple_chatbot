import os
from typing import Any

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.logger import get_logger
from service.db import get_connection

load_dotenv()
logger = get_logger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "products")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "5"))

_client: QdrantClient | None = None
_embedder: TextEmbedding | None = None
_vector_size: int | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        _embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embedder


def _get_vector_size() -> int:
    global _vector_size
    if _vector_size is None:
        sample = list(_get_embedder().embed(["sample"]))
        _vector_size = len(sample[0])
    return _vector_size


def _embed_one(text: str) -> list[float]:
    return list(_get_embedder().embed([text]))[0].tolist()


def _embed_many(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_embedder().embed(texts)]


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION in existing:
        return

    vector_size = _get_vector_size()
    logger.info("Creating Qdrant collection: %s", QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def _fetch_products(limit: int = 2000) -> list[dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT
              product_id,
              product_name,
              description,
              category,
              price,
              stock_qty
            FROM products
            ORDER BY product_id
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "product_id": row[0],
                "product_name": row[1],
                "description": row[2],
                "category": row[3],
                "price": float(row[4]),
                "stock_qty": row[5],
            }
            for row in rows
        ]
    finally:
        cur.close()
        conn.close()


def upsert_products_to_qdrant(limit: int = 2000, recreate: bool = False) -> dict[str, Any]:
    client = _get_client()
    if recreate:
        existing = {c.name for c in client.get_collections().collections}
        if QDRANT_COLLECTION in existing:
            logger.info("Recreating Qdrant collection: %s", QDRANT_COLLECTION)
            client.delete_collection(collection_name=QDRANT_COLLECTION)
    _ensure_collection(client)

    products = _fetch_products(limit=limit)
    if not products:
        return {"indexed": 0, "message": "No products found in database."}

    docs = [
        f"{p['product_name']} {p['description']} category {p['category']} stock {p['stock_qty']} price {p['price']}"
        for p in products
    ]
    vectors = _embed_many(docs)

    points = []
    for p, vec in zip(products, vectors, strict=False):
        points.append(
            models.PointStruct(
                id=int(p["product_id"]),
                vector=vec,
                payload=p,
            )
        )

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info("Indexed %s products into Qdrant", len(points))
    return {"indexed": len(points), "collection": QDRANT_COLLECTION, "recreate": recreate}


def _collection_size(client: QdrantClient) -> int:
    info = client.get_collection(QDRANT_COLLECTION)
    return info.points_count or 0


def semantic_product_search(query: str, limit: int = SEMANTIC_TOP_K) -> list[dict[str, Any]]:
    client = _get_client()
    _ensure_collection(client)

    if _collection_size(client) == 0:
        upsert_products_to_qdrant()

    query_vector = _embed_one(query)
    request_limit = max(1, min(limit, 20))

    if not hasattr(client, "query_points"):
        raise RuntimeError(
            "Installed qdrant-client does not support query_points(). "
            "Please upgrade qdrant-client to >=1.16."
        )

    query_response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=request_limit,
        with_payload=True,
    )
    hits = query_response.points

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "score": float(hit.score),
                "product_id": payload.get("product_id"),
                "product_name": payload.get("product_name"),
                "description": payload.get("description"),
                "category": payload.get("category"),
                "price": payload.get("price"),
                "stock_qty": payload.get("stock_qty"),
            }
        )
    return results
