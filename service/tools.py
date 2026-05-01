from service.db import run_query
from service.semantic_search import semantic_product_search, upsert_products_to_qdrant


def get_data_from_db(query: str):
    return run_query(query)


def semantic_search_products(query: str, limit: int = 5):
    return semantic_product_search(query=query, limit=limit)


def reindex_products_semantic():
    return upsert_products_to_qdrant()
