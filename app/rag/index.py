from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.rag.config import Settings as AppSettings


def _configure_models(app_settings: AppSettings) -> None:
    Settings.llm = OpenAI(
        model=app_settings.openai_model,
        api_key=app_settings.openai_api_key,
        temperature=0,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=app_settings.embedding_model,
        api_key=app_settings.openai_api_key,
    )


def _ensure_collection(client: QdrantClient, collection_name: str) -> None:
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    if exists:
        return

    # text-embedding-3-large is 3072 dimensions.
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )


def get_index(app_settings: AppSettings) -> VectorStoreIndex:
    _configure_models(app_settings)
    client = QdrantClient(url=app_settings.qdrant_url)
    _ensure_collection(client, app_settings.qdrant_collection)
    vector_store = QdrantVectorStore(client=client, collection_name=app_settings.qdrant_collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def upsert_documents(index: VectorStoreIndex, documents: list[Document]) -> None:
    for doc in documents:
        file_id = (doc.metadata or {}).get("file_id")
        if file_id:
            delete_document_by_file_id(index=index, file_id=file_id)
    if documents:
        index.insert_batch(documents)


def delete_document_by_file_id(index: VectorStoreIndex, file_id: str) -> None:
    vs = index.vector_store
    if hasattr(vs, "client") and hasattr(vs, "collection_name"):
        vs.client.delete(
            collection_name=vs.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))],
            ),
        )


def build_acl_qdrant_filter(user_email: str, user_groups: list[str]) -> Filter:
    should_filters = [FieldCondition(key="is_public", match=MatchValue(value=True))]

    normalized_email = user_email.strip().lower()
    if normalized_email:
        should_filters.append(FieldCondition(key="allowed_users[]", match=MatchValue(value=normalized_email)))

    for group in {g.strip().lower() for g in user_groups if g.strip()}:
        should_filters.append(FieldCondition(key="allowed_groups[]", match=MatchValue(value=group)))

    return Filter(should=should_filters)
