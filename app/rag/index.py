from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, VectorParams

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.rag.config import Settings as AppSettings


def build_index(app_settings: AppSettings) -> VectorStoreIndex:
    Settings.llm = OpenAI(model=app_settings.openai_model, api_key=app_settings.openai_api_key, temperature=0)
    Settings.embed_model = OpenAIEmbedding(
        model=app_settings.embedding_model,
        api_key=app_settings.openai_api_key,
    )

    client = QdrantClient(url=app_settings.qdrant_url)
    client.recreate_collection(
        collection_name=app_settings.qdrant_collection,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(client=client, collection_name=app_settings.qdrant_collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def load_existing_index(app_settings: AppSettings) -> VectorStoreIndex:
    Settings.llm = OpenAI(model=app_settings.openai_model, api_key=app_settings.openai_api_key, temperature=0)
    Settings.embed_model = OpenAIEmbedding(
        model=app_settings.embedding_model,
        api_key=app_settings.openai_api_key,
    )
    client = QdrantClient(url=app_settings.qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name=app_settings.qdrant_collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def upsert_documents(index: VectorStoreIndex, documents: list[Document]) -> None:
    if not documents:
        return

    for doc in documents:
        file_id = (doc.metadata or {}).get("file_id")
        if not file_id:
            continue
        delete_document_by_file_id(index=index, file_id=file_id)

    if hasattr(index, "insert_batch"):
        index.insert_batch(documents)
        return

    for doc in documents:
        index.insert(doc)


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

    user_email = user_email.strip().lower()
    if user_email:
        should_filters.append(FieldCondition(key="allowed_users[]", match=MatchValue(value=user_email)))

    for group in {g.strip().lower() for g in user_groups if g.strip()}:
        should_filters.append(FieldCondition(key="allowed_groups[]", match=MatchValue(value=group)))

    return Filter(should=should_filters)


def build_llama_acl_filter(user_email: str, user_groups: list[str]) -> MetadataFilters:
    filters = [
        MetadataFilter(key="is_public", value=True, operator=FilterOperator.EQ),
    ]

    if user_email.strip():
        filters.append(MetadataFilter(key="allowed_users", value=user_email.strip().lower(), operator=FilterOperator.CONTAINS))

    for group in {g.strip().lower() for g in user_groups if g.strip()}:
        filters.append(MetadataFilter(key="allowed_groups", value=group, operator=FilterOperator.CONTAINS))

    return MetadataFilters(filters=filters, condition=FilterCondition.OR)
