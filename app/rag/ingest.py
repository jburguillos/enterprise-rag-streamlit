from __future__ import annotations

import inspect
from typing import Any, Iterable

from llama_index.core import Document
from llama_index.readers.google import GoogleDriveReader

from app.rag.drive_acl import acl_metadata_for_file


def _attempt_load_data(reader: GoogleDriveReader, kwargs_options: list[dict[str, Any]]) -> list[Document]:
    errors: list[str] = []
    for kwargs in kwargs_options:
        try:
            sig = inspect.signature(reader.load_data)
            supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return reader.load_data(**supported)
        except Exception as exc:  # noqa: PERF203
            errors.append(str(exc))
    raise RuntimeError("GoogleDriveReader load_data failed with all known signatures: " + " | ".join(errors))


def load_drive_documents(
    folder_id: str,
    service_account_json_path: str,
    file_ids: list[str] | None = None,
) -> list[Document]:
    reader = GoogleDriveReader()
    kwargs_options = [
        {
            "folder_id": folder_id,
            "service_account_key": service_account_json_path,
            "file_ids": file_ids,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        },
        {
            "folder_id": folder_id,
            "service_account_key_file": service_account_json_path,
            "file_ids": file_ids,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        },
        {
            "folder_id": folder_id,
            "service_account_key": service_account_json_path,
            "file_ids": file_ids,
        },
    ]
    docs = _attempt_load_data(reader, kwargs_options)
    return enrich_documents_with_acl(docs, service_account_json_path)


def enrich_documents_with_acl(documents: Iterable[Document], service_account_json_path: str) -> list[Document]:
    enriched: list[Document] = []
    for doc in documents:
        metadata = dict(doc.metadata or {})
        file_id = (
            metadata.get("file_id")
            or metadata.get("id")
            or metadata.get("source_id")
            or metadata.get("document_id")
        )
        acl = acl_metadata_for_file(service_account_json_path=service_account_json_path, file_id=file_id)

        if not file_id:
            metadata.update(acl)
            metadata["file_id"] = ""
            doc.metadata = metadata
            enriched.append(doc)
            continue

        metadata.update(acl)
        metadata["file_id"] = file_id
        metadata["source"] = metadata.get("source") or metadata.get("name") or file_id
        doc.metadata = metadata
        doc.id_ = str(file_id)
        enriched.append(doc)

    return enriched
