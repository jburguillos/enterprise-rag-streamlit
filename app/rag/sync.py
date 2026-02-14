from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag.drive_acl import list_folder_files
from app.rag.ingest import load_drive_documents
from app.rag.index import delete_document_by_file_id, upsert_documents


Manifest = dict[str, dict[str, Any]]


def read_manifest(path: Path) -> Manifest:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _build_manifest_snapshot(files: dict[str, dict[str, Any]]) -> Manifest:
    return {
        fid: {
            "modifiedTime": item.get("modifiedTime", ""),
            "name": item.get("name", ""),
            "mimeType": item.get("mimeType", ""),
        }
        for fid, item in files.items()
    }


def incremental_sync(index, folder_id: str, service_account_json_path: str, manifest_path: Path) -> dict[str, int]:
    current_files = list_folder_files(service_account_json_path=service_account_json_path, folder_id=folder_id)
    current_manifest = _build_manifest_snapshot(current_files)
    previous_manifest = read_manifest(manifest_path)

    current_ids = set(current_manifest.keys())
    previous_ids = set(previous_manifest.keys())

    added = sorted(current_ids - previous_ids)
    deleted = sorted(previous_ids - current_ids)
    updated = sorted(
        fid
        for fid in (current_ids & previous_ids)
        if current_manifest[fid].get("modifiedTime") != previous_manifest[fid].get("modifiedTime")
    )

    for file_id in deleted:
        delete_document_by_file_id(index=index, file_id=file_id)

    changed_ids = added + updated
    if changed_ids:
        docs = load_drive_documents(
            folder_id=folder_id,
            service_account_json_path=service_account_json_path,
            file_ids=changed_ids,
        )
        upsert_documents(index=index, documents=docs)

    write_manifest(manifest_path, current_manifest)
    return {"added": len(added), "updated": len(updated), "deleted": len(deleted)}


def full_ingest(index, folder_id: str, service_account_json_path: str, manifest_path: Path) -> dict[str, int]:
    current_files = list_folder_files(service_account_json_path=service_account_json_path, folder_id=folder_id)
    docs = load_drive_documents(folder_id=folder_id, service_account_json_path=service_account_json_path)
    upsert_documents(index=index, documents=docs)
    write_manifest(manifest_path, _build_manifest_snapshot(current_files))
    return {"added": len(docs), "updated": 0, "deleted": 0}
