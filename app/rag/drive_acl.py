from __future__ import annotations

from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def build_drive_service(service_account_json_path: str):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json_path,
        scopes=DRIVE_SCOPES,
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_folder_files(service_account_json_path: str, folder_id: str) -> dict[str, dict[str, Any]]:
    service = build_drive_service(service_account_json_path)
    files: dict[str, dict[str, Any]] = {}
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"

    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                corpora="allDrives",
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )
        for item in response.get("files", []):
            file_id = item.get("id")
            if file_id:
                files[file_id] = item

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


def get_file_permissions(service_account_json_path: str, file_id: str) -> dict[str, Any]:
    default = {
        "allowed_users": [],
        "allowed_groups": [],
        "allowed_domains": [],
        "is_public": False,
    }
    try:
        service = build_drive_service(service_account_json_path)
        response = (
            service.permissions()
            .list(
                fileId=file_id,
                fields="permissions(type, emailAddress, domain, role, allowFileDiscovery)",
                supportsAllDrives=True,
            )
            .execute()
        )
    except Exception:
        return default

    users: set[str] = set()
    groups: set[str] = set()
    domains: set[str] = set()
    is_public = False

    for permission in response.get("permissions", []):
        p_type = permission.get("type")
        if p_type == "user" and permission.get("emailAddress"):
            users.add(permission["emailAddress"].lower())
        elif p_type == "group" and permission.get("emailAddress"):
            groups.add(permission["emailAddress"].lower())
        elif p_type == "domain" and permission.get("domain"):
            domains.add(permission["domain"].lower())
        elif p_type == "anyone":
            is_public = True

    return {
        "allowed_users": sorted(users),
        "allowed_groups": sorted(groups),
        "allowed_domains": sorted(domains),
        "is_public": is_public,
    }


def acl_metadata_for_file(service_account_json_path: str, file_id: str | None) -> dict[str, Any]:
    if not file_id:
        return {
            "allowed_users": [],
            "allowed_groups": [],
            "allowed_domains": [],
            "is_public": False,
        }
    return get_file_permissions(service_account_json_path=service_account_json_path, file_id=file_id)
