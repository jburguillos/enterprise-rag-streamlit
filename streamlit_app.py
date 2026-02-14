from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

from app.rag.audit import write_audit_event
from app.rag.config import get_settings
from app.rag.index import get_index
from app.rag.rag_chat import run_chat_query
from app.rag.sync import full_ingest, incremental_sync

st.set_page_config(page_title="Enterprise Conversational RAG", layout="wide")


def _init_state() -> None:
    defaults = {
        "messages": [],
        "last_sync": None,
        "last_sync_stats": {"added": 0, "updated": 0, "deleted": 0},
        "next_sync_due": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sources(sources) -> None:
    if not sources:
        return
    st.subheader("Sources")
    for idx, source in enumerate(sources, start=1):
        metadata = source.node.metadata or {}
        preview = source.node.get_content()[:220].replace("\n", " ")
        st.markdown(
            f"**{idx}. {metadata.get('file_name') or metadata.get('title') or metadata.get('source') or metadata.get('file_id','unknown')}**  \\n"
            f"score={source.score:.4f} | is_public={metadata.get('is_public', False)}  \\n"
            f"`file_id={metadata.get('file_id', '')}`  \\n"
            f"{preview}..."
        )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _run_incremental(index, settings, user_email, user_groups, reason: str) -> None:
    stats = incremental_sync(
        index=index,
        folder_id=settings.drive_folder_id,
        service_account_json_path=settings.google_service_account_json,
        manifest_path=settings.manifest_path,
    )
    st.session_state.last_sync = _now_utc()
    st.session_state.last_sync_stats = stats
    write_audit_event(
        settings.audit_log_path,
        {
            "event_type": "sync_incremental",
            "reason": reason,
            "user_email": user_email,
            "user_groups": user_groups,
            "stats": stats,
        },
    )


def main() -> None:
    settings = get_settings()
    _init_state()
    index = get_index(settings)

    with st.sidebar:
        st.header("Access Context")
        user_email = st.text_input("user_email", value="")
        user_groups_raw = st.text_area("user_groups (one per line)", value="")
        user_groups = [line.strip().lower() for line in user_groups_raw.splitlines() if line.strip()]

        st.divider()
        st.header("Ingestion / Sync")
        if st.button("Initial Ingest (full)", use_container_width=True):
            with st.spinner("Running full ingest..."):
                stats = full_ingest(
                    index=index,
                    folder_id=settings.drive_folder_id,
                    service_account_json_path=settings.google_service_account_json,
                    manifest_path=settings.manifest_path,
                )
            st.session_state.last_sync = _now_utc()
            st.session_state.last_sync_stats = stats
            write_audit_event(
                settings.audit_log_path,
                {
                    "event_type": "ingest_full",
                    "user_email": user_email,
                    "user_groups": user_groups,
                    "stats": stats,
                },
            )
            st.success(f"Full ingest complete: +{stats['added']} ~{stats['updated']} -{stats['deleted']}")

        if st.button("Force Sync Now (incremental)", use_container_width=True):
            with st.spinner("Running incremental sync..."):
                _run_incremental(index, settings, user_email, user_groups, reason="manual")
            stats = st.session_state.last_sync_stats
            st.success(f"Incremental sync done: +{stats['added']} ~{stats['updated']} -{stats['deleted']}")

        enable_auto_sync = st.checkbox("enable_auto_sync", value=True)
        sync_interval_minutes = st.number_input("sync interval (minutes)", min_value=1, max_value=120, value=10)

        st.caption(f"Last sync: {st.session_state.last_sync}")
        s = st.session_state.last_sync_stats
        st.caption(f"Last sync stats: +{s['added']} ~{s['updated']} -{s['deleted']}")

    st.title("Enterprise-intermediate Conversational RAG")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if hasattr(st, "fragment"):

        @st.fragment(run_every="10s")
        def auto_sync_fragment() -> None:
            if not enable_auto_sync:
                return
            now = _now_utc()
            last_sync = st.session_state.last_sync
            if last_sync is None or now - last_sync >= timedelta(minutes=int(sync_interval_minutes)):
                _run_incremental(index, settings, user_email, user_groups, reason="auto")

        auto_sync_fragment()

    question = st.chat_input("Ask a question about your Drive docs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_chat_query(
                    index=index,
                    app_settings=settings,
                    messages=st.session_state.messages,
                    question=question,
                    user_email=user_email,
                    user_groups=user_groups,
                )
            st.markdown(result.answer)
            _render_sources(result.sources)

        st.session_state.messages.append({"role": "assistant", "content": result.answer})
        write_audit_event(
            settings.audit_log_path,
            {
                "event_type": "query",
                "user_email": user_email,
                "user_groups": user_groups,
                "prompt": question,
                "standalone_question": result.standalone_question,
                "num_sources": len(result.sources),
                "source_files": [
                    {
                        "file_id": (node.node.metadata or {}).get("file_id"),
                        "file_name": (node.node.metadata or {}).get("file_name")
                        or (node.node.metadata or {}).get("title")
                        or (node.node.metadata or {}).get("source"),
                    }
                    for node in result.sources
                ],
            },
        )


if __name__ == "__main__":
    main()
