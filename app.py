"""Streamlit web interface for MiMo-RAG."""

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import RAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MiMo-RAG",
    page_icon="📄",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "MiMo API Key",
        type="password",
        placeholder="Enter your MiMo API key...",
        help="Get your API key from https://100t.xiaomimimo.com/",
    )

    model = st.selectbox(
        "Model",
        ["MiMo-V2.5", "MiMo-V2.5-Pro", "MiMo-V2-Flash", "MiMo-V2-Pro", "MiMo-7B-RL"],
        index=0,
        help="MiMo-V2.5 is the latest open-source model (310B MoE, 1M context)",
    )

    st.divider()
    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        accept_multiple_files=True,
        type=["pdf", "docx", "md", "txt"],
        label_visibility="collapsed",
    )

    if st.button("🚀 Ingest Documents", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your MiMo API key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Indexing documents..."):
                st.session_state.pipeline = RAGPipeline(api_key=api_key)
                st.session_state.ingested = []
                total_chunks = 0

                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        n = st.session_state.pipeline.ingest(tmp_path)
                        st.session_state.ingested.append(uploaded_file.name)
                        total_chunks += n
                    except Exception as e:
                        st.error(f"Error ingesting {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

                st.success(f"✅ Indexed {len(st.session_state.ingested)} file(s) → {total_chunks} chunks")

    if st.session_state.ingested:
        st.subheader("📑 Indexed Documents")
        for name in st.session_state.ingested:
            st.write(f"• {name}")

# ── Main chat UI ──────────────────────────────────────────────────────────────
st.title("📄 MiMo-RAG")
st.caption("Chat with your documents powered by MiMo API")

if not st.session_state.pipeline:
    st.info("👈 Configure your API key and upload documents in the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.write(f"• `{src}`")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.pipeline.query(prompt)
                    st.markdown(response.answer)
                    if response.sources:
                        with st.expander("📎 Sources"):
                            for src in response.sources:
                                st.write(f"• `{src}`")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
