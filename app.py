import streamlit as st
from pathlib import Path

from retriever import search_documents
from generator import generate_answer
from ingest import ingest_documents
from build_vectorstore import rebuild_vectorstore


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="RAG Search Engine",
    page_icon="🔎",
    layout="wide",
)


# -----------------------------
# Header
# -----------------------------

st.title("🔎 RAG Search Engine")
st.caption(
    "Search your documents using Retrieval-Augmented Generation"
)


# -----------------------------
# Folder locations
# -----------------------------

DOCUMENTS_DIR = Path("data/documents")

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("Settings")

    # -------------------------
    # PDF Upload
    # -------------------------

    st.subheader("📄 Add a document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:

        st.caption(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "⚙️ Process PDF",
            use_container_width=True
        ):

            # -------------------------
            # Save uploaded PDF
            # -------------------------

            pdf_path = DOCUMENTS_DIR / uploaded_file.name

            with open(
                pdf_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            st.success(
                f"Uploaded: {uploaded_file.name}"
            )

            # -------------------------
            # Ingest PDF
            # -------------------------

            with st.spinner(
                "📖 Extracting and processing PDF..."
            ):

                ingestion_success = ingest_documents()

            if ingestion_success:

                st.success(
                    "PDF text processed successfully!"
                )

                # -------------------------
                # Build vector store
                # -------------------------

                with st.spinner(
                    "🧠 Building vector store..."
                ):

                    vectorstore_success = (
                        rebuild_vectorstore()
                    )

                if vectorstore_success:

                    st.success(
                        "✅ PDF indexed successfully!"
                    )

                else:

                    st.error(
                        "❌ Could not build the vector store."
                    )

            else:

                st.error(
                    "❌ Could not process the PDF."
                )

    st.divider()

    # -------------------------
    # Retrieval settings
    # -------------------------

    st.subheader("Search Settings")

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    st.info(
        "Upload a PDF, process it, and then "
        "ask questions about your documents."
    )


# -----------------------------
# Search box
# -----------------------------

query = st.text_input(
    "Ask a question",
    placeholder="e.g. How many cards are in a standard deck?",
)


# -----------------------------
# Search
# -----------------------------

if query.strip():

    with st.spinner(
        "🔎 Searching documents..."
    ):

        # -------------------------
        # Retrieve relevant documents
        # -------------------------

        results = search_documents(
            query=query,
            k=top_k,
        )

        # -------------------------
        # Generate answer using Gemini
        # -------------------------

        answer = generate_answer(
            question=query,
            results=results,
        )

    # -------------------------
    # Answer
    # -------------------------

    st.subheader("Answer")

    st.write(answer)

    # -------------------------
    # Sources
    # -------------------------

    st.divider()

    st.subheader("📚 Sources")

    if results:

        for i, result in enumerate(
            results,
            start=1
        ):

            source = result.metadata.get(
                "source",
                "Unknown source"
            )

            page = result.metadata.get(
                "page",
                "N/A"
            )

            text = result.page_content

            with st.expander(
                f"📄 Source {i}: {source} — Page {page}"
            ):

                st.write(text)

    else:

        st.warning(
            "No relevant sources were found."
        )