import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# File locations
# -----------------------------

CHUNKS_FILE = Path("data/chunks.json")
VECTORSTORE_DIR = Path("vectorstore")


# -----------------------------
# Load chunks
# -----------------------------

def load_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. "
            "Run ingest.py first."
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return chunks


# -----------------------------
# Build vector store
# -----------------------------

def build_vectorstore(chunks):

    documents = []

    for chunk in chunks:

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "id": chunk["id"]
                }
            )
        )

    print(
        f"Creating embeddings for "
        f"{len(documents)} chunks..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    print(
        "\nVector store created successfully!"
    )

    print(
        f"Saved to: {VECTORSTORE_DIR}"
    )


# -----------------------------
# Build from current chunks
# -----------------------------

def rebuild_vectorstore():

    print(
        "Loading chunks..."
    )

    chunks = load_chunks()

    if not chunks:

        print(
            "No chunks found in chunks.json."
        )

        return False

    build_vectorstore(chunks)

    return True


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    rebuild_vectorstore()

    print("\nDone!")