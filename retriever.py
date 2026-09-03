from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# Vector store location
# -----------------------------

VECTORSTORE_DIR = Path("vectorstore")


# -----------------------------
# Load vector store
# -----------------------------

def load_vectorstore():
    """
    Load the existing FAISS vector store.
    """

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. "
            "Run build_vectorstore.py first."
        )

    # Use the same embedding model that was used
    # when the vector store was created.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# -----------------------------
# Search documents
# -----------------------------

def search_documents(query, k=5):
    """
    Search the FAISS vector store for documents
    relevant to the user's query.

    Parameters:
        query: User's question/search query.
        k: Number of relevant documents to retrieve.

    Returns:
        A list of LangChain Document objects.
    """

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results