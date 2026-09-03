import fitz
import json
from pathlib import Path


# -----------------------------
# Folder locations
# -----------------------------

DOCUMENTS_DIR = Path("data/documents")
OUTPUT_DIR = Path("data")

OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


# -----------------------------
# Chunking settings
# -----------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# -----------------------------
# Extract text from PDFs
# -----------------------------

def load_pdfs():

    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        print(f"Put your PDFs inside: {DOCUMENTS_DIR}")
        return documents

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        pdf = fitz.open(pdf_path)

        for page_number, page in enumerate(pdf):

            text = page.get_text().strip()

            if not text:
                continue

            documents.append({
                "text": text,
                "source": pdf_path.name,
                "page": page_number + 1
            })

        pdf.close()

    return documents


# -----------------------------
# Split text into chunks
# -----------------------------

def create_chunks(documents):

    chunks = []

    chunk_id = 0

    for document in documents:

        text = document["text"]
        source = document["source"]
        page = document["page"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "source": source,
                    "page": page
                })

                chunk_id += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# -----------------------------
# Save chunks
# -----------------------------

def save_chunks(chunks):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved {len(chunks)} chunks.")
    print(f"Output: {OUTPUT_FILE}")


# -----------------------------
# Main ingestion function
# -----------------------------

def ingest_documents():

    print("Starting document ingestion...\n")

    documents = load_pdfs()

    if not documents:
        return False

    print(f"\nExtracted {len(documents)} pages.")

    chunks = create_chunks(documents)

    save_chunks(chunks)

    print("\nIngestion complete!")

    return True


# -----------------------------
# Run directly
# -----------------------------

if __name__ == "__main__":

    ingest_documents()