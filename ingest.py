import os
import re
from datetime import date

import pymupdf


# Bump when the chunk format changes so every PDF is re-indexed
INDEX_VERSION = 2


# ============================================================
# 1. Extract text from PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    document = pymupdf.open(pdf_path)

    pages = []

    for page in document:

        text = page.get_text()

        if text.strip():

            pages.append(text)

    document.close()

    return "\n".join(pages)


# ============================================================
# 2. Create chunks
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def describe_source(filename):

    # "AI & Tech News Digest - 2026-09-29.pdf" ->
    # "Document: AI & Tech News Digest - 2026-09-29
    #  Date: 29 September 2026 (2026-09-29)"
    title = os.path.splitext(filename)[0]

    header = f"Document: {title}"

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)

    if match:

        try:

            day = date(*map(int, match.groups()))

            header += (
                f"\nDate: {day.day} {day:%B %Y} ({day.isoformat()})"
            )

        except ValueError:

            pass

    return header


# ============================================================
# 3. Files already in the collection
# ============================================================

def get_indexed_files(collection):

    results = collection.get(include=["metadatas"])

    return {
        metadata["source"]: (
            metadata.get("size"),
            metadata.get("version")
        )
        for metadata in results["metadatas"]
    }


# ============================================================
# 4. Ingest documents (only new or changed PDFs)
# ============================================================

def ingest_documents(collection, embedding_model, folder="documents"):

    pdf_files = [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(".pdf")
    ]

    indexed = get_indexed_files(collection)

    # Remove chunks of PDFs that were deleted from the folder
    for source in indexed:

        if source not in pdf_files:

            print(f"Removing: {source}")

            collection.delete(where={"source": source})

    if not pdf_files:

        print("No PDF files found.")

        return

    total_chunks = 0

    for filename in pdf_files:

        pdf_path = os.path.join(
            folder,
            filename
        )

        size = os.path.getsize(pdf_path)

        if indexed.get(filename) == (size, INDEX_VERSION):

            print(f"Already indexed: {filename}")

            continue

        # Changed file or old format: drop its old chunks before re-adding
        if filename in indexed:

            collection.delete(where={"source": filename})

        print()
        print("=" * 50)
        print(f"Processing: {filename}")
        print("=" * 50)

        # Extract text
        text = extract_text_from_pdf(
            pdf_path
        )

        print(
            f"Characters extracted: {len(text)}"
        )

        if not text.strip():

            print("No text found in PDF.")

            continue

        # Create chunks
        # Put the document name and date in every chunk so questions
        # like "AI headlines on 29 September" match the right PDF
        header = describe_source(filename)

        chunks = [
            f"{header}\n\n{chunk}"
            for chunk in create_chunks(text)
        ]

        print(
            f"Chunks created: {len(chunks)}"
        )

        # Create embeddings
        print("Creating embeddings...")

        embeddings = embedding_model.encode(
            chunks,
            show_progress_bar=True
        ).tolist()

        # IDs
        ids = [
            f"{filename}_{i}"
            for i in range(len(chunks))
        ]

        # Metadata
        metadatas = [
            {
                "source": filename,
                "chunk": i,
                "size": size,
                "version": INDEX_VERSION
            }
            for i in range(len(chunks))
        ]

        # Store
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"Stored {len(chunks)} chunks"
        )

        total_chunks += len(chunks)

    print()
    print("=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(
        f"Total chunks: {total_chunks}"
    )


# ============================================================
# 5. Run
# ============================================================

if __name__ == "__main__":

    import chromadb
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded.")

    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="documents")

    ingest_documents(collection, embedding_model)