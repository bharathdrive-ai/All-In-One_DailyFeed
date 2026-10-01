import os
import re
from datetime import date

import pymupdf


# Bump when the chunk format changes so every PDF is re-indexed
INDEX_VERSION = 3

# Readable titles for the feed keys in "DailyDigest_<Feed>_<date>.pdf"
FEED_TITLES = {
    "AITechNews": "AI & Tech News Digest",
    "FinanceNews": "Finance News Digest",
    "PoliticsNews": "Politics News Digest",
    "TN-News": "The Hindu - Tamil Nadu News",
    "AmazonDeals": "Amazon.in Deals",
}


# Background PDFs named "REF_<Topic>.pdf" are searched like any other PDF
# but hidden from the sidebar, and cited only by topic
REFERENCE_PREFIX = "REF_"


def is_reference(filename):

    return filename.upper().startswith(REFERENCE_PREFIX)


def reference_topic(filename):

    # "REF_Artificial Intelligence.pdf" / "REF_Artificial_Intelligence.pdf"
    # / "REF_ArtificialIntelligence.pdf" -> "Artificial Intelligence"
    topic = os.path.splitext(filename)[0][len(REFERENCE_PREFIX):]

    topic = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", topic.replace("_", " "))

    return " ".join(topic.split()) or "Reference"


def source_label(filename):

    # What users (and Gemini) see as the source of an answer
    if is_reference(filename):

        return reference_topic(filename)

    return filename


def parse_source(filename):

    # Returns (readable title, date or None) for both naming styles:
    #   "DailyDigest_AITechNews_2026-09-29.pdf"
    #       -> ("AI & Tech News Digest", 2026-09-29)
    #   "AI & Tech News Digest - 2026-09-29.pdf"
    #       -> ("AI & Tech News Digest", 2026-09-29)
    #   "REF_Artificial Intelligence.pdf"
    #       -> ("Artificial Intelligence", None)
    if is_reference(filename):

        return reference_topic(filename), None

    stem = os.path.splitext(filename)[0]

    match = re.match(r"^DailyDigest_(.+?)_(\d{4}-\d{2}-\d{2})$", stem)

    if match:

        key, stamp = match.groups()

        # Unknown feeds: "WorldSportsNews" -> "World Sports News"
        title = FEED_TITLES.get(
            key,
            re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key).replace("-", " ")
        )

    else:

        match = re.search(r"\s*-?\s*(\d{4}-\d{2}-\d{2})$", stem)

        if not match:

            return stem, None

        title, stamp = stem[:match.start()], match.group(1)

    try:

        return title, date.fromisoformat(stamp)

    except ValueError:

        return title, None


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

    # "DailyDigest_AITechNews_2026-09-29.pdf" ->
    # "Document: AI & Tech News Digest
    #  Date: 29 September 2026 (2026-09-29)"
    title, day = parse_source(filename)

    header = f"Document: {title}"

    if day:

        header += (
            f"\nDate: {day.day} {day:%B %Y} ({day.isoformat()})"
        )

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