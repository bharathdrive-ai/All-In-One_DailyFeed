import os
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic


# ============================================================
# 1. Load API key
# ============================================================

load_dotenv()

api_key = os.getenv(
    "ANTHROPIC_API_KEY"
)

if not api_key:

    raise ValueError(
        "ANTHROPIC_API_KEY not found in .env"
    )


# ============================================================
# 2. Claude client
# ============================================================

claude = Anthropic(
    api_key=api_key
)


# ============================================================
# 3. Embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 4. Open ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="documents"
)


# ============================================================
# 5. Retrieve relevant chunks
# ============================================================

def retrieve_documents(question):

    # Convert question into vector
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search vector database
    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=3
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    return documents, metadatas


# ============================================================
# 6. Generate answer
# ============================================================

def generate_answer(
    question,
    documents,
    metadatas
):

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        context_parts.append(
            f"""
Source: {metadata['source']}
Chunk: {metadata['chunk']}

{document}
"""
        )

    context = "\n".join(
        context_parts
    )


    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the provided context.

Do not use outside knowledge.

Do not make up information.

If the answer cannot be found in the context,
say:

"Sorry dear, I couldn't find any information from the provided document(s)."

Keep the answer clear and concise.


==============================
CONTEXT
==============================

{context}


==============================
QUESTION
==============================

{question}
"""


    # ========================================================
    # Claude API
    # ========================================================

    response = claude.messages.create(

        # Current Claude Sonnet model
        model="claude-sonnet-4-6",

        max_tokens=500,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # Extract response text
    answer = ""

    for block in response.content:

        if block.type == "text":

            answer += block.text


    return answer


# ============================================================
# 7. Main
# ============================================================

def main():

    print()
    print("=" * 50)
    print("             MINI RAG")
    print("=" * 50)

    print("Type 'exit' to quit.")

    while True:

        question = input("\nHow may I assist you? ")

        if question.strip().lower() == "exit":

            print("Thank you for using MINI RAG... Please come again...!")

            break

        if not question.strip():

            continue


        # Retrieve
        documents, metadatas = retrieve_documents(
            question
        )


        # Show sources
        print()
        print("Retrieved sources:")

        for metadata in metadatas:

            print(
                f"- {metadata['source']} "
                f"(chunk {metadata['chunk']})"
            )


        # Generate
        print()
        print("I'm thinking... Give me a moment...")

        answer = generate_answer(
            question,
            documents,
            metadatas
        )


        # Answer
        print()
        print("Answer:")
        print(answer)


# ============================================================
# 8. Start
# ============================================================

if __name__ == "__main__":

    main()