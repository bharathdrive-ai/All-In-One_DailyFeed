import os
import re
import time
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors
from ingest import ingest_documents

# ============================================================
# 1. Page setup
# ============================================================
st.set_page_config(
    page_title="Bharath's All-In-One Daily New RAG",
    page_icon="📰"
)
st.title("📰 Bharath's All-In-One Daily New RAG")
st.caption("Ask questions about the last 7 days of daily digests.")

# ============================================================
# 2. Load API key (Streamlit secrets first, env var as fallback)
# ============================================================
api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it in Streamlit's Secrets settings.")
    st.stop()

# ============================================================
# 3. Cache the heavy resources so they load once, not every rerun
# ============================================================
@st.cache_resource
def load_gemini_client():
    return genai.Client(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def documents_signature(folder="documents"):
    # Changes whenever a PDF is added, removed or replaced
    return tuple(sorted(
        (name, os.path.getsize(os.path.join(folder, name)))
        for name in os.listdir(folder)
        if name.lower().endswith(".pdf")
    ))

@st.cache_resource
def load_collection(signature):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="documents")
    with st.spinner("Indexing new documents..."):
        ingest_documents(collection, embedding_model)
    return collection

gemini = load_gemini_client()
embedding_model = load_embedding_model()
signature = documents_signature()
collection = load_collection(signature)

with st.sidebar:
    st.header("Documents")
    st.write(f"{len(signature)} PDFs indexed")
    # Newest dated PDFs first, undated ones (no YYYY-MM-DD) last
    def newest_first(item):
        match = re.search(r"\d{4}-\d{2}-\d{2}", item[0])
        return (match.group(0) if match else "", item[0])
    for name, _ in sorted(signature, key=newest_first, reverse=True):
        st.write(f"- {name[:-4]}")

# ============================================================
# 4. Retrieve relevant chunks (unchanged)
# ============================================================
def retrieve_documents(question):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=6
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return documents, metadatas

# ============================================================
# 5. Generate answer (unchanged)
# ============================================================
def generate_answer(question, documents, metadatas):
    context_parts = []
    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"""
Source: {metadata['source']}
Chunk: {metadata['chunk']}
{document}
"""
        )
    context = "\n".join(context_parts)
    prompt = f"""
You are a document question-answering assistant.
Answer the question using ONLY the provided context.
Do not use outside knowledge.
Do not make up information.
If the answer cannot be found in the context,
say:
"I couldn't find the answer in the provided documents."
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
    # Gemini returns 503 (overloaded) or 429 (rate limited) at busy
    # times; these usually clear within seconds, so retry a few times
    for attempt in range(3):
        try:
            response = gemini.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            return response.text
        except (errors.ServerError, errors.ClientError) as e:
            retryable = isinstance(e, errors.ServerError) or e.code == 429
            if not retryable:
                raise
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return None

# ============================================================
# 6. Chat UI
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']} (chunk {s['chunk']})")
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            documents, metadatas = retrieve_documents(question)
            answer = generate_answer(question, documents, metadatas)
        if answer is None:
            st.warning("Gemini is busy right now. Please try again in a minute.")
            # Drop the question so the retry starts clean
            st.session_state.messages.pop()
            st.stop()
        st.write(answer)
        with st.expander("Sources"):
            for m in metadatas:
                st.write(f"- {m['source']} (chunk {m['chunk']})")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": metadatas
    })