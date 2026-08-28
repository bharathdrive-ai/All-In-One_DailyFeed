import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ============================================================
# 1. Page setup
# ============================================================
st.set_page_config(page_title="Mini RAG", page_icon="📚")
st.title("📚 Mini RAG")

# ============================================================
# 2. Load API key (Streamlit secrets first, env var as fallback)
# ============================================================
api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
if not api_key:
    st.error("ANTHROPIC_API_KEY not found. Add it in Streamlit's Secrets settings.")
    st.stop()

# ============================================================
# 3. Cache the heavy resources so they load once, not every rerun
# ============================================================
@st.cache_resource
def load_claude_client():
    return Anthropic(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    return chroma_client.get_collection(name="documents")

claude = load_claude_client()
embedding_model = load_embedding_model()
collection = load_collection()

# ============================================================
# 4. Retrieve relevant chunks (unchanged)
# ============================================================
def retrieve_documents(question):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
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
    response = claude.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = ""
    for block in response.content:
        if block.type == "text":
            answer += block.text
    return answer

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
        st.write(answer)
        with st.expander("Sources"):
            for m in metadatas:
                st.write(f"- {m['source']} (chunk {m['chunk']})")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": metadatas
    })