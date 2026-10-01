import html
import importlib
import os
import time
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors
import ingest

# Streamlit Cloud reruns chat.py after a git update but keeps the old
# ingest module in memory; reload it so new functions are always found
importlib.reload(ingest)
from ingest import ingest_documents, is_reference, parse_source, source_label

# ============================================================
# 1. Page setup
# ============================================================
st.set_page_config(
    page_title="Bharath's All-In-One Daily News",
    page_icon="📰"
)

with open(os.path.join(os.path.dirname(__file__), "style.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================================================
# 2. Load API key (Streamlit secrets first, env var as fallback)
# ============================================================
api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it in Streamlit's Secrets settings.")
    st.stop()

# Lighter model first (larger free allowance), full Flash as backup
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]

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

# ============================================================
# 3b. Header and sidebar
# ============================================================
# Shorter sidebar names for feeds (PDF filenames stay unchanged)
DISPLAY_NAMES = {
    "The Hindu - Tamil Nadu News": "TN News",
    "AI & Tech News Digest": "AI & Tech News",
    "Amazon.in Deals": "OnlineShopping Deals",
    "Finance News Digest": "Finance News",
    "Politics News Digest": "Politics News",
}

FEED_ICONS = [
    ("ai & tech", "🧠"),
    ("finance", "💰"),
    ("politics", "🏛️"),
    ("tamil nadu", "த"),
    ("amazon", "🛒"),
]

def feed_icon(title):
    lowered = title.lower()
    for keyword, icon in FEED_ICONS:
        if keyword in lowered:
            return icon
    return "📄"

# REF_ background PDFs are searchable but not listed in the UI
docs = [parse_source(name) for name, _ in signature if not is_reference(name)]
days = sorted({day for _, day in docs if day}, reverse=True)
latest = days[0] if days else None
feeds = sorted({title for title, _ in docs})

latest_label = f"{latest.day} {latest:%b %Y}" if latest else "—"
st.markdown(
    f"""
<div class="hero">
  <span class="hero-badge"><span class="live-dot"></span>Daily update available from 06:00 AM IST</span>
  <h1>Bharath's All-In-One Daily News</h1>
  <p>Ask anything about the last 7 days of news digests, deals and markets.</p>
  <div class="stats">
    <div class="stat"><b>{len(feeds)}</b><span>feeds</span></div>
    <div class="stat"><b>{latest_label}</b><span>latest</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

with st.sidebar:
    parts = [
        '<div class="side-title">📰 Latest News</div>'
    ]
    # Newest day first, undated PDFs last
    for day in days + [None]:
        items = sorted(
            (title for title, d in docs if d == day),
            key=lambda title: DISPLAY_NAMES.get(title, title).lower()
        )
        if not items:
            continue
        label = f"{day:%a} · {day.day} {day:%b}" if day else "Other"
        parts.append(f'<div class="day">{label}</div>')
        for title in items:
            parts.append(
                f'<div class="doc"><span class="doc-icon">{feed_icon(title)}</span>'
                f"DailyDigest_{html.escape(DISPLAY_NAMES.get(title, title))}</div>"
            )
    st.markdown("\n".join(parts), unsafe_allow_html=True)
    st.write("")
    if st.button("🧹 Clear chat", width="stretch"):
        st.session_state.messages = []

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
Source: {source_label(metadata['source'])}
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
    # Try each model in turn; each has its own free-tier allowance.
    #   503 (overloaded): retry the same model once after a short pause
    #   429 (quota used up) / 404 (model not available): move straight
    #   to the next model - retrying would only burn more quota
    # Returns (answer, None) or (None, last error).
    last_error = None
    for model in GEMINI_MODELS:
        for attempt in range(2):
            try:
                response = gemini.models.generate_content(
                    model=model,
                    contents=prompt
                )
                return response.text, None
            except (errors.ServerError, errors.ClientError) as e:
                # Visible under "Manage app" -> logs, to tell overload from quota
                print(f"Gemini error ({model}, attempt {attempt + 1}): {e.code} {e.status}: {e.message}", flush=True)
                if isinstance(e, errors.ClientError) and e.code not in (429, 404):
                    raise
                last_error = e
                if isinstance(e, errors.ServerError) and attempt == 0:
                    time.sleep(2)
                    continue
                break
    return None, last_error

# ============================================================
# 6. Chat UI
# ============================================================
def show_text(text):
    # Streamlit reads $...$ as a maths formula, which garbles
    # amounts like "$1B ... $10B"; escape $ so it shows as-is
    st.markdown(text.replace("$", "\\$"))

def show_sources(metadatas):
    # REF_ PDFs are shown by topic only ("Source = Artificial Intelligence"),
    # once each, never by file name
    seen = set()
    with st.expander("Sources"):
        for m in metadatas:
            if is_reference(m["source"]):
                line = f"- Source = {source_label(m['source'])}"
            else:
                line = f"- {m['source']} (chunk {m['chunk']})"
            if line not in seen:
                seen.add(line)
                show_text(line)

if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        show_text(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            show_sources(msg["sources"])

def ask(text):
    st.session_state.pending_question = text

typed = st.chat_input("Ask about today's news, deals, markets…")
question = typed or st.session_state.pop("pending_question", None)

# Clickable starter questions while the chat is empty
if not st.session_state.messages and not question:
    on = f" on {latest.day} {latest:%B}" if latest else ""
    # (button label, question sent to the RAG)
    suggestions = [
        (f"🧠 Top AI & tech headlines{on}", f"Top AI & tech headlines{on}"),
        (f"💰 Summarise the finance news{on}", f"Summarise the finance news{on}"),
        ("🛒 New Offers and Shopping Trends", f"What are the new offers and shopping trends{on}?"),
        (f"த What's happening in Tamil Nadu{on}?", f"What's happening in Tamil Nadu{on}?"),
    ]
    st.markdown('<div class="suggest-label">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, prompt) in enumerate(suggestions):
        cols[i % 2].button(
            label,
            key=f"suggest_{i}",
            on_click=ask,
            args=(prompt,),
            width="stretch"
        )

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        show_text(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            documents, metadatas = retrieve_documents(question)
            answer, error = generate_answer(question, documents, metadatas)
        if answer is None:
            if error is not None and error.code == 429:
                st.warning(
                    "The Gemini usage limit for this API key has been reached. "
                    "It resets daily; check your quota in Google AI Studio."
                )
            else:
                st.warning("Gemini is busy right now. Please try again in a minute.")
            # Drop the question so the retry starts clean
            st.session_state.messages.pop()
            st.stop()
        show_text(answer)
        show_sources(metadatas)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": metadatas
    })