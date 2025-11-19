
import streamlit as st
from typing import List, Tuple
import tempfile
import os
import re
from io import StringIO
import numpy as np

# Lazy imports for heavy libraries
@st.cache_resource
def load_sentence_transformer(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_resource
def load_faiss_index(dim: int):
    import faiss
    # We'll create an index dynamically when documents are added
    return None

@st.cache_resource
def load_generator(model_name="t5-small"):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, model

@st.cache_resource
def load_sentiment_pipeline(model_name="siebert/sentiment-roberta-large-english"):
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=model_name)

def text_from_uploaded_file(uploaded_file) -> str:
    # handle text and simple PDF fallback
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".txt") or name.endswith(".md") or name.endswith(".csv"):
        try:
            return data.decode("utf-8", errors="ignore")
        except:
            return str(data)
    elif name.endswith(".pdf"):
        try:
            # lazy import
            import pdfplumber
            text = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception as e:
            return f"[could not extract PDF text: {e}]"
    else:
        try:
            return data.decode("utf-8", errors="ignore")
        except:
            return str(data)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_faiss_index(embeddings: np.ndarray):
    import faiss
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    return index

def retrieve_top_k(query: str, embedder, index, doc_chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype(np.float32), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append((doc_chunks[int(idx)], float(dist)))
    return results

def generate_answer(prompt: str, tokenizer, model, max_length=128) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_bleu(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    smooth = SmoothingFunction().method1
    try:
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
    except Exception:
        score = 0.0
    return float(score)

# --- Streamlit UI ---
st.set_page_config(page_title="Doc-Chat (Streamlit prototype)", layout="wide")
st.title("📄🔎 Doc-Chat — Streamlit prototype")

with st.sidebar:
    st.header("Settings & Models")
    st.markdown("Upload documents (txt, md, csv, pdf). The app will chunk and index them for retrieval.")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt","md","csv","pdf"])
    k = st.number_input("Retrieval top-k", min_value=1, max_value=10, value=3, step=1)
    use_sentiment = st.checkbox("Run sentiment analysis on responses", value=True)
    show_bleu = st.checkbox("Show BLEU evaluation controls", value=True)
    st.markdown("**Model backends (lazy-loaded)**")
    st.write("Generator: t5-small (recommended for prototype)")
    st.write("Embedder: sentence-transformers/all-MiniLM-L6-v2")
    st.write("Sentiment: siebert/sentiment-roberta-large-english")

# Load models (lazy)
with st.spinner("Loading embedder..."):
    embedder = load_sentence_transformer()
with st.spinner("Loading generator (tokenizer + model)... this may take a while first run"):
    gen_tokenizer, gen_model = load_generator()
if use_sentiment:
    with st.spinner("Loading sentiment model..."):
        sentiment_pipe = load_sentiment_pipeline()
else:
    sentiment_pipe = None

# Process uploaded docs
all_text = ""
doc_chunks = []
if uploaded_files:
    for f in uploaded_files:
        text = text_from_uploaded_file(f)
        all_text += "\n\n" + text
    doc_chunks = chunk_text(all_text, chunk_size=250, overlap=50)
    if len(doc_chunks) == 0:
        st.warning("No text extracted from uploaded files.")
else:
    st.info("No documents uploaded — retrieval will be empty. You can still ask general questions.")

# Build embeddings + FAISS
if doc_chunks:
    with st.spinner("Creating embeddings and FAISS index..."):
        embeddings = embedder.encode(doc_chunks, convert_to_numpy=True)
        faiss_index = build_faiss_index(np.array(embeddings))
else:
    faiss_index = None

# Chat interface
if "history" not in st.session_state:
    st.session_state.history = []

st.header("Chat")
col1, col2 = st.columns([3,1])

with col1:
    user_input = st.text_input("You:", value="", placeholder="Ask something about the uploaded docs or anything general")
    if st.button("Send", key="send"):
        if not user_input.strip():
            st.warning("Please enter a question.")
        else:
            # retrieval
            context = ""
            if faiss_index is not None:
                hits = retrieve_top_k(user_input, embedder, faiss_index, doc_chunks, k=k)
                context = "\n\n".join([h[0] for h in hits])
            prompt = f"Context: {context}\n\nQuestion: {user_input}\nAnswer concisely:"
            answer = generate_answer(prompt, gen_tokenizer, gen_model, max_length=256)
            sentiment_res = None
            if sentiment_pipe is not None:
                try:
                    sentiment_res = sentiment_pipe(answer[:512])
                except Exception as e:
                    sentiment_res = [{"label": "ERROR", "score": 0.0, "error": str(e)}]
            st.session_state.history.append({
                "question": user_input,
                "answer": answer,
                "context": context,
                "sentiment": sentiment_res
            })
            # clear input box (Streamlit persists it — workaround is nothing)
    if st.button("Clear chat", key="clear"):
        st.session_state.history = []

with col2:
    st.markdown("### Controls")
    if show_bleu:
        st.markdown("Evaluate BLEU between a reference and the last generated answer")
        ref = st.text_area("Reference (gold) text for BLEU", value="", height=120)
        if st.button("Compute BLEU", key="bleu"):
            if not st.session_state.history:
                st.warning("No generated answers yet to compare with.")
            else:
                hyp = st.session_state.history[-1]["answer"]
                score = compute_bleu(ref, hyp)
                st.success(f"BLEU score: {score:.4f}")

# Display chat history
st.markdown("---")
for i, turn in enumerate(reversed(st.session_state.history[-20:])):
    idx = len(st.session_state.history) - i - 1
    st.markdown(f"**You:** {turn['question']}")
    st.markdown(f"**Bot:** {turn['answer']}")
    if use_sentiment and turn.get("sentiment"):
        st.markdown(f"**Sentiment:** {turn['sentiment']}")
    if st.expander("Context (retrieved chunks)"):
        st.write(turn.get("context",""))

st.markdown("---")
st.markdown("### Notes / Troubleshooting")
st.markdown("""
- To run this app locally:
```
pip install streamlit sentence-transformers transformers torch faiss-cpu pdfplumber nltk
python -m nltk.downloader punkt
streamlit run streamlit_app.py
```
- First run will download models (may take several minutes).
- If you see version conflicts, create a fresh virtual environment (recommended).
- This prototype prioritizes simplicity and avoids saving history to disk.
""")
