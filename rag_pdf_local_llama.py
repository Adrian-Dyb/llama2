import os
import fitz
import faiss
import numpy as np
import streamlit as st
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings

# ---- CONFIG ----
EMBED_MODEL_ID = 'intfloat/e5-small-v2'
LLAMA_PATH = "llama-2-7b-chat.ggmlv3.q2_K.bin"

# ---- LOAD LOCAL MODEL ----
@st.cache_resource
def load_llama():
    return Llama(model_path=LLAMA_PATH, n_ctx=512, n_batch=128)

llm = load_llama()

# ---- LOAD EMBEDDINGS ----
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID, model_kwargs={"device": "cpu"})

embedder = load_embedder()

# ---- PARSE PDF ----
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---- CREATE FAISS INDEX ----
def create_faiss_index(docs):
    texts = [doc["text"] for doc in docs]
    metadata = [{"filename": doc["filename"], "text": doc["text"]} for doc in docs]
    embeddings = [embedder.embed_query(t) for t in texts]
    matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, metadata

# ---- RAG Search ----
def retrieve_similar(query, index, metadata, k=3):
    embedding = np.array([embedder.embed_query(query)]).astype("float32")
    D, I = index.search(embedding, k)
    return [metadata[i] for i in I[0]]

def ask_llama(context, question):
    prompt = f"""You are a helpful assistant. Use the following context to answer.

Context:
{context}

Question: {question}
Answer:"""
    response = llm(prompt, max_tokens=256, temperature=0.7, top_p=0.9)
    return response['choices'][0]['text'].strip()

# ---- STREAMLIT UI ----
st.title("📄🔍 RAG PDF + Local LLaMA Chatbot")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
question = st.text_input("Ask a question based on the uploaded PDFs")

if uploaded_files:
    docs = []
    for f in uploaded_files:
        text = load_pdf(f)
        docs.append({"filename": f.name, "text": text})
    
    index, metadata = create_faiss_index(docs)
    st.success("✅ PDFs indexed.")

    if question:
        results = retrieve_similar(question, index, metadata)
        context = "\n\n".join([r["text"][:1000] for r in results])  # ogranicz długość
        answer = ask_llama(context, question)
        
        st.subheader("🧠 Answer:")
        st.write(answer)

        with st.expander("🔍 Context used"):
            st.write(context)
