
import os
import io
import re
import json
import time
import shutil
import pathlib
import tempfile
import requests
from typing import List, Tuple, Optional


import numpy as np
import faiss


from bs4 import BeautifulSoup
import fitz  
import docx  


import gradio as gr

HF_API_KEY = "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"  
HF_GENERATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
INDEX_DIR = "vector_index"
INDEX_FILE = os.path.join(INDEX_DIR, "docs.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.jsonl")
EMBED_DIM = 384  


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5



def ensure_index_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)

def sanitize_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return chunks

def read_pdf(file_obj: io.BytesIO) -> str:
    text = []
    with fitz.open(stream=file_obj.read(), filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return sanitize_text("\n".join(text))

def read_docx(file_obj: io.BytesIO) -> str:
    
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_obj.read())
        tmp.flush()
        p = tmp.name
    doc = docx.Document(p)
    out = "\n".join(p.text for p in doc.paragraphs)
    os.remove(p)
    return sanitize_text(out)

def read_txt(file_obj: io.BytesIO) -> str:
    content = file_obj.read().decode("utf-8", errors="ignore")
    return sanitize_text(content)

def fetch_url_to_text(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "RAGBot/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Remove script/style
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    text = soup.get_text(separator=" ")
    return sanitize_text(text)

def hf_inference_generate(messages: List[dict], max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.9) -> str:
    """
    Uses Hugging Face Inference API chat/messages endpoint when available,
    otherwise falls back to text-generation style prompt formatting.
    """
    url = f"https://api-inference.huggingface.co/models/{HF_GENERATION_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": messages,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    }
    # Some models expect "inputs" as a string prompt; many chat-tuned models accept message lists.
    # Try messages first, then fall back.
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        js = resp.json()
        # Responses differ by model; unify common cases.
        if isinstance(js, list) and len(js) > 0:
            # text-generation returns list of dicts with 'generated_text'
            if "generated_text" in js:
                return js["generated_text"]
            # Some chat-tuned models return {'generated_text': '...'} too
            if "content" in js:
                return js["content"]
        elif isinstance(js, dict):
            # Some models return a dict with 'generated_text' or 'message'
            if "generated_text" in js:
                return js["generated_text"]
            if "message" in js and isinstance(js["message"], dict) and "content" in js["message"]:
                return js["message"]["content"]
        # Fallback: stringify for debugging
        return str(js)

    # If rate-limited or loading, simple retry with backoff
    if resp.status_code in (429, 503):
        time.sleep(2)
        resp2 = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp2.status_code == 200:
            js = resp2.json()
            if isinstance(js, list) and len(js) > 0 and "generated_text" in js:
                return js["generated_text"]
            return str(js)
    # If error
    return f"Error: {resp.status_code} - {resp.text}"

def hf_inference_embed(texts: List[str]) -> np.ndarray:
    """
    Batched embeddings via HF Inference Providers / inference endpoint for embedding models.
    """
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    # The feature-extraction pipeline supports lists as inputs
    payload = {"inputs": texts}
    # Increased timeout duration to 120 seconds
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        arr = np.array(resp.json(), dtype=np.float32)
        # For single item, shape may be (seq, dim); pool mean token embeddings
        if arr.ndim == 2:
            # Single text returns [seq_len, dim], mean-pool
            arr = arr.mean(axis=0, keepdims=True)
        elif arr.ndim == 3:
            # Batch: [batch, seq_len, dim] -> mean-pool per text
            arr = arr.mean(axis=1)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        return arr.astype(np.float32)
    # Simple retry for cold starts
    if resp.status_code in (429, 503):
        time.sleep(2)
        resp2 = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp2.status_code == 200:
            arr = np.array(resp2.json(), dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=0, keepdims=True)
            elif arr.ndim == 3:
                arr = arr.mean(axis=1)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
            return arr.astype(np.float32)
    raise RuntimeError(f"Embedding error {resp.status_code}: {resp.text}")

# =========================
# Vector Index Management
# =========================

class VectorStore:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.index = None
        self.chunks_meta = []  # list of dicts: {"text": str, "source": str, "id": int}

    def load(self):
        ensure_index_dir()
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            # load meta
            self.chunks_meta = []
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    self.chunks_meta.append(json.loads(line))
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine sim with normalized vectors
            self.chunks_meta = []

    def save(self):
        ensure_index_dir()
        if self.index is not None:
            faiss.write_index(self.index, INDEX_FILE)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            for m in self.chunks_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def add_texts(self, chunks: List[Tuple[str, str]]):
        """
        chunks: list of (text, source)
        """
        texts = [c for c in chunks]
        embs = hf_inference_embed(texts)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        # Add to index
        self.index.add(embs)
        # Add meta
        start_id = len(self.chunks_meta)
        for i, (t, src) in enumerate(chunks):
            self.chunks_meta.append({"id": start_id + i, "text": t, "source": src})

    def search(self, query: str, k: int = TOP_K) -> List[dict]:
        q_emb = hf_inference_embed([query])
        D, I = self.index.search(q_emb, k)
        results = []
        for idx, score in zip(I, D):
            if idx < 0 or idx >= len(self.chunks_meta):
                continue
            m = self.chunks_meta[idx]
            results.append({"text": m["text"], "source": m["source"], "score": float(score)})
        return results

# =========================
# RAG Pipeline
# =========================

def build_context(retrieved: List[dict]) -> str:
    ctx = []
    for r in retrieved:
        ctx.append(f"[Source: {r['source']}] {r['text']}")
    return "\n\n".join(ctx)

def answer_question(query: str, vs: VectorStore) -> str:
    if vs.index is None or vs.index.ntotal == 0:
        return "No documents indexed yet. Please upload documents first."
    retrieved = vs.search(query, k=TOP_K)
    context = build_context(retrieved)
    system = (
        "You are a helpful assistant for document Q&A. Use ONLY the provided context. "
        "If the answer is not in the context, say you don't have enough information."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    response = hf_inference_generate(messages, max_new_tokens=512, temperature=0.2, top_p=0.9)
    return response

# =========================
# Ingestion
# =========================

def ingest_files_and_links(files: List[gr.File], links_text: str, vs: VectorStore) -> str:
    new_chunks = []
    # Files
    if files:
        for f in files:
            name = f.name or "uploaded"
            suffix = pathlib.Path(name).suffix.lower()
            with open(f.name, "rb") as rf:
                data = io.BytesIO(rf.read())
            if suffix == ".pdf":
                text = read_pdf(data)
            elif suffix == ".docx":
                text = read_docx(data)
            elif suffix in [".txt", ".text"]:
                text = read_txt(data)
            else:
                # Try text anyway
                try:
                    text = read_txt(data)
                except Exception:
                    text = ""
            if text:
                for ch in split_into_chunks(text):
                    new_chunks.append((ch, name))
    # Links
    if links_text:
        for raw in re.split(r"[,\n ]+", links_text.strip()):
            url = raw.strip()
            if not url:
                continue
            try:
                page_text = fetch_url_to_text(url)
                for ch in split_into_chunks(page_text):
                    new_chunks.append((ch, url))
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")

    if not new_chunks:
        return "No valid content found to ingest."

    vs.add_texts(new_chunks)
    vs.save()
    return f"Ingested {len(new_chunks)} chunks. Index size: {vs.index.ntotal}"

# =========================
# Gradio UI with Retrofuturistic Theme
# =========================

def build_theme():
    # Use Gradio theming + custom CSS
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Orbitron"), gr.themes.GoogleFont("Space Grotesk"), "Arial", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "SF Mono", "Menlo", "monospace"],
    ).set(
        body_background_fill="#0b0f1a",
        body_text_color="#d1e3ff",
        color_accent="#7df9ff",
        color_accent_soft="#213a54",
        block_title_text_color="#7df9ff",
        block_label_text_color="#a3b8ff",
        button_primary_background_fill="#7df9ff",
        button_primary_text_color="#0b0f1a",
        button_primary_background_fill_hover="#9ffcff",
        input_background_fill="#11162a",
        input_border_color="#284a78",
        input_placeholder_color="#7aa2f7",
        block_background_fill="#0e1426",
        block_border_color="#263b63",
        shadow_drop="#0c1a33",
        link_text_color="#9ffcff",
    )
    return theme

CUSTOM_CSS = """
/* Retrofuturistic neon grid styling */
.gradio-container {
  background: radial-gradient(circle at 20% 20%, #10162b 0%, #0b0f1a 50%, #070b14 100%);
}
footer {visibility: hidden}
#title {
  font-family: 'Orbitron', 'Space Grotesk', sans-serif;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 1.8rem;
  color: #7df9ff;
  text-shadow: 0 0 8px rgba(125, 249, 255, 0.7), 0 0 16px rgba(125, 249, 255, 0.35);
}
.neon-card {
  border: 1px solid #263b63;
  border-radius: 14px;
  background: linear-gradient(145deg, rgba(24,36,66,0.6), rgba(9,14,28,0.6));
  box-shadow: 0 0 0 1px #213a54, 0 0 18px rgba(125, 249, 255, 0.08) inset;
}
label, .wrap.svelte-1cll6wc, .wrap.svelte-1j4n0m7 {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: 0.02em;
}
textarea, input, .token-textbox {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-size: 14px;
}
.gr-button-primary {
  border: 0;
  border-radius: 10px;
  box_shadow: 0 0 8px rgba(125, 249, 255, 0.35);
}
a { text-decoration: none; }
"""

def launch_app():
    vs = VectorStore(dim=EMBED_DIM)
    vs.load()

    with gr.Blocks(theme=build_theme(), css=CUSTOM_CSS, title="Neon RAG Q&A") as demo:
        gr.Markdown("# AI Document Q&A", elem_id="title")
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["neon-card"]):
                gr.Markdown("Upload documents and/or paste links, then build the knowledge index.", elem_classes=["neon-card"])
                file_input = gr.Files(
                    label="Upload PDF, DOCX, TXT",
                    file_types=[".pdf", ".docx", ".txt"],
                )
                link_input = gr.Textbox(
                    label="Web links (comma or newline separated)",
                    placeholder="https://example.com, https://another.com/page",
                    lines=3
                )
                ingest_btn = gr.Button("Ingest into Index")
                ingest_status = gr.Markdown("Status: awaiting ingestion...")
            with gr.Column(scale=1, elem_classes=["neon-card"]):
                gr.Markdown("Ask questions answered from your indexed documents.", elem_classes=["neon-card"])
                query_box = gr.Textbox(
                    label="Question",
                    placeholder="Ask something about your documents...",
                    lines=2
                )
                ask_btn = gr.Button("Ask")
                answer_box = gr.Markdown("Answer will appear here.")

        def on_ingest(files, links):
            return ingest_files_and_links(files, links, vs)

        def on_ask(q):
            return answer_question(q, vs)

        ingest_btn.click(on_ingest, inputs=[file_input, link_input], outputs=[ingest_status])
        ask_btn.click(on_ask, inputs=[query_box], outputs=[answer_box])

    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_app()