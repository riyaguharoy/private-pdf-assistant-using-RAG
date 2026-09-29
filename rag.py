"""RAG core: read PDFs, chunk, embed with Ollama, store/query in ChromaDB."""
import hashlib
import chromadb
import ollama
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"          # try "llama3.2:1b" for faster CPU answers
CHUNK_SIZE = 1000                # characters per chunk
CHUNK_OVERLAP = 200
TOP_K = 4

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pdfs", metadata={"hnsw:space": "cosine"})


def read_pdf(path):
    """Return a list of (page_number, text) tuples."""
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())  # normalise whitespace
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts,
                        options={"num_gpu": 0})["embeddings"]


def ingest_pdf(path, filename):
    """Chunk a PDF and add it to the vector store. Returns number of chunks."""
    ids, docs, metas = [], [], []
    for page_no, page_text in read_pdf(path):
        for j, chunk in enumerate(chunk_text(page_text)):
            ids.append(hashlib.md5(f"{filename}-{page_no}-{j}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({"source": filename, "page": page_no})
    if not docs:
        return 0
    # Embed in small batches so CPU-only machines don't stall
    for i in range(0, len(docs), 16):
        collection.upsert(
            ids=ids[i:i + 16],
            documents=docs[i:i + 16],
            metadatas=metas[i:i + 16],
            embeddings=embed(docs[i:i + 16]),
        )
    return len(docs)


def list_documents():
    metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def delete_document(filename):
    collection.delete(where={"source": filename})


def answer(question, source=None):
    """Answer a question. If source is a filename, search only that PDF."""
    if collection.count() == 0:
        return {"answer": "No documents yet. Upload a PDF first.", "sources": []}

    query = {"query_embeddings": embed([question]), "n_results": TOP_K}
    if source:
        query["where"] = {"source": source}
    results = collection.query(**query)
    if not results["documents"][0]:
        return {"answer": f"Nothing found in {source}. Try uploading it again.", "sources": []}
    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"[{m['source']}, page {m['page']}]\n{c}" for c, m in zip(chunks, metas)
    )
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 4096, "temperature": 0.2, "num_gpu": 0},
    )
    sources = [
        {"source": m["source"], "page": m["page"], "text": c[:300]}
        for c, m in zip(chunks, metas)
    ]
    return {"answer": response["message"]["content"], "sources": sources}
