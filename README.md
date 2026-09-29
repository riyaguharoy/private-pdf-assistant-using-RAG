# Private PDF Assistant

**Retrieval-Augmented Generation on own machine.** Chat with PDFs fully offline: no internet, no API keys, and no data leaves the computer.

Private PDF Assistant is a local Retrieval-Augmented Generation (RAG) app for asking questions about PDF documents. Uploaded PDFs are split into overlapping chunks, embedded with `nomic-embed-text`, and stored in a persistent ChromaDB vector store. When I ask a question, the most relevant passages are retrieved and passed to Llama 3.2 through Ollama, which answers using only that context. Each answer shows its sources with file name and page number, so every claim can be checked against the original document.

The app is designed to run on modest hardware, including CPU-only laptops.

screenshot.png

## Features

- Upload PDFs by button or drag and drop
- Ask about one PDF or search across all of them
- Answers grounded in documents, with page-level sources
- Runs entirely locally with Ollama; works on CPU
- Light and dark mode

## Tech stack

Python · Flask · ChromaDB · Ollama (Llama 3.2, nomic-embed-text) · pypdf · HTML/CSS/JS

## How it works

1. **Ingest:** text is extracted from each PDF page with pypdf and split into 1,000-character chunks with 200-character overlap.
2. **Embed:** each chunk is embedded with `nomic-embed-text` via Ollama and stored in ChromaDB with its file name and page number.
3. **Retrieve:** a question is embedded the same way, and the 4 most similar chunks are fetched (optionally filtered to one PDF).
4. **Generate:** Llama 3.2 answers using only the retrieved context and says when the answer isn't there.

## Getting started

### Prerequisites

- Python 3.10–3.12
- [Ollama](https://ollama.com/download)

### Setup

```bash
# 1. Pull the models
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Clone and install
git clone https://github.com/<your-username>/private-pdf-assistant.git
cd private-pdf-assistant
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt

# 3. Run
python app.py
```

Open http://localhost:5000, upload a PDF, and ask a question.

### Configuration

Settings are at the top of `rag.py`:

| Setting | Default | Notes |
|---|---|---|
| `CHAT_MODEL` | `llama3.2` | Use `llama3.2:1b` for faster answers on CPU |
| `EMBED_MODEL` | `nomic-embed-text` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | Characters |
| `TOP_K` | 4 | Chunks retrieved per question |

The app runs models on CPU (`num_gpu: 0`) for compatibility with older GPUs. Remove that option in `rag.py` to use your GPU.

## Project structure

```
├── app.py              # Flask routes: upload, ask, list, delete
├── rag.py              # Chunking, embeddings, retrieval, generation
├── requirements.txt
└── templates/
    └── index.html      # Web interface
```

## Limitations

- Scanned PDFs (images of text) need OCR before they can be used.
- Answer quality depends on the local model; small models can miss details in long documents.
