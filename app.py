"""Flask web app for chatting with your PDFs locally."""
import os
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import rag

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/documents")
def documents():
    return jsonify(rag.list_documents())


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Choose a PDF file."}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)
    try:
        n = rag.ingest_pdf(path, filename)
    except Exception as e:
        return jsonify({"error": f"Could not process the PDF: {e}"}), 500
    if n == 0:
        return jsonify({"error": "No text found. Scanned PDFs need OCR first."}), 400
    return jsonify({"filename": filename, "chunks": n})


@app.delete("/documents/<name>")
def delete(name):
    rag.delete_document(name)
    return jsonify({"deleted": name})


@app.post("/ask")
def ask():
    body = request.get_json() or {}
    question = body.get("question", "").strip()
    source = body.get("source") or None  # None means search all PDFs
    if not question:
        return jsonify({"error": "Type a question."}), 400
    try:
        return jsonify(rag.answer(question, source))
    except Exception as e:
        return jsonify({"error": f"Ollama error: {e}. Is Ollama running?"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
