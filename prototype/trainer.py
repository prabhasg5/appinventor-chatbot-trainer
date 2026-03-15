import argparse
import json
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEN_MODEL = "local-slm-placeholder"
SYSTEM_PROMPT = (
    "You are an App Inventor chatbot. Answer concisely using only the provided context. "
    "If the context is missing the answer, say you do not know."
)

_embedder = None
_index = None
_metadata: Dict[str, List[Dict[str, str]]] | None = None


def load_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def chunk_text(text: str, chunk_size: int = 480, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def read_documents(path: Path) -> List[Tuple[str, str]]:
    files = []
    if path.is_file():
        files = [path]
    else:
        files = sorted([p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".md"}])
    documents: List[Tuple[str, str]] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        documents.append((file.name, text))
    if not documents:
        raise ValueError("No .txt or .md files found to ingest.")
    return documents


def build_index(data_path: Path, chunk_size: int, overlap: int) -> None:
    documents = read_documents(data_path)
    embedder = load_embedder()

    chunks: List[Dict[str, str]] = []
    for source, text in documents:
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append({"text": chunk, "source": source})

    embeddings = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(json.dumps({"chunks": chunks}, indent=2))
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "embedding_model": DEFAULT_EMBED_MODEL,
                "generator_model": DEFAULT_GEN_MODEL,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k_default": 4,
            },
            indent=2,
        )
    )
    print(f"Ingested {len(chunks)} chunks from {len(documents)} files. Index saved to {INDEX_PATH}.")


def load_artifacts() -> None:
    global _index, _metadata
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Artifacts not found. Run ingest first.")
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    if _metadata is None:
        _metadata = json.loads(METADATA_PATH.read_text())


def search(query: str, top_k: int = 4) -> List[Dict[str, str]]:
    load_artifacts()
    embedder = load_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = _index.search(query_vec, top_k)
    chunks = _metadata["chunks"]
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[int(idx)]
        results.append({"text": chunk["text"], "source": chunk["source"], "score": float(score)})
    return results


def build_prompt(question: str, contexts: List[Dict[str, str]]) -> str:
    context_block = "\n".join(f"- ({c['score']:.2f}) {c['text']}" for c in contexts)
    return textwrap.dedent(
        f"""
        {SYSTEM_PROMPT}
        Context:
        {context_block}

        User question: {question}
        """
    ).strip()


def generate_answer(question: str, contexts: List[Dict[str, str]]) -> str:
    if not contexts:
        return "I do not have enough information to answer that yet."
    best = contexts[0]
    summary = best["text"][:220]
    return f"Based on the project notes: {summary}"


def ingest_command(args: argparse.Namespace) -> None:
    build_index(Path(args.data), chunk_size=args.chunk_size, overlap=args.overlap)


def query_command(args: argparse.Namespace) -> None:
    contexts = search(args.question, top_k=args.top_k)
    answer = generate_answer(args.question, contexts)
    print("Answer:\n", answer)
    print("\nTop contexts:")
    for ctx in contexts:
        print(f"- {ctx['source']} (score {ctx['score']:.2f}): {ctx['text'][:160]}...")


def export_command(args: argparse.Namespace) -> None:
    load_artifacts()
    dest = Path(args.bundle_path)
    dest.mkdir(parents=True, exist_ok=True)
    for file in [INDEX_PATH, METADATA_PATH, MANIFEST_PATH]:
        shutil.copy(file, dest / file.name)
    archive_path = shutil.make_archive(str(dest), "zip", root_dir=dest)
    print(f"Bundle copied to {dest} and zipped at {archive_path}.")


def serve_command(args: argparse.Namespace) -> None:
    uvicorn.run("prototype.trainer:app", host=args.host, port=args.port, reload=args.reload)


# FastAPI surface
app = FastAPI(title="App Inventor Chatbot Trainer Prototype", version="0.1")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query_endpoint(payload: QueryRequest) -> Dict[str, object]:
    contexts = search(payload.question, top_k=payload.top_k)
    answer = generate_answer(payload.question, contexts)
    return {"answer": answer, "contexts": contexts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="App Inventor chatbot trainer prototype")
    sub = parser.add_subparsers(dest="command", required=True)
#ingest
    ingest = sub.add_parser("ingest", help="Build the embedding index from text files")
    ingest.add_argument("--data", required=True, help="Path to a file or directory of .txt/.md docs")
    ingest.add_argument("--chunk-size", type=int, default=480)
    ingest.add_argument("--overlap", type=int, default=64)
    ingest.set_defaults(func=ingest_command)
#query
    query = sub.add_parser("query", help="Ask a question against the built index")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=4)
    query.set_defaults(func=query_command)
#export
    export = sub.add_parser("export", help="Copy artifacts into a bundle directory and zip it")
    export.add_argument("--bundle-path", required=True, help="Destination directory for the bundle")
    export.set_defaults(func=export_command)
#serve
    serve = sub.add_parser("serve", help="Run the FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=serve_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
if __name__ =="__main__":
    main()
