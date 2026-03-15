# MIT App Inventor Chatbot Trainer — GSoC 2026 Prototype

This is a thin, demo-ready prototype: “PIC/PAC for chatbots.” Users upload their own knowledge, test retrieval-augmented answers, and export a bundle that an App Inventor extension can load. Everything runs offline (no API keys); the responder is a local template we can swap for a small SLM/LLM later.

## What it demonstrates
- RAG loop: ingest .txt/.md, chunk, embed (MiniLM), search with FAISS, respond with a local template using retrieved chunks.
- Bundle export: FAISS index + chunk metadata + manifest (models, chunking, top-k) for an App Inventor extension loader.
- Minimal API: FastAPI `/query` to drive a future web UI or extension tests.

## Quickstart (maintainer demo)
1) Setup (Python 3.10+):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r prototype/requirements.txt
   ```
2) Ingest sample docs and build an index:
   ```bash
   python prototype/trainer.py ingest --data prototype/data
   ```
3) Ask a question via CLI:
   ```bash
   python prototype/trainer.py query --question "How does the bundle work?"
   ```
4) Run the API server (optional UI/extension hook):
   ```bash
   uvicorn prototype.trainer:app --reload
   # POST {"question": "..."} to http://localhost:8000/query
   ```
5) Export a bundle (artifacts for an App Inventor extension):
   ```bash
   python prototype/trainer.py export --bundle-path prototype/dist/demo-bundle
   ```

## Sample run (CLI)
![Sample query result](Screenshot%202026-03-15%20at%209.20.26%E2%80%AFPM.png)

## Repo map
- prototype/trainer.py — ingest, retrieve, local answer stub, bundle export, FastAPI.
- prototype/requirements.txt — minimal deps (FAISS, sentence-transformers, FastAPI).
- prototype/data/ — sample docs for demo ingest.
- docs/prototype-walkthrough.md — demo script and talking points.
