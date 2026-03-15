# Chatbot Trainer FAQ (sample content)

## What is this prototype?
A small retrieval-augmented chatbot trainer inspired by PIC/PAC. It ingests user text, builds an embedding index, and serves answers.

## What artifacts get exported?
The prototype exports a FAISS vector index, chunk metadata (text + source), and a manifest describing the embedding model and retrieval settings.

## How does an App Inventor app use this?
An extension would load the bundle, embed queries locally or via a service, retrieve top chunks, assemble a prompt, and generate a reply with a small language model.

## Is this production-ready?
No. It is a starter and a baseline to iterate on UI, safety, and App Inventor integration.
