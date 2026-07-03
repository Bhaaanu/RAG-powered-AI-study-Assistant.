# RAG-Powered AI Study Assistant

A Retrieval-Augmented Generation (RAG) chatbot that ingests course PDFs and
notes, then answers student questions with cited source passages — built to
minimize hallucinated answers.

**Stack:** LangChain · OpenAI API · ChromaDB · FastAPI · Streamlit

## Features

- Ingests course PDFs/notes and splits them into overlapping chunks
- Embeds chunks with OpenAI embeddings and stores them in a persistent
  ChromaDB vector store
- Retrieves the most relevant passages for a student's query
- Grounds LLM answers in retrieved context, returning inline citations
  (`[Source N]`) with file name, page number, and excerpt
- FastAPI backend (`/ingest`, `/ask`, `/health`) + Streamlit chat UI

## Project Structure

```
rag_study_assistant/
├── backend/
│   ├── main.py         # FastAPI app (ingest, ask, health endpoints)
│   ├── rag_engine.py    # Chunking, embedding, retrieval, grounded generation
│   └── ingest.py         # CLI ingestion script
├── frontend/
│   └── app.py            # Streamlit chat UI
├── data/
│   ├── course_pdfs/       # Drop your course PDFs here
│   └── chroma_db/         # Persistent vector store (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

Add course PDFs to `data/course_pdfs/`.

## Run

```bash
# Terminal 1 — backend
cd backend
uvicorn main:app --reload

# Terminal 2 — frontend
cd frontend
streamlit run app.py
```

Then open the Streamlit URL, click **Re-index course materials**, and start
asking questions.

## How it Works

1. **Chunking** — `RecursiveCharacterTextSplitter` splits PDFs into
   ~1000-character overlapping chunks to preserve context across boundaries.
2. **Embedding** — Each chunk is embedded with OpenAI's
   `text-embedding-3-small` and stored in ChromaDB with source/page metadata.
3. **Retrieval** — On a query, the top-k most similar chunks are retrieved
   via vector similarity search.
4. **Prompt grounding** — Retrieved chunks are inserted into a strict prompt
   template instructing the LLM to answer only from context and to cite
   sources by passage number, reducing hallucination.
5. **Citation surfacing** — The API returns structured source metadata so
   the UI can show exactly which document/page each part of the answer
   came from.
