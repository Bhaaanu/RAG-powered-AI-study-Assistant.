# 📚 RAG Study Assistant — AI-Powered Course Q&A

> An AI study companion that reads your course PDFs and notes, then answers questions using **Retrieval-Augmented Generation (RAG)** — with every answer backed by a cited source passage.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-orange?logo=openai)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?logo=fastapi)

---

## 🧠 What Makes This Useful for Students?

Generic chatbots can sound confident while being completely wrong — a real risk when studying for an exam. RAG Study Assistant fixes that. It:

1. **Reads** your uploaded course PDFs and notes
2. **Retrieves** the most relevant passages for your question via semantic search
3. **Grounds** its answer strictly in that retrieved context — never in guesswork
4. **Cites** the exact file, page number, and excerpt behind every claim

The result: answers you can actually verify, in seconds, instead of skimming an entire chapter to check if the AI got it right.

---

## 🏗️ Architecture

```
Student Question
    │
    ▼
FastAPI /ask endpoint
    │
    ├── Embed query (OpenAI text-embedding-3-small)
    │
    ▼
ChromaDB Vector Store (persistent)
    │
    ├── Top-k similarity search → most relevant chunks
    │
    ▼
Grounded Prompt Template (LangChain)
    │
    ├── LLM answers ONLY from retrieved context
    │
    ▼
Cited Answer → [Source N: file, page, excerpt]
    │
    ▼
Streamlit Chat UI
```

**Ingestion pipeline (runs once per document, or on re-index):**
```
Course PDFs/Notes → RecursiveCharacterTextSplitter (~1000 char chunks, overlapping)
                            │
                            ▼
                  OpenAI Embeddings → ChromaDB (+ source/page metadata)
```

---

## 📁 Project Structure

```
rag_study_assistant/
├── backend/
│   ├── main.py          # FastAPI app (ingest, ask, health endpoints)
│   ├── rag_engine.py    # Chunking, embedding, retrieval, grounded generation
│   └── ingest.py        # CLI ingestion script
├── frontend/
│   └── app.py            # Streamlit chat UI
├── data/
│   ├── course_pdfs/      # Drop your course PDFs here (gitignored)
│   └── chroma_db/         # Persistent vector store (auto-created, gitignored)
├── .env.example           # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-study-assistant.git
cd rag-study-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 5. Add your course materials
Place PDFs and notes into `data/course_pdfs/`.

### 6. Run the app

**Terminal 1 — backend:**
```bash
cd backend
uvicorn main:app --reload
```

**Terminal 2 — frontend:**
```bash
cd frontend
streamlit run app.py
```

Open your browser at the Streamlit URL shown in the terminal.

---

## 💡 How to Use

1. **Drop** your course PDFs and notes into `data/course_pdfs/`
2. Click **🔄 Re-index course materials** in the Streamlit sidebar — this builds/updates the knowledge base
3. **Ask questions** in the chat, for example:
   - *"What are the key assumptions behind this economic model?"*
   - *"Summarize chapter 4 in three sentences"*
   - *"What did the lecture notes say about mitosis vs meiosis?"*
   - *"Where is backpropagation defined in these materials?"*
4. Check the **inline citations** under each answer to jump straight to the source file and page

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangChain |
| **LLM** | OpenAI API |
| **Embeddings** | text-embedding-3-small (OpenAI) |
| **Vector Store** | ChromaDB (persistent) |
| **Backend API** | FastAPI |
| **UI** | Streamlit |
| **Language** | Python 3.10+ |

---

## 📡 API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Index new or updated course materials |
| `POST /ask` | Submit a question and receive a grounded, cited answer |
| `GET /health` | Check that the backend service is running |

---

## 🔮 Future Enhancements

- [ ] Multi-course workspaces for students juggling several subjects
- [ ] Support for lecture recordings and transcripts
- [ ] Answer confidence scoring based on retrieval quality
- [ ] Export cited answers as study notes (PDF/Word)
- [ ] Flashcard generation from indexed materials

---

## 👩‍💻 Author

**Your Name**
Your Program / Institution
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License — free to use and modify for educational and personal projects.
