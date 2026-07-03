"""
rag_engine.py
--------------
Core Retrieval-Augmented Generation logic for the AI Study Assistant.

Responsibilities:
  1. Ingest course PDFs / notes and split them into overlapping chunks.
  2. Embed chunks and store them in a persistent ChromaDB vector store.
  3. Retrieve the most relevant chunks for a student query.
  4. Ground the LLM's answer in retrieved passages and return citations,
     reducing hallucinated answers.
"""

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
PDF_DIR = os.getenv("PDF_DIR", "data/course_pdfs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")

GROUNDED_PROMPT = ChatPromptTemplate.from_template(
    """You are a study assistant. Answer the student's question using ONLY
the context passages below. If the answer is not contained in the context,
say you don't have enough information from the course materials.

Cite each fact using the passage number it came from, e.g. [Source 2].

Context:
{context}

Question: {question}

Answer (with inline [Source N] citations):"""
)


@dataclass
class RetrievedChunk:
    source: str
    page: int
    text: str


class RAGEngine:
    """Wraps ingestion, retrieval, and grounded generation."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        self.vectorstore = Chroma(
            collection_name="course_materials",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ---------------------------------------------------------------
    # Ingestion
    # ---------------------------------------------------------------
    def ingest_directory(self, directory: str = PDF_DIR) -> int:
        """Load every PDF in `directory`, chunk it, embed it, and persist
        it to ChromaDB. Returns the number of chunks written."""
        loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        raw_docs: List[Document] = loader.load()
        chunks = self.splitter.split_documents(raw_docs)

        # Give every chunk a stable, human-readable citation label.
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata.setdefault("source", chunk.metadata.get("source", "unknown"))

        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        return len(chunks)

    # ---------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------
    def retrieve(self, query: str, k: int = TOP_K) -> List[RetrievedChunk]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [
            RetrievedChunk(
                source=os.path.basename(doc.metadata.get("source", "unknown")),
                page=doc.metadata.get("page", 0) + 1,
                text=doc.page_content,
            )
            for doc in results
        ]

    # ---------------------------------------------------------------
    # Grounded generation
    # ---------------------------------------------------------------
    def answer(self, question: str, k: int = TOP_K) -> dict:
        chunks = self.retrieve(question, k=k)

        if not chunks:
            return {
                "answer": "I couldn't find anything relevant in the course materials.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[Source {i+1}] ({c.source}, p.{c.page})\n{c.text}"
            for i, c in enumerate(chunks)
        )

        prompt = GROUNDED_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [
                {"label": f"Source {i+1}", "file": c.source, "page": c.page, "excerpt": c.text[:300]}
                for i, c in enumerate(chunks)
            ],
        }
