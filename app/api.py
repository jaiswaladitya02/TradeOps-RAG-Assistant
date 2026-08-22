from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import TradeOpsRAG


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TradeOps RAG Assistant API",
    description="API for the TradeOps RAG Assistant",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================
#
# Frontend:
#   http://127.0.0.1:5500
#
# Also allow:
#   http://localhost:5500
#
# This is required because frontend and backend use
# different ports.
#

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL RAG INSTANCE
# ============================================================
#
# Load the embedding model and ChromaDB once when the API
# starts instead of loading them for every request.
#

print("=" * 70)
print("TRADEOPS RAG API")
print("=" * 70)

print("\nInitializing TradeOps RAG system...\n")

try:

    rag = TradeOpsRAG()

    print("\nTradeOps RAG system initialized successfully!")

except Exception as error:

    rag = None

    print("\nERROR INITIALIZING RAG SYSTEM:")
    print(error)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    question: str

    top_k: int = 5


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "TradeOps RAG Assistant API",
        "version": "1.0.0",
        "status": "running",
        "health": "/health",
        "documentation": "/docs"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    if rag is None:

        return {
            "status": "error",
            "rag_initialized": False
        }

    try:

        document_count = rag.collection.count()

    except Exception:

        document_count = None

    return {
        "status": "healthy",
        "rag_initialized": True,
        "documents": document_count
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask(request: AskRequest):

    if rag is None:

        raise HTTPException(
            status_code=500,
            detail="TradeOps RAG system is not initialized."
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Protect the backend from unreasonable values.

    top_k = max(
        1,
        min(request.top_k, 10)
    )

    try:

        answer, sources = rag.ask(
            query=question,
            top_k=top_k
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except Exception as error:

        print("\nERROR PROCESSING QUESTION:")
        print(error)

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# QUERY ALIAS
# ============================================================
#
# Some frontend versions may call /query instead of /ask.
# Keep both available.
#

@app.post("/query")
def query(request: AskRequest):

    return ask(request)


# ============================================================
# CHAT ALIAS
# ============================================================
#
# Also provide /chat so the frontend can use a conventional
# chat endpoint if required.
#

@app.post("/chat")
def chat(request: AskRequest):

    return ask(request)


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
def startup_event():

    print("\n" + "=" * 70)
    print("TRADEOPS RAG API READY")
    print("=" * 70)

    print("\nAPI:")
    print("http://127.0.0.1:8000")

    print("\nSwagger documentation:")
    print("http://127.0.0.1:8000/docs")

    print("\nHealth:")
    print("http://127.0.0.1:8000/health")

    print("\nAsk:")
    print("POST http://127.0.0.1:8000/ask")

    print("\n" + "=" * 70)