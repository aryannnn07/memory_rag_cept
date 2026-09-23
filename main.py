"""
FastAPI server for Memory RAG.

Run:   uvicorn main:app --reload
UI:    http://127.0.0.1:8000
Docs:  http://127.0.0.1:8000/docs
"""
import io
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from memory_rag import MemoryRAG  # noqa: E402

BASE = Path(__file__).parent
app = FastAPI(
    title="Memory RAG — a tutor that remembers",
    description="Retrieval-Augmented Generation with long-term user memory, built on ChromaDB.",
    version="1.0.0",
)
rag = MemoryRAG(db_path=os.getenv("CHROMA_PATH", str(BASE / "chroma_db")))

# Load the course notes the first time the server starts.
if rag.knowledge.count() == 0 and (BASE / "data" / "course_notes.md").exists():
    rag.add_knowledge((BASE / "data" / "course_notes.md").read_text(), source="course_notes")


class ChatIn(BaseModel):
    user_id: str = Field("aryan", description="Whose long-term memory to use")
    session_id: str = Field("session-1", description="Short-term memory lives per session")
    message: str
    mode: Literal["memory", "vanilla"] = "memory"


class KnowledgeIn(BaseModel):
    text: str
    source: str = "manual"


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/chat", tags=["chat"])
def chat(body: ChatIn):
    """One turn: recall memories, retrieve knowledge, answer, then update memory."""
    try:
        return rag.chat(body.user_id, body.session_id, body.message, body.mode)
    except Exception as exc:  # LLM/network errors surface clearly in the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")


@app.post("/sessions/{session_id}/reset", tags=["chat"])
def reset_session(session_id: str):
    """Clear short-term memory only. Long-term memories survive — that's the point."""
    rag.reset_session(session_id)
    return {"status": "short-term memory cleared", "session_id": session_id}


@app.post("/knowledge", tags=["knowledge"])
def add_knowledge(body: KnowledgeIn):
    return {"chunks_added": rag.add_knowledge(body.text, body.source)}


@app.post("/knowledge/upload", tags=["knowledge"])
async def upload_knowledge(file: UploadFile = File(...), source: str = Form(None)):
    """Upload a .txt, .md or .pdf file into the knowledge collection."""
    raw = await file.read()
    name = (file.filename or "upload").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    elif name.endswith((".txt", ".md")):
        text = raw.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(400, "Upload a .txt, .md or .pdf file")
    return {"file": file.filename, "chunks_added": rag.add_knowledge(text, source or file.filename)}


@app.get("/knowledge/stats", tags=["knowledge"])
def knowledge_stats():
    return {"chunks": rag.knowledge.count(), "memories_total": rag.memories.count()}


@app.get("/memories/{user_id}", tags=["memory"])
def list_memories(user_id: str):
    """Everything the tutor remembers about this user (transparency)."""
    return rag.list_memories(user_id)


@app.delete("/memories/{user_id}/{memory_id}", tags=["memory"])
def delete_memory(user_id: str, memory_id: str):
    if not rag.forget(user_id, memory_id):
        raise HTTPException(404, "Memory not found for this user")
    return {"deleted": memory_id}


@app.delete("/memories/{user_id}", tags=["memory"])
def forget_user(user_id: str):
    """'Forget me' — delete every long-term memory for a user."""
    return {"deleted": rag.forget(user_id)}


@app.post("/memories/{user_id}/prune", tags=["memory"])
def prune(user_id: str, min_importance: int = 4, unused_days: float = 30):
    """Forgetting policy: remove low-importance memories unused for `unused_days`."""
    return {"pruned": rag.prune(user_id, min_importance, unused_days)}
