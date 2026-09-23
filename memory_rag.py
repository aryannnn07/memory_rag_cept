"""
Memory RAG — a Retrieval-Augmented Generation tutor that remembers its users.

Storage (ChromaDB, persisted to disk):
    knowledge : course material, chunked         -> classic RAG
    memories  : long-term facts about each user  -> the "memory" in Memory RAG
Plus an in-process short-term buffer of the last few messages per session.

Every chat turn runs two loops:
    READ  : recall scored memories -> memory-guided query expansion
            -> retrieve knowledge -> generate a personalised, grounded answer
    WRITE : extract durable facts from the exchange
            -> ADD / UPDATE / DELETE long-term memories
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


# --------------------------------------------------------------------------- config

@dataclass
class MemoryConfig:
    short_term_messages: int = 6       # last 3 exchanges kept verbatim per session
    memory_candidates: int = 10        # memories Chroma returns before re-scoring
    memory_top_k: int = 4              # memories that reach the prompt
    knowledge_top_k: int = 3           # course chunks that reach the prompt
    min_relevance: float = 0.25        # ignore memories less similar than this
    duplicate_threshold: float = 0.92  # an ADD this similar to an existing memory is skipped
    w_relevance: float = 1.0           # score = w_rel*relevance + w_rec*recency + w_imp*importance
    w_recency: float = 0.5
    w_importance: float = 0.5
    recency_decay: float = 0.995       # multiplied once per hour since the memory was last used


# --------------------------------------------------------------------------- prompts

ANSWER_PROMPT = """You are a friendly, patient tutor for a Generative AI course.

Use USER MEMORY to personalise your answer: match their level, goals and preferred
style. Weave it in naturally; never list the memories back to the user.
Use COURSE CONTEXT for facts. If the context does not cover the question, say so
in one short sentence, then answer briefly from general knowledge.

USER MEMORY:
{memory}

COURSE CONTEXT:
{context}"""

CLUE_PROMPT = """A student asked: "{question}"

What we remember about this student:
{memory}

Write up to 2 short search queries (3-8 words each) that would find the course
material most useful for THIS student's question. Use the memory only when it
makes the question clearer (for example, "what should I revise" plus "struggles
with attention" -> "self-attention query key value").
Return ONLY a JSON list of strings."""

WRITE_PROMPT = """You manage the long-term memory of an AI tutor.

From the latest exchange, extract durable facts about the USER: name, background,
goals, deadlines, preferences, learning style, topics they find hard or easy,
decisions. Ignore small talk and general facts that are not about the user.

Existing related memories (id: text):
{existing}

Latest exchange:
USER: {user}
ASSISTANT: {assistant}

Return ONLY a JSON list of operations:
  {{"op": "ADD", "text": "...", "importance": 1-10}}               new fact
  {{"op": "UPDATE", "id": "...", "text": "...", "importance": 1-10}} new info changes an existing memory
  {{"op": "DELETE", "id": "..."}}                                   user says it is wrong or asks to forget it
Write each memory as one short third-person sentence ("User prefers ...").
Importance: 1 = trivial, 5 = useful, 10 = core goal, identity or deadline.
Return [] if nothing is worth remembering."""


# --------------------------------------------------------------------------- helpers

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 60) -> list[str]:
    """Split text into overlapping chunks, preferring to cut at sentence ends."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + chunk_size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def parse_json(raw: str, default):
    """Pull the first JSON list/object out of an LLM reply (small models add chatter)."""
    raw = re.sub(r"```(?:json)?", "", raw or "").strip()
    match = re.search(r"(\[.*\]|\{.*\})", raw, re.S)
    if not match:
        return default
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return default


def clamp_importance(value) -> int:
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return 5


# --------------------------------------------------------------------------- engine

class MemoryRAG:
    def __init__(self, db_path: str = "./chroma_db", config: MemoryConfig | None = None,
                 llm_client=None, model: str | None = None, embedding_function=None):
        self.cfg = config or MemoryConfig()
        self.db = chromadb.PersistentClient(path=db_path)
        ef = embedding_function or embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2
        space = {"hnsw:space": "cosine"}  # so distance = 1 - cosine similarity
        self.knowledge = self.db.get_or_create_collection("knowledge", embedding_function=ef, metadata=space)
        self.memories = self.db.get_or_create_collection("memories", embedding_function=ef, metadata=space)
        self.llm = llm_client or OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("LLM_API_KEY", "missing-key"),
        )
        self.model = model or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.short_term: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.cfg.short_term_messages))

    # ---------------------------------------------------------------- LLM
    def _chat(self, messages, temperature=0.3, max_tokens=700) -> str:
        resp = self.llm.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content or ""

    # ---------------------------------------------------------------- knowledge (classic RAG)
    def add_knowledge(self, text: str, source: str = "notes") -> int:
        chunks = chunk_text(text)
        if chunks:
            self.knowledge.add(
                ids=[f"{source}-{uuid.uuid4().hex[:8]}" for _ in chunks],
                documents=chunks,
                metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
            )
        return len(chunks)

    def search_knowledge(self, queries: list[str], k: int) -> list[dict]:
        total = self.knowledge.count()
        if total == 0:
            return []
        res = self.knowledge.query(query_texts=queries, n_results=min(k, total))
        best: dict[str, dict] = {}
        for ids, docs, metas, dists in zip(res["ids"], res["documents"], res["metadatas"], res["distances"]):
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                if cid not in best or dist < best[cid]["distance"]:
                    best[cid] = {"id": cid, "text": doc, "source": meta.get("source"),
                                 "distance": round(dist, 4), "relevance": round(1 - dist, 4)}
        return sorted(best.values(), key=lambda c: c["distance"])[:k]

    # ---------------------------------------------------------------- memory: read path
    def _search_memories(self, user_id: str, text: str, n: int) -> list[dict]:
        total = self.memories.count()
        if total == 0:
            return []
        res = self.memories.query(query_texts=[text], n_results=min(n, total),
                                  where={"user_id": user_id})
        return [{"id": mid, "text": doc, "meta": meta, "relevance": 1 - dist}
                for mid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                                res["metadatas"][0], res["distances"][0])]

    def recall(self, user_id: str, query: str) -> list[dict]:
        """Retrieve memories and re-rank them: relevance + recency + importance."""
        now, cfg = time.time(), self.cfg
        scored = []
        for m in self._search_memories(user_id, query, cfg.memory_candidates):
            if m["relevance"] < cfg.min_relevance:
                continue
            hours = max(0.0, (now - m["meta"]["last_accessed"]) / 3600)
            recency = cfg.recency_decay ** hours
            importance = m["meta"]["importance"] / 10
            score = cfg.w_relevance * m["relevance"] + cfg.w_recency * recency + cfg.w_importance * importance
            scored.append({"id": m["id"], "text": m["text"], "meta": m["meta"],
                           "relevance": round(m["relevance"], 3), "recency": round(recency, 3),
                           "importance": m["meta"]["importance"], "score": round(score, 3)})
        top = sorted(scored, key=lambda m: m["score"], reverse=True)[:cfg.memory_top_k]

        if top:  # using a memory refreshes it, like rehearsal in human memory
            self.memories.update(
                ids=[m["id"] for m in top],
                metadatas=[{**m["meta"], "last_accessed": now,
                            "access_count": m["meta"].get("access_count", 0) + 1} for m in top],
            )
        for m in top:
            m.pop("meta")
        return top

    def _clues(self, question: str, memories: list[dict]) -> list[str]:
        """Memory-guided retrieval: rewrite the search using what we know about the user."""
        memory_txt = "\n".join(f"- {m['text']}" for m in memories)
        raw = self._chat([{"role": "user", "content": CLUE_PROMPT.format(question=question, memory=memory_txt)}],
                         temperature=0.0, max_tokens=120)
        clues = parse_json(raw, [])
        return [str(c).strip() for c in clues if isinstance(c, str) and c.strip()][:2] if isinstance(clues, list) else []

    # ---------------------------------------------------------------- memory: write path
    def remember(self, user_id: str, user_msg: str, assistant_msg: str) -> list[dict]:
        existing = self._search_memories(user_id, user_msg, n=5)
        existing_txt = "\n".join(f"{m['id']}: {m['text']}" for m in existing) or "(none)"
        raw = self._chat([{"role": "user", "content": WRITE_PROMPT.format(
            existing=existing_txt, user=user_msg, assistant=assistant_msg[:1500])}],
            temperature=0.0, max_tokens=400)
        ops = parse_json(raw, [])
        ops = [ops] if isinstance(ops, dict) else ops if isinstance(ops, list) else []

        known = {m["id"]: m for m in existing}
        applied, now = [], time.time()
        for op in ops:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "")).upper()
            text = str(op.get("text", "")).strip()
            mid = op.get("id")
            importance = clamp_importance(op.get("importance", 5))

            if kind == "ADD" and text:
                dup = self._search_memories(user_id, text, n=1)
                if dup and dup[0]["relevance"] >= self.cfg.duplicate_threshold:
                    applied.append({"op": "SKIP", "id": dup[0]["id"], "text": text, "reason": "duplicate"})
                    continue
                mid = f"mem-{uuid.uuid4().hex[:10]}"
                self.memories.add(ids=[mid], documents=[text], metadatas=[{
                    "user_id": user_id, "importance": importance,
                    "created_at": now, "last_accessed": now, "access_count": 0}])
                applied.append({"op": "ADD", "id": mid, "text": text, "importance": importance})

            elif kind == "UPDATE" and mid in known and text:
                meta = {**known[mid]["meta"], "importance": importance, "last_accessed": now, "updated_at": now}
                self.memories.update(ids=[mid], documents=[text], metadatas=[meta])
                applied.append({"op": "UPDATE", "id": mid, "text": text,
                                "old": known[mid]["text"], "importance": importance})

            elif kind == "DELETE" and mid in known:
                self.memories.delete(ids=[mid])
                applied.append({"op": "DELETE", "id": mid, "text": known[mid]["text"]})
        return applied

    # ---------------------------------------------------------------- one full turn
    def chat(self, user_id: str, session_id: str, message: str, mode: str = "memory") -> dict:
        """mode="memory": full Memory RAG.  mode="vanilla": stateless classic RAG, for comparison."""
        t0 = time.time()
        use_memory = mode == "memory"

        memories = self.recall(user_id, message) if use_memory else []
        clues = self._clues(message, memories) if memories else []
        knowledge = self.search_knowledge([message, *clues], self.cfg.knowledge_top_k)

        system = ANSWER_PROMPT.format(
            memory="\n".join(f"- {m['text']}" for m in memories) or "(nothing remembered yet)",
            context="\n\n".join(f"[{i + 1}] ({c['source']}) {c['text']}" for i, c in enumerate(knowledge))
                    or "(no course material found)",
        )
        history = list(self.short_term[session_id]) if use_memory else []
        answer = self._chat([{"role": "system", "content": system}, *history,
                             {"role": "user", "content": message}])

        operations = []
        if use_memory:
            self.short_term[session_id].extend([{"role": "user", "content": message},
                                                {"role": "assistant", "content": answer}])
            operations = self.remember(user_id, message, answer)

        return {"answer": answer, "mode": mode, "memories_used": memories, "clues": clues,
                "knowledge_used": knowledge, "memory_operations": operations,
                "short_term_messages": len(history), "latency_s": round(time.time() - t0, 2)}

    # ---------------------------------------------------------------- memory management
    def list_memories(self, user_id: str) -> list[dict]:
        res = self.memories.get(where={"user_id": user_id})
        items = [{"id": mid, "text": doc, **meta}
                 for mid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(items, key=lambda m: (-m["importance"], -m["created_at"]))

    def forget(self, user_id: str, memory_id: str | None = None) -> int:
        if memory_id:
            if not self.memories.get(ids=[memory_id], where={"user_id": user_id})["ids"]:
                return 0
            self.memories.delete(ids=[memory_id])
            return 1
        count = len(self.memories.get(where={"user_id": user_id})["ids"])
        if count:
            self.memories.delete(where={"user_id": user_id})
        return count

    def prune(self, user_id: str, min_importance: int = 4, unused_days: float = 30) -> list[str]:
        """Forgetting: drop low-importance memories nobody has used for a while."""
        cutoff = time.time() - unused_days * 86400
        stale = [m["id"] for m in self.list_memories(user_id)
                 if m["importance"] < min_importance and m["last_accessed"] < cutoff]
        if stale:
            self.memories.delete(ids=stale)
        return stale

    def reset_session(self, session_id: str) -> None:
        self.short_term.pop(session_id, None)
