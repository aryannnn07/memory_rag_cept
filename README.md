# Memory RAG — a tutor that remembers

A Retrieval-Augmented Generation system with **long-term user memory**, built on **ChromaDB**.
Classic RAG answers every question as if from a stranger. Memory RAG remembers who you are,
what you find hard and what you're working towards, and uses that to personalise both the
answer **and the search itself**.

<img width="954" height="398" alt="Screenshot 2026-09-23 at 11 23 34 PM" src="https://github.com/user-attachments/assets/78044b92-efa4-40e4-81a1-7ca4ef5dc940" />



## What's inside

| File | Purpose |
|---|---|
| `memory_rag.py` | The engine: two ChromaDB collections, scored recall, memory-guided retrieval, ADD/UPDATE/DELETE memory writes, forgetting |
| `main.py` | FastAPI server: chat, knowledge upload, memory inspection and deletion |
| `static/index.html` | Demo UI: chat on the left, live memory inspector on the right |
| `eval.py` | Scripted multi-session test comparing Memory RAG with Vanilla RAG |
| `data/course_notes.md` | Knowledge base (GenAI course notes), loaded on first start |

## Setup (macOS)

```bash
cd memory-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Groq key in .env (or switch to Ollama)
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** for the demo UI and **http://127.0.0.1:8000/docs** for the API docs.

The first start downloads ChromaDB's embedding model (all-MiniLM-L6-v2, about 80 MB), so you need
internet once. After that, with the Ollama option in `.env`, the whole system runs offline.

## Live demo script (about 3 minutes)

1. In **Memory RAG** mode, send three messages:
   - "Hi, I'm Aryan, a third-year CS student."
   - "I'm struggling with query, key and value in attention."
   - "My exam is on 5th October and I like simple analogies."

   Watch the inspector show `ADD` operations with importance scores.
2. Click **New session** (short-term memory is wiped, long-term memory stays).
3. Ask: **"What should I revise?"** The inspector shows the recalled memories with their
   relevance, recency and importance bars, and the search clues the memories produced.
   The answer targets attention and uses an analogy.
4. Switch to **Vanilla RAG** and ask the same question. You get a generic answer, because
   vanilla RAG has no memory.
5. Say: **"My exam moved to 12th October."** You'll see an `UPDATE` with the old value struck through.
6. Say: **"Please forget my exam date."** You'll see a `DELETE`. Then show **Forget me** for full deletion.
7. Stop the server with Ctrl+C, restart it, and ask again. The memories persist, because ChromaDB writes to disk.

## API

| Method | Endpoint | What it does |
|---|---|---|
| POST | `/chat` | One turn. Body: `user_id`, `session_id`, `message`, `mode` (`memory` or `vanilla`) |
| POST | `/sessions/{id}/reset` | Clear short-term memory only |
| POST | `/knowledge` | Add text to the knowledge base |
| POST | `/knowledge/upload` | Upload a `.txt`, `.md` or `.pdf` file |
| GET | `/knowledge/stats` | Chunk and memory counts |
| GET | `/memories/{user_id}` | Everything remembered about a user |
| DELETE | `/memories/{user_id}/{memory_id}` | Delete one memory |
| DELETE | `/memories/{user_id}` | Forget the user completely |
| POST | `/memories/{user_id}/prune` | Forgetting policy: drop unused low-importance memories |

## How recall is scored

```
score = 1.0 × relevance + 0.5 × recency + 0.5 × (importance / 10)
relevance = 1 − cosine distance (from ChromaDB)
recency   = 0.995 ^ hours since the memory was last used
```

This is inspired by the retrieval function in *Generative Agents* (Park et al., 2023). You can
tune the weights in `MemoryConfig`.

## Evaluation

```bash
python eval.py
```

This teaches the tutor four facts in separate sessions, then checks recall, personalised
retrieval, updating, forgetting and user isolation, and compares the results with Vanilla RAG.
It uses its own database folder (`chroma_eval/`), so your demo data isn't touched.

## Troubleshooting

- **"LLM call failed"**: check `.env`. For Groq, check the key. For Ollama, make sure the app is running.
- **Memories not being saved with Ollama**: small models sometimes return malformed JSON. The
  parser tolerates extra text, but `qwen2.5:3b` works better than `qwen2.5-coder:3b` here.
  Groq's `llama-3.1-8b-instant` is the most reliable choice.
- **Mac feels slow**: with 8 GB of RAM, close Chrome tabs before running Ollama.
- **Start fresh**: stop the server and delete the `chroma_db/` folder.
