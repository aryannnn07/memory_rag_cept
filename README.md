# Memory RAG — a tutor that remembers

A Retrieval-Augmented Generation system with **long-term user memory**, built on **ChromaDB**.
Classic RAG answers every question as if from a stranger. Memory RAG remembers who you are,
what you find hard and what you're working towards, and uses that to personalise both the
answer **and the search itself**.

Workflow Screenshots:
1. The FastAPI backend, with interactive Swagger documentation generated automatically from the request/response models - every endpoint here can be tested directly from the browser:
<img width="1453" height="848" alt="Screenshot 2026-09-24 at 9 48 48 AM" src="https://github.com/user-attachments/assets/fbb67bf0-65ed-40ae-bb03-5fe37db45548" />

2. The knowledge base auto-loads on first start — this endpoint confirms the course notes were chunked and embedded into ChromaDB before any question was ever asked:
<img width="1445" height="782" alt="Screenshot 2026-09-24 at 9 49 49 AM" src="https://github.com/user-attachments/assets/2a57c86b-ab09-4bdd-856a-a0058c670f97" />

3.After one message, the write path extracts distinct, durable facts about the student - name, weak topic, and deadline - and stores each as its own memory, not the raw transcript. It also shows the importance of each fact.
<img width="1462" height="801" alt="Screenshot 2026-09-24 at 9 51 29 AM" src="https://github.com/user-attachments/assets/223748fe-cc74-47a5-89be-fe49277ff90d" />

4. After I start a new session, the short term memory gets deleted and the long term memory is preserved. After I ask the model what should I revise without mentioning anything, it answers correctly based on its memory of its date. The final answer is grounded in course material but shaped by what the tutor remembers about this specific student, rather than a generic response anyone would receive:
<img width="1470" height="815" alt="Screenshot 2026-09-24 at 9 53 43 AM" src="https://github.com/user-attachments/assets/12733f9d-cb7c-463c-add7-802165fa391d" />

5. The identical question, identical knowledge base and identical model - but with memory switched off. Vanilla RAG treats every question as if from a stranger and answers generically; Memory RAG personalises the same question using what it knows about the student:
<img width="1469" height="803" alt="Screenshot 2026-09-24 at 9 59 25 AM" src="https://github.com/user-attachments/assets/2c72dbdc-14a3-451f-a736-78f35403210f" />

Some More Working Features:
a. Updates facts instead of duplicating them — when something changes (like an exam date), the system corrects the existing memory rather than storing two conflicting versions.
b. Survives a full server restart — memories are written to disk via ChromaDB, so stopping and restarting the server doesn't lose anything, unlike keeping history only in RAM.
c. Forgets things on request — a user can ask in plain conversation to stop remembering something specific, and it's actually deleted, not just ignored.
d. Recognises duplicate information — repeating a fact it already knows gets skipped instead of cluttering memory with the same thing stored twice.
e. Supports manual memory management — beyond conversation, individual memories can be deleted directly by the user through the interface.
f. Keeps memories private per user — one person's stored facts are never visible to or retrievable by another user.
g. Offers a full "forget me" option — a user can wipe every memory stored about them in one action, not just individual facts.
h. Lets the knowledge base grow over time — new documents can be uploaded and indexed at any point without restarting the server.
i. Automatically forgets stale, low-value memories — a prune policy can clear out unused, low-importance memories after a set time, so memory doesn't grow forever.


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
