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
a. Updates facts instead of duplicating them - when something changes (like an exam date), the system corrects the existing memory rather than storing two conflicting versions.
b. Survives a full server restart - memories are written to disk via ChromaDB, so stopping and restarting the server doesn't lose anything, unlike keeping history only in RAM.
c. Forgets things on request - a user can ask in plain conversation to stop remembering something specific, and it's actually deleted, not just ignored.
d. Recognises duplicate information - repeating a fact it already knows gets skipped instead of cluttering memory with the same thing stored twice.
e. Supports manual memory management - beyond conversation, individual memories can be deleted directly by the user through the interface.
f. Keeps memories private per user - one person's stored facts are never visible to or retrievable by another user.
g. Offers a full "forget me" option - a user can wipe every memory stored about them in one action, not just individual facts.
h. Lets the knowledge base grow over time - new documents can be uploaded and indexed at any point without restarting the server.
i. Automatically forgets stale, low-value memories - a prune policy can clear out unused, low-importance memories after a set time, so memory doesn't grow forever.
