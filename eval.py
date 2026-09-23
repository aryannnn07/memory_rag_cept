"""
Evaluate Memory RAG against Vanilla RAG with a scripted, multi-session scenario.

    python eval.py

Each check asks a question in a NEW session (so short-term memory cannot help)
and looks for expected keywords in the answer. The same questions are asked in
vanilla mode for comparison. Uses a separate database folder, so your demo data
is untouched.
"""
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory_rag import MemoryRAG  # noqa: E402

DB = "./chroma_eval"
USER, OTHER = "eval_student", "someone_else"

# Things the student tells the tutor, one session each.
TEACH = [
    "Hi! My name is Aryan and I'm a third-year computer science student.",
    "I learn best with simple real-life analogies rather than heavy maths.",
    "Honestly I'm struggling with the query, key and value part of attention.",
    "My GenAI exam is on 5th October.",
]

# (name, question, keywords — pass if ANY keyword appears, lower-cased)
CHECKS = [
    ("Recall: name", "Do you remember my name?", ["aryan"]),
    ("Recall: weak topic", "Which topic did I say I was struggling with?", ["attention", "query", "key"]),
    ("Recall: deadline", "When is my exam?", ["5", "october"]),
    ("Personalised retrieval", "What should I revise before my exam?", ["attention", "query", "key", "value"]),
]


def ask(rag, session, question, mode="memory", user=USER):
    return rag.chat(user, session, question, mode)


def contains(answer, keywords):
    a = answer.lower()
    return any(k in a for k in keywords)


def main():
    shutil.rmtree(DB, ignore_errors=True)
    rag = MemoryRAG(db_path=DB)
    rag.add_knowledge(Path("data/course_notes.md").read_text(), source="course_notes")

    print("Teaching the tutor about the student (one session per fact)...")
    for i, msg in enumerate(TEACH):
        ops = ask(rag, f"teach-{i}", msg)["memory_operations"]
        print(f"  {msg[:55]:<57} -> {[o['op'] for o in ops]}")

    rows = []
    for i, (name, question, keys) in enumerate(CHECKS):
        mem = ask(rag, f"check-{i}", question)["answer"]
        van = ask(rag, f"check-{i}-v", question, mode="vanilla")["answer"]
        rows.append((name, contains(mem, keys), contains(van, keys)))

    # Update: a fact changes.
    ask(rag, "update", "Update: my exam got moved to 12th October.")
    upd = ask(rag, "update-check", "When is my exam now?")["answer"]
    rows.append(("Update: changed deadline", contains(upd, ["12"]) and "5th" not in upd.lower(), False))

    # Forget: the user withdraws information.
    ask(rag, "forget", "Please forget my exam date, I don't want you to store it.")
    stored = " ".join(m["text"].lower() for m in rag.list_memories(USER))
    rows.append(("Forget: exam date removed", "october" not in stored and "12" not in stored, False))

    # Isolation: another user must not see Aryan's memories.
    iso = ask(rag, "iso", "What is my name?", user=OTHER)
    rows.append(("Isolation: other user", not iso["memories_used"] and "aryan" not in iso["answer"].lower(), True))

    print(f"\n{'Check':<30}{'Memory RAG':>12}{'Vanilla RAG':>14}")
    print("-" * 56)
    for name, mem_ok, van_ok in rows:
        van = "n/a" if name.startswith(("Update", "Forget")) else ("PASS" if van_ok else "fail")
        print(f"{name:<30}{'PASS' if mem_ok else 'fail':>12}{van:>14}")
    passed = sum(1 for _, ok, _ in rows if ok)
    print(f"\nMemory RAG passed {passed}/{len(rows)} checks.")
    print("\nFinal memories stored for the student:")
    for m in rag.list_memories(USER):
        print(f"  [{m['importance']:>2}] {m['text']}")


if __name__ == "__main__":
    main()
