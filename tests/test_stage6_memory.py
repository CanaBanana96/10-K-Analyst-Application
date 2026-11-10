"""Stage 6: the agent remembers the conversation in Neon. Each run uses new chat IDs."""
import uuid

from backend.agent import close_pool, get_checkpointer
from backend.pipelines import answer
from tests.conftest import get_question


def ask_in_chat(question: str, chat_id: str) -> dict:
    return answer(question, "agent", [], [], chat_id)


def test_q8_follow_up_uses_the_previous_question():
    q = get_question(8)
    chat_id = str(uuid.uuid4())  # a new chat on every run, so old runs cannot help
    first = ask_in_chat(q["conversation"][0], chat_id)
    second = ask_in_chat(q["conversation"][1], chat_id)
    print("\nTurn 1:", first["answer"], first["tools_used"])
    print("Turn 2:", second["answer"], second["tools_used"])
    for s in second["sources"]:
        print("  source:", {k: s[k] for k in s if k not in ("excerpt", "url")})
    assert "185.0" in first["answer"]
    assert "187.3" in second["answer"]
    # Sources and tools come only from the second question's own tool calls (here: Ford 2025 only).
    assert second["tools_used"] and all(s["year"] == 2025 for s in second["sources"])


def test_a_different_chat_does_not_remember():
    chat_id = str(uuid.uuid4())
    ask_in_chat("What was Ford's revenue in 2024?", chat_id)
    other = ask_in_chat("And in 2025?", str(uuid.uuid4()))  # a different, new chat
    print("\nNew chat, follow-up only:", other["answer"])
    assert "187.3" not in other["answer"]


def test_the_conversation_is_stored_in_neon_not_in_memory():
    chat_id = str(uuid.uuid4())
    ask_in_chat("What was Ford's revenue in 2024?", chat_id)
    close_pool()  # drop every connection; a new pool must read the chat back from the database
    saved = get_checkpointer().get_tuple({"configurable": {"thread_id": chat_id}})
    assert saved is not None
    texts = [str(m.content) for m in saved.checkpoint["channel_values"]["messages"]]
    print("\nMessages saved in Neon:", len(texts))
    assert any("Ford's revenue in 2024" in t for t in texts)
    assert any("185.0" in t for t in texts)
