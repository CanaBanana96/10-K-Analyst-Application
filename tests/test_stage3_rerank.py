"""Stage 3: the rerank pipeline (search 20, keep the best 5)."""
from backend.pipelines import answer
from backend.rag import rerank, search
from tests.conftest import get_question


def test_q2_with_rerank_gives_five_gm_2024_sources():
    q = get_question(2)
    result = answer(q["question"], "rerank", [], [], "test-chat")
    assert result["answer"].strip()
    assert len(result["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in result["sources"])
    print("\nQ2 (rerank) answer:", result["answer"][:400])
    for s in result["sources"]:
        print("  source:", s["company"], s["year"], "Item", s["item"], "-", s["item_title"])


def test_rerank_keeps_five_of_the_twenty_without_repeats():
    # No LLM here. Search 20 chunks, rerank, and compare.
    question = "What does GM say about cybersecurity in 2024?"
    candidates = search(question, k=20, companies=["GM"], years=[2024])
    kept = rerank(question, candidates, top_n=5)
    assert len(candidates) == 20 and len(kept) == 5
    candidate_texts = [c.page_content for c in candidates]
    assert all(k.page_content in candidate_texts for k in kept)          # nothing new appears
    assert len({k.page_content for k in kept}) == 5                       # no repeats
    # Show the order change: where did each kept chunk rank before reranking?
    print("\nPositions in the original 20 of the 5 kept chunks (best first):",
          [candidate_texts.index(k.page_content) + 1 for k in kept])
