"""Stage 4: the decompose pipeline (split into sub-questions, search each, answer from all)."""
import uuid

from langchain_core.documents import Document

from backend.pipelines import answer, merge_chunks, part_filters
from tests.conftest import ask, get_question


def doc(company, year, item, text, _id=None):
    meta = {"company": company, "year": year, "item": item, "item_title": "t", "source": "u"}
    if _id:
        meta["_id"] = _id
    return Document(page_content=text, metadata=meta)


# ---- Live tests (call OpenRouter) ----
def test_q3_sources_include_tesla_2025_and_ford_2025():
    result = ask(get_question(3))
    found = {(s["company"], s["year"]) for s in result["sources"]}
    print("\nQ3 answer:", result["answer"][:600])
    for s in result["sources"]:
        print("  source:", s["company"], s["year"], "Item", s["item"], "-", s["item_title"])
    assert ("TSLA", 2025) in found and ("F", 2025) in found
    assert 0 < len(result["sources"]) <= 12
    assert len({s["excerpt"] for s in result["sources"]}) == len(result["sources"])  # no repeated chunk


def test_sidebar_limits_what_decompose_can_search():
    q = get_question(3)
    result = answer(q["question"], "decompose", ["TSLA", "F"], [2025], str(uuid.uuid4()))
    assert result["sources"]
    assert all(s["company"] in ("TSLA", "F") and s["year"] == 2025 for s in result["sources"])


def test_decompose_stops_on_sidebar_conflict_and_out_of_scope():
    q = get_question(3)
    r = answer(q["question"], "decompose", ["TSLA"], [], str(uuid.uuid4()))  # Ford is left out
    assert "Ford" in r["answer"] and r["sources"] == []
    r = answer("Compare Toyota and Ford tariff risks in 2025.", "decompose", [], [], str(uuid.uuid4()))
    assert r["sources"] == []


# ---- Rule tests (no LLM, no cost) ----
def test_part_filters_use_sub_question_lists():
    assert part_filters(["TSLA"], [2025], [], [], ["TSLA", "F"], [2025]) == (["TSLA"], [2025])


def test_part_filters_fall_back_to_the_question_lists():
    assert part_filters([], [], [], [], ["TSLA", "F"], [2025]) == (["TSLA", "F"], [2025])


def test_part_filters_are_trimmed_to_a_non_empty_sidebar():
    # The sub-question wants GM, but the sidebar has only TSLA and F: it must not widen.
    assert part_filters(["GM"], [2025], ["TSLA", "F"], [], [], [])[0] == ["TSLA", "F"]
    assert part_filters(["TSLA", "GM"], [], ["TSLA", "F"], [], [], [])[0] == ["TSLA"]
    # An empty sidebar list means "not set": no trimming.
    assert part_filters(["GM"], [], [], [], [], [])[0] == ["GM"]


def test_merge_keeps_a_chunk_found_by_two_sub_questions_only_once():
    a = [doc("F", 2025, "1A", "x", "id1"), doc("F", 2025, "1A", "y", "id2")]
    b = [doc("F", 2025, "1A", "y", "id2"), doc("TSLA", 2025, "1A", "z", "id3")]
    merged = merge_chunks(a, b)
    assert [d.metadata["_id"] for d in merged] == ["id1", "id2", "id3"]
