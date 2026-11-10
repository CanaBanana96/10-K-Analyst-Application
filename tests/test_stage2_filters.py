"""Stage 2: the sidebar check and the filters pipeline."""
from backend.pipelines import choose_filters, sidebar_message
from tests.conftest import ask, get_question


# ---- Live tests (call OpenRouter) ----
def test_q2_all_sources_are_gm_2024():
    result = ask(get_question(2))
    assert len(result["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in result["sources"])
    print("\nQ2 answer:", result["answer"])
    for s in result["sources"]:
        print("  source:", s["company"], s["year"], "Item", s["item"], "-", s["item_title"])


def test_q9_sidebar_conflict_names_general_motors_and_searches_nothing():
    q = get_question(9)
    result = ask(q)
    print("\nQ9 answer:", result["answer"])
    assert result["answer"].startswith(q["expect"]["answer_starts_with"])
    assert "General Motors" in result["answer"]
    assert result["sources"] == [] and result["tools_used"] == []


# ---- Rule tests (no LLM, no cost) ----
def test_sidebar_message_text_and_order():
    msg = sidebar_message(["TSLA", "F"], [2025], ["GM", "F", "TSLA"], [2024, 2025])
    assert msg == ("This question asks about something that is not selected in the sidebar: "
                   "General Motors, FY2024. Select it in the sidebar, or change the question.")


def test_empty_sidebar_never_conflicts():
    assert sidebar_message([], [], ["GM"], [2024]) is None


def test_each_sidebar_list_is_handled_on_its_own():
    # Companies set in the sidebar do not hide a missing year, and an empty year list is "not set".
    assert sidebar_message(["GM"], [], ["GM"], [2024]) is None
    assert "FY2024" in sidebar_message(["GM"], [2025], ["GM"], [2024])


def test_choose_filters_rules():
    # Sidebar wins, per list.
    assert choose_filters(["TSLA"], [], ["GM"], [2024]) == (["TSLA"], [2024])
    # Otherwise the question's lists.
    assert choose_filters([], [], ["GM"], [2024]) == (["GM"], [2024])
    # Nothing anywhere: empty lists, which means "search all".
    assert choose_filters([], [], [], []) == ([], [])
