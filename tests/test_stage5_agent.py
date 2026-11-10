"""Stage 5: the agent and its two tools."""
from backend.agent import NOT_REPORTED_MESSAGE, limit_to_sidebar, make_tools, read_trace
from backend.sec_data import format_amount
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tests.conftest import ask, get_question


def call(tool, **args):
    """Call a tool directly (no LLM). Returns the ToolMessage, which has .content and .artifact."""
    return tool.invoke({"name": tool.name, "args": args, "id": "t1", "type": "tool_call"})


def show(label, result):
    print(f"\n{label} answer:", result["answer"][:500])
    print("  tools_used:", result["tools_used"])
    for s in result["sources"]:
        print("  source:", {k: s[k] for k in s if k not in ("excerpt", "url")})


# ---- Live tests (call OpenRouter, and SEC) ----
def test_q4_ford_rd_2025_is_9_4_from_sec_data():
    q = get_question(4)
    result = ask(q)
    show("Q4", result)
    assert "9.4" in result["answer"]
    assert "sec_financial_data" in result["tools_used"]


def test_q5_gm_rd_2025_is_8_5_from_the_10k_text():
    q = get_question(5)
    result = ask(q)
    show("Q5", result)
    assert "8.5" in result["answer"]
    assert any(s["type"] == "10-K" and s["company"] == "GM" and s["year"] == 2025 for s in result["sources"])


def test_q6_carvana_rd_is_not_reported():
    q = get_question(6)
    result = ask(q)
    show("Q6", result)
    assert NOT_REPORTED_MESSAGE in result["answer"]
    assert "sec_financial_data" in result["tools_used"]


def test_q7_toyota_gets_company_message_and_no_sources():
    q = get_question(7)
    result = ask(q)
    show("Q7", result)
    assert result["answer"] == q["expect"]["answer_equals"]
    assert result["sources"] == [] and result["tools_used"] == []


def test_q10_sidebar_conflict_runs_no_tools():
    q = get_question(10)
    result = ask(q)
    show("Q10", result)
    assert result["answer"].startswith(q["expect"]["answer_starts_with"])
    assert "General Motors" in result["answer"]
    assert result["sources"] == [] and result["tools_used"] == []


# ---- Tool tests (SEC data, no LLM) ----
def test_tool_returns_ford_revenue_2024_as_185_0_billion():
    sec_tool = make_tools([], [])[1]
    msg = call(sec_tool, ticker="F", metric="revenue", years=[2024])
    print("\n", msg.content)
    assert "$185.0 billion (FY2024)" in msg.content
    assert msg.artifact[0]["value"] == 184_992_000_000 and msg.artifact[0]["company"] == "F"


def test_tool_does_not_return_gm_rd_2025():
    sec_tool = make_tools([], [])[1]
    msg = call(sec_tool, ticker="GM", metric="rd_expense", years=[2025])
    print("\n", msg.content)
    assert "not in the SEC financial data" in msg.content
    assert msg.artifact == []  # no invented number, and no SEC source


def test_tool_lists_what_is_supported_for_other_inputs():
    sec_tool = make_tools([], [])[1]
    for args in [dict(ticker="TM", metric="revenue", years=[2025]),
                 dict(ticker="F", metric="profit", years=[2025]),
                 dict(ticker="F", metric="revenue", years=[2022])]:
        msg = call(sec_tool, **args)
        assert "Supported companies" in msg.content and msg.artifact == []


# ---- Sidebar limits on the tools (the agent can never widen the search) ----
def test_sec_tool_respects_sidebar():
    sec_tool = make_tools(["TSLA"], [2025])[1]
    assert "sidebar does not include Ford" in call(sec_tool, ticker="F", metric="revenue", years=[2025]).content
    msg = call(sec_tool, ticker="TSLA", metric="revenue", years=[2024])
    assert "sidebar does not include this year" in msg.content and msg.artifact == []


def test_search_tool_respects_sidebar():
    search_tool = make_tools(["TSLA"], [2025])[0]
    # Asking for Ford, which the sidebar leaves out: nothing is searched.
    msg = call(search_tool, query="tariffs", companies=["F"])
    assert msg.artifact == [] and "sidebar" in msg.content
    # Asking for both: only Tesla is searched, and only 2025.
    msg = call(search_tool, query="tariffs", companies=["F", "TSLA"], years=[2024, 2025])
    assert len(msg.artifact) == 5
    assert all(s["company"] == "TSLA" and s["year"] == 2025 for s in msg.artifact)
    # Asking for nothing specific: the sidebar lists are used.
    msg = call(search_tool, query="battery supply")
    assert all(s["company"] == "TSLA" and s["year"] == 2025 for s in msg.artifact)


def test_limit_to_sidebar_rules():
    assert limit_to_sidebar(["F", "GM"], []) == (["F", "GM"], False)          # no sidebar: no limit
    assert limit_to_sidebar([], ["F"]) == (["F"], False)                       # nothing asked: use sidebar
    assert limit_to_sidebar(["F", "GM"], ["F"]) == (["F"], False)              # trimmed
    assert limit_to_sidebar(["GM"], ["F"]) == ([], True)                       # nothing left


# ---- Format and bookkeeping (no cost) ----
def test_format_amount_follows_the_number_format():
    assert format_amount(9_400_000_000, 2025) == "$9.4 billion (FY2025)"
    assert format_amount(-3_700_000_000, 2025) == "-$3.7 billion (FY2025)"
    assert format_amount(12_000_000, 2025) == "$12 million (FY2025)"


def test_read_trace_counts_only_the_current_question_in_order_of_first_call():
    old = [HumanMessage("old"), AIMessage("", tool_calls=[{"name": "search_10k", "args": {}, "id": "a"}]),
           ToolMessage("x", tool_call_id="a", artifact=[{"type": "10-K", "excerpt": "old"}]), AIMessage("done")]
    new = [HumanMessage("new"),
           AIMessage("", tool_calls=[{"name": "sec_financial_data", "args": {}, "id": "b"}]),
           ToolMessage("y", tool_call_id="b", artifact=[{"type": "sec", "value": 1}]),
           AIMessage("", tool_calls=[{"name": "search_10k", "args": {}, "id": "c"},
                                      {"name": "sec_financial_data", "args": {}, "id": "d"}]),
           ToolMessage("z", tool_call_id="c", artifact=[{"type": "10-K", "excerpt": "new"}]),
           ToolMessage("z", tool_call_id="d", artifact=[{"type": "sec", "value": 1}]),  # same source again
           AIMessage("answer")]
    tools_used, sources = read_trace(old + new)
    assert tools_used == ["sec_financial_data", "search_10k"]
    assert sources == [{"type": "sec", "value": 1}, {"type": "10-K", "excerpt": "new"}]
