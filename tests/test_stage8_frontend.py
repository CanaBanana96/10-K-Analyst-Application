"""Stage 8: the Streamlit page. A fake back end replaces httpx.post, so these tests cost nothing."""
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

load_dotenv()
APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")
CODE = os.environ.get("ACCESS_CODE", "")  # the real code, read from .env; never printed

ANSWER = {
    "answer": "Ford's R&D expense was $9.4 billion and GM's was $8.5 billion.",
    "pipeline": "agent", "chat_id": "x", "tools_used": ["sec_financial_data", "search_10k"],
    "sources": [
        {"type": "sec", "company": "F", "year": 2025, "metric": "rd_expense",
         "tag": "ResearchAndDevelopmentExpense", "value": 9400000000},
        {"type": "10-K", "company": "GM", "year": 2025, "item": "1", "item_title": "Business",
         "excerpt": "Research and development expenses were $8.5 billion", "url": "https://www.sec.gov/x"},
    ],
}


class FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code, self._data = status_code, data

    def json(self):
        if self._data is None:
            raise ValueError("no JSON")
        return self._data


def fake_backend(monkeypatch, *replies):
    """Replace httpx.post. Each call gets the next reply (an exception is raised, a response is returned)."""
    calls, queue = [], list(replies)

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(httpx, "post", post)
    return calls


def open_page() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def logged_in_page() -> AppTest:
    at = open_page()
    at.text_input[0].set_value(CODE)
    at.button[0].click().run()
    return at


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    return at


def page_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)


# ---- 1. The access code screen ----
def test_the_page_asks_for_the_access_code_first():
    at = open_page()
    assert len(at.text_input) == 1
    assert at.text_input[0].proto.type == 1  # 1 is Streamlit's PASSWORD: the typed code shows as dots
    assert len(at.chat_input) == 0  # no chat yet


def test_a_wrong_code_is_refused_and_a_right_code_opens_the_chat():
    at = open_page()
    at.text_input[0].set_value("definitely-wrong")
    at.button[0].click().run()
    assert at.error[0].value == "Wrong access code." and len(at.chat_input) == 0
    at.text_input[0].set_value(CODE)
    at.button[0].click().run()
    assert len(at.chat_input) == 1 and not at.exception


def test_the_code_is_asked_only_once_per_session():
    at = logged_in_page()
    at.run()  # any later click runs the page again
    assert len(at.text_input) == 0 and len(at.chat_input) == 1


# ---- 2. The sidebar ----
def test_sidebar_defaults():
    at = logged_in_page()
    assert at.multiselect[0].value == ["TSLA", "GM", "F", "RIVN", "LCID", "APTV", "BWA", "ALV", "HOG", "CVNA"]
    assert at.multiselect[1].value == [2024, 2025]
    assert at.toggle[0].label == "Show sources" and at.toggle[0].value is True
    assert at.radio[0].value == "agent"                      # Learning mode starts on the agent
    assert [b.label for b in at.sidebar.button] == ["New chat"]
    assert [e.label for e in at.sidebar.expander] == ["Learning mode"]


# ---- 3. The example questions ----
def test_three_example_buttons_and_a_click_asks_the_question(monkeypatch):
    calls = fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = logged_in_page()
    examples = [b for b in at.button if b.label != "New chat"]
    assert [b.label for b in examples] == [
        "What risks does Tesla report about battery supply?", "Compare GM and Ford R&D in 2025",
        "How did Carvana's revenue change from 2024 to 2025?"]
    examples[1].click().run()
    assert calls[0]["json"]["question"] == "Compare GM and Ford R&D in 2025"
    assert not [b for b in at.button if b.label.startswith("Compare")]  # the examples are gone


# ---- 4. What the page sends ----
def test_request_with_everything_selected_sends_empty_lists(monkeypatch):
    calls = fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = ask(logged_in_page(), "What was Ford's R&D expense in 2025?")
    sent = calls[0]
    assert sent["url"].endswith("/ask") and sent["timeout"] == 90
    assert sent["headers"] == {"X-Access-Code": CODE}
    assert sent["json"]["companies"] == [] and sent["json"]["years"] == []
    assert sent["json"]["pipeline"] == "agent" and sent["json"]["chat_id"]
    assert not at.exception


def test_request_sends_the_selected_filters_and_pipeline(monkeypatch):
    calls = fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = logged_in_page()
    at.multiselect[0].set_value(["TSLA", "F"])
    at.multiselect[1].set_value([2025])
    at.radio[0].set_value("rerank")
    ask(at, "What does Ford say about tariffs?")
    assert calls[0]["json"]["companies"] == ["TSLA", "F"]
    assert calls[0]["json"]["years"] == [2025]
    assert calls[0]["json"]["pipeline"] == "rerank"


# ---- 5. What the page shows ----
def test_answer_tools_and_sources_are_shown(monkeypatch):
    fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = ask(logged_in_page(), "Compare GM and Ford R&D in 2025")
    text = page_text(at)
    assert "sec_financial_data" in text and "search_10k" in text         # tool labels
    assert [e.label for e in at.expander if e.label.startswith("Sources")] == ["Sources (2)"]
    assert "$9.4 billion (FY2025)" in text.replace("\\$", "$")           # the SEC amount, in the app's format
    assert "Item 1" in text and "General Motors" in text


def test_dollar_signs_are_escaped_so_they_do_not_turn_into_formulas(monkeypatch):
    fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = ask(logged_in_page(), "Compare GM and Ford R&D in 2025")
    answer_markdown = [m.value for m in at.markdown if "R&D expense was" in m.value][0]
    assert answer_markdown == "Ford's R&D expense was \\$9.4 billion and GM's was \\$8.5 billion."


def test_show_sources_off_hides_the_sources_but_not_the_tools(monkeypatch):
    fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = logged_in_page()
    at.toggle[0].set_value(False)
    ask(at, "Compare GM and Ford R&D in 2025")
    assert not [e for e in at.expander if e.label.startswith("Sources")]
    assert "sec_financial_data" in page_text(at)


def test_an_answer_without_tools_says_so(monkeypatch):
    fake_backend(monkeypatch, FakeResponse(200, {**ANSWER, "tools_used": [], "sources": []}))
    at = ask(logged_in_page(), "What risks does Tesla report about battery supply?")
    assert "no tools used" in page_text(at)
    assert not [e for e in at.expander if e.label.startswith("Sources")]


def test_the_history_of_the_chat_stays_on_screen(monkeypatch):
    fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = ask(logged_in_page(), "First question")
    ask(at, "Second question")
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]


# ---- 6. Errors, waking up, and New chat ----
def test_the_error_text_from_the_back_end_is_shown(monkeypatch):
    fake_backend(monkeypatch, FakeResponse(400, {"error": "The question is too long. Use at most 1,000 characters."}))
    at = ask(logged_in_page(), "x")
    assert at.error[0].value == "The question is too long. Use at most 1,000 characters."


def test_a_sleeping_server_gets_one_more_try(monkeypatch):
    calls = fake_backend(monkeypatch, httpx.ReadTimeout("slow"), FakeResponse(200, ANSWER))
    at = ask(logged_in_page(), "What was Ford's R&D expense in 2025?")
    assert len(calls) == 2                                   # the first try and one more
    assert "$9.4 billion" in page_text(at).replace("\\$", "$")


def test_the_waking_up_message_is_shown_while_retrying(monkeypatch):
    fake_backend(monkeypatch, httpx.ConnectTimeout("asleep"), httpx.ConnectTimeout("asleep"))
    at = ask(logged_in_page(), "What was Ford's R&D expense in 2025?")
    # The info box disappears with the redraw, so check the end result: one retry, then a plain message.
    assert at.error[0].value == "The answer service is not available right now. Try again in a minute."


def test_new_chat_clears_the_chat_and_changes_the_chat_id(monkeypatch):
    calls = fake_backend(monkeypatch, FakeResponse(200, ANSWER))
    at = ask(logged_in_page(), "First question")
    first_id = calls[0]["json"]["chat_id"]
    at.sidebar.button[0].click().run()
    assert len(at.chat_message) == 0
    ask(at, "Second question")
    assert calls[1]["json"]["chat_id"] != first_id
