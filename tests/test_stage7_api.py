"""Stage 7: the FastAPI back end."""
import asyncio
import logging
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.tracers.context import collect_runs

from backend import api
from backend.api import app, check_environment
from backend.pipelines import answer, run_config
from tests.conftest import get_question

client = TestClient(app)  # not used as a "with" block, so the start-up check does not run here
CODE = {"X-Access-Code": os.environ.get("ACCESS_CODE", "")}  # the real code, read from .env; never printed


def body(**changes) -> dict:
    return {"question": "What was Ford's R&D expense in 2025?", "chat_id": str(uuid.uuid4()),
            "pipeline": "agent", "companies": [], "years": [], **changes}


# ---- /health ----
def test_health_returns_200_without_a_code():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


# ---- The error table (no LLM is called: every one of these stops before answer()) ----
def test_missing_access_code_is_401():
    r = client.post("/ask", json=body())
    assert r.status_code == 401 and r.json() == {"error": "Wrong or missing access code."}


def test_wrong_access_code_is_401():
    r = client.post("/ask", json=body(), headers={"X-Access-Code": "definitely-wrong"})
    assert r.status_code == 401 and r.json() == {"error": "Wrong or missing access code."}


def test_empty_question_is_400():
    for q in ["", "   "]:
        r = client.post("/ask", json=body(question=q), headers=CODE)
        assert r.status_code == 400 and r.json() == {"error": "The question is empty."}


def test_too_long_question_is_400():
    r = client.post("/ask", json=body(question="x" * 1001), headers=CODE)
    assert r.status_code == 400
    assert r.json() == {"error": "The question is too long. Use at most 1,000 characters."}


def test_unknown_pipeline_is_400():
    r = client.post("/ask", json=body(pipeline="magic"), headers=CODE)
    assert r.status_code == 400
    assert r.json() == {"error": "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."}


def test_the_access_code_is_checked_first():
    r = client.post("/ask", json=body(question="", pipeline="magic"))  # three things wrong, no code
    assert r.status_code == 401
    r = client.post("/ask", json={"nonsense": 1})  # not even a valid request body
    assert r.status_code == 401


def test_a_question_of_exactly_1000_characters_is_allowed(monkeypatch):
    seen = {}

    def fake_answer(question, pipeline, companies, years, chat_id):
        seen.update(question=question, pipeline=pipeline, companies=companies, years=years, chat_id=chat_id)
        return {"answer": "ok", "pipeline": pipeline, "chat_id": chat_id, "tools_used": [], "sources": []}

    monkeypatch.setattr(api, "answer", fake_answer)
    r = client.post("/ask", json=body(question="x" * 1000, companies=["F"], years=[2025]), headers=CODE)
    assert r.status_code == 200 and r.json()["answer"] == "ok"
    assert len(seen["question"]) == 1000 and seen["companies"] == ["F"] and seen["years"] == [2025]


# ---- 503: a failing service gives the fixed message, and the log shows only the error type ----
def test_service_failure_is_503_and_nothing_secret_leaks(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise ConnectionError("failed with key sk-SECRET-12345")

    monkeypatch.setattr(api, "answer", broken)
    with caplog.at_level(logging.ERROR, logger="tenk"):
        r = client.post("/ask", json=body(), headers=CODE)
    assert r.status_code == 503
    assert r.json() == {"error": "The answer service is not available right now. Try again in a minute."}
    assert "sk-SECRET" not in r.text and "Traceback" not in r.text
    assert "ConnectionError" in caplog.text            # the type is logged
    assert "sk-SECRET" not in caplog.text              # the message is not


# ---- The start-up check ----
GOOD_ENV = {name: "value" for name in api.REQUIRED_VARIABLES}


def test_startup_check_accepts_a_good_environment():
    assert check_environment(GOOD_ENV) == []
    assert check_environment({**GOOD_ENV, "LANGSMITH_API_KEY": "ls-key"}) == []  # LangSmith is optional


def test_startup_check_names_missing_variables_and_never_shows_values():
    env = {**GOOD_ENV, "ACCESS_CODE": ""}
    del env["QDRANT_URL"]
    problems = check_environment(env)
    assert problems == ["QDRANT_URL is not set", "ACCESS_CODE is not set"]


def test_startup_check_finds_line_breaks_including_langsmith():
    env = {**GOOD_ENV, "OPENROUTER_API_KEY": "sk-abc\nextra-line", "LANGSMITH_API_KEY": "ls\r\nkey"}
    problems = check_environment(env)
    assert problems == ["OPENROUTER_API_KEY contains a line break", "LANGSMITH_API_KEY contains a line break"]
    assert all("sk-abc" not in p and "extra-line" not in p for p in problems)


def test_the_server_refuses_to_start_when_a_variable_is_missing(monkeypatch, capsys):
    monkeypatch.delenv("SEC_USER_AGENT")

    async def start():
        async with api.lifespan(app):  # this is the start-up code that the server runs
            pass

    with pytest.raises(SystemExit):
        asyncio.run(start())
    assert "SEC_USER_AGENT is not set" in capsys.readouterr().out


def test_the_server_starts_with_the_real_environment():
    with TestClient(app) as started:
        assert started.get("/health").status_code == 200


# ---- Tag and thread_id for LangSmith ----
def test_run_config_has_the_tag_and_thread_id():
    assert run_config("rerank", "chat-1") == {
        "configurable": {"thread_id": "chat-1"}, "tags": ["rerank"], "metadata": {"thread_id": "chat-1"}}


def test_real_runs_carry_the_pipeline_tag_and_thread_id():
    # A scope-check-only call is cheap. collect_runs sees the runs whether or not LangSmith is on.
    for pipeline in ["simple", "agent"]:
        chat_id = str(uuid.uuid4())
        with collect_runs() as cb:
            answer("What risks does Toyota report?", pipeline, [], [], chat_id)
        assert cb.traced_runs, "no runs were recorded"
        for run in cb.traced_runs:
            assert pipeline in run.tags
            assert run.extra["metadata"]["thread_id"] == chat_id


# ---- One full call through the API (calls OpenRouter and SEC) ----
def test_ask_returns_the_documented_response_for_q4():
    q = get_question(4)
    chat_id = str(uuid.uuid4())
    r = client.post("/ask", json=body(question=q["question"], chat_id=chat_id, pipeline=q["pipeline"]),
                    headers=CODE)
    data = r.json()
    print("\nResponse:", {k: data[k] for k in ("answer", "pipeline", "chat_id", "tools_used")})
    print("Sources:", [{k: s[k] for k in s if k != "excerpt"} for s in data["sources"]])
    assert r.status_code == 200
    assert set(data) == {"answer", "pipeline", "chat_id", "tools_used", "sources"}
    assert data["pipeline"] == "agent" and data["chat_id"] == chat_id
    assert "9.4" in data["answer"] and "sec_financial_data" in data["tools_used"]
    assert data["sources"][0] == {"type": "sec", "company": "F", "year": 2025, "metric": "rd_expense",
                                  "tag": "ResearchAndDevelopmentExpense", "value": 9400000000}
