"""Helpers shared by the test files."""
import json
import uuid
from pathlib import Path

import pytest

from backend.pipelines import answer

QUESTIONS = json.loads((Path(__file__).resolve().parent.parent / "test_data" / "questions.json").read_text())


def get_question(qid: int) -> dict:
    return next(q for q in QUESTIONS["questions"] if q["id"] == qid)


def ask(q: dict) -> dict:
    """Run one entry of questions.json through answer(), with its sidebar and a new chat ID."""
    sidebar = q.get("sidebar", {})
    return answer(q["question"], q["pipeline"], sidebar.get("companies", []),
                  sidebar.get("years", []), str(uuid.uuid4()))


@pytest.fixture(scope="session")
def questions():
    return QUESTIONS


@pytest.fixture(scope="session", autouse=True)
def send_traces_before_exit():
    """Tests end quickly. Wait so that the last LangSmith traces are sent (does nothing if tracing is off)."""
    yield
    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
    from backend.agent import close_pool
    close_pool()  # close the connections to Neon (does nothing if they were never opened)
