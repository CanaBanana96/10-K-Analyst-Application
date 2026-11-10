"""Stage 1: the simple pipeline (scope_check -> retrieve -> generate). These tests call OpenRouter."""
import uuid

from backend.pipelines import COMPANY_MESSAGE, YEAR_MESSAGE, answer, generate_answer
from backend.rag import chunk_to_source
from langchain_core.documents import Document
from tests.conftest import QUESTIONS, ask, get_question


def test_q1_battery_supply_returns_five_sources():
    result = ask(get_question(1))
    assert result["answer"].strip()
    assert result["tools_used"] == []
    assert len(result["sources"]) == 5
    for s in result["sources"]:
        assert s["company"] and s["year"] and s["item"]
    # Print what was found so we can look at it with our own eyes (pytest -s).
    print("\nQ1 answer:", result["answer"])
    for s in result["sources"]:
        print("  source:", s["company"], s["year"], "Item", s["item"], "-", s["item_title"])


def test_prompt_injection_is_ignored():
    t = QUESTIONS["injection_test"]
    chunk = Document(page_content=t["chunk_text"],
                     metadata={"company": "TSLA", "year": 2025, "item": "1A",
                               "item_title": "Risk Factors", "source": "https://www.sec.gov/x"})
    reply = generate_answer(t["question"], [chunk])
    print("\nInjection reply:", reply)
    assert t["expect"]["answer_not_contains"] not in reply


def test_company_outside_the_ten_gets_company_message():
    result = answer("What risks does Toyota report?", "simple", [], [], str(uuid.uuid4()))
    assert result["answer"] == COMPANY_MESSAGE
    assert result["sources"] == [] and result["tools_used"] == []


def test_supplier_partner_counts_as_outside_company():
    result = answer("What does Tesla say about its Panasonic battery supply?", "simple", [], [], str(uuid.uuid4()))
    assert result["answer"] == COMPANY_MESSAGE
    assert result["sources"] == []


def test_year_outside_2024_2025_gets_year_message():
    result = answer("What did Ford say about tariffs in 2022?", "simple", [], [], str(uuid.uuid4()))
    assert result["answer"] == YEAR_MESSAGE
    assert result["sources"] == []


def test_outside_company_and_outside_year_gives_company_message():
    result = answer("What did Toyota say about tariffs in 2022?", "simple", [], [], str(uuid.uuid4()))
    assert result["answer"] == COMPANY_MESSAGE


def test_chunk_to_source_has_the_documented_fields():
    chunk = Document(page_content="x" * 500,
                     metadata={"company": "F", "year": 2025, "item": "7", "item_title": "MD&A",
                               "source": "https://www.sec.gov/y"})
    s = chunk_to_source(chunk)
    assert s["type"] == "10-K" and s["company"] == "F" and s["year"] == 2025
    assert len(s["excerpt"]) == 300
