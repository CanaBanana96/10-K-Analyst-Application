"""Exact numbers from SEC's XBRL company facts (revenue, net income, R&D, operating income)."""
import csv
import functools
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).resolve().parent.parent / "data"

# Which SEC tags to try for each metric, in this order.
METRIC_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense",
                   "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "operating_income": ["OperatingIncomeLoss"],
}


@functools.lru_cache(maxsize=1)
def ciks() -> dict[str, str]:
    """Ticker -> CIK (SEC's company number), from data/filings.csv."""
    with open(DATA / "filings.csv", encoding="utf-8", newline="") as f:
        return {row["ticker"]: row["cik"] for row in csv.DictReader(f)}


@functools.lru_cache(maxsize=16)  # one download per company; the file is large and does not change
def company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    # SEC requires a name and email in the User-Agent. It comes from .env, never from the code.
    response = httpx.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=60)
    response.raise_for_status()
    return response.json()


def annual_value(cik: str, tags: list[str], year: int):
    """Return (value, tag) for a full-year amount, or (None, None) if SEC data does not have it.

    The frame 'CY2025' marks a full calendar year. It picks the right value even though
    each 10-K also repeats older years.
    """
    facts = company_facts(cik)["facts"].get("us-gaap", {})
    for tag in tags:
        for values in facts.get(tag, {}).get("units", {}).values():
            for v in values:
                if v.get("frame") == f"CY{year}":
                    return v["val"], tag
    return None, None


def get_metric(ticker: str, metric: str, year: int):
    """Look up one metric for one company and year. Returns (value, tag) or (None, None)."""
    return annual_value(ciks()[ticker], METRIC_TAGS[metric], year)


def format_amount(value: float, year: int) -> str:
    """Write an amount the way the app always writes it, so the LLM never converts numbers.

    $50 million or more -> billions with one decimal: "$9.4 billion (FY2025)", "-$3.7 billion (FY2025)".
    Smaller amounts -> as the filing states them: "$12 million (FY2025)".
    """
    sign = "-" if value < 0 else ""
    size = abs(value)
    if size >= 50_000_000:  # $0.05 billion
        text = f"${size / 1e9:.1f} billion"
    elif size >= 1_000_000:
        text = f"${size / 1e6:,.1f}".removesuffix(".0") + " million"
    else:
        text = f"${size:,.0f}"
    return f"{sign}{text} (FY{year})"
