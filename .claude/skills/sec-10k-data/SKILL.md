---
name: sec-10k-data
description: Use when reading the 10-K files in data/, splitting them by Item, or getting financial numbers (revenue, net income, R&D, and so on) from SEC's XBRL company facts. Also use for the rule about the SEC User-Agent and for the fallback when a number is not in the SEC data.
---

# The 10-K data and SEC financial data

## The 10-K files

The starter kit already contains the filings. Do not download 10-Ks from SEC.

- `data/filings.csv` has one row per filing: `ticker, company, cik, fiscal_year, period_end, filed, accession, source_url, file`.
- `data/10k/<TICKER>_<YEAR>.md` is the plain text of one filing. Each Item starts with a level-2 heading, for example `## Item 1A. Risk Factors`. Tables are rows of cells separated by ` | `.

Split a filing into Items with the headings. Keep the Item code as metadata.

```python
import re
from pathlib import Path

ITEM = re.compile(r"^## Item (\w+)\. (.+)$", re.M)

def read_items(path: Path) -> list[dict]:
    """Return one dict per Item: code, title, and text."""
    text = path.read_text(encoding="utf-8")
    heads = list(ITEM.finditer(text))
    items = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append({"code": h.group(1), "title": h.group(2), "text": text[h.end():end].strip()})
    return items
```

Useful Items: 1 (Business), 1A (Risk Factors), 7 (MD&A), 7A (Market Risk), 8 (Financial Statements). Items 10 to 14 are usually short, because most companies refer to their proxy statement.

## SEC financial data (XBRL company facts)

SEC publishes the numbers from every 10-K as XBRL data. One request returns all facts for one company:

`https://data.sec.gov/api/xbrl/companyfacts/CIK<10-digit CIK>.json`

**Rules.**

1. SEC requires a `User-Agent` header with a name and an email. Read it from the environment variable `SEC_USER_AGENT`. Never hard-code a person's email.
2. Stay well under 10 requests per second. Cache each company's facts in memory, because the file is large and does not change during a session.
3. The CIK of the ten companies is in `data/filings.csv`. For any other company, use `https://www.sec.gov/files/company_tickers.json`.

**Getting a full-year number.** Use the `frame` field. A full calendar year has the frame `CY<year>`, for example `CY2025`. This picks the right value even though each 10-K also repeats older years.

```python
import functools, json, os, urllib.request

@functools.lru_cache(maxsize=32)
def company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    req = urllib.request.Request(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def annual_value(cik: str, tags: list[str], year: int):
    """Return (value, tag) for a full-year amount, or (None, None) if it is not reported."""
    facts = company_facts(cik)["facts"].get("us-gaap", {})
    for tag in tags:
        for values in facts.get(tag, {}).get("units", {}).values():
            for v in values:
                if v.get("frame") == f"CY{year}":
                    return v["val"], tag
    return None, None
```

**Common tags.** Try them in this order.

| Metric | Tags |
|---|---|
| Revenue | `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax` |
| Net income | `NetIncomeLoss`, `ProfitLoss` |
| R&D expense | `ResearchAndDevelopmentExpense`, `ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost` |
| Operating income | `OperatingIncomeLoss` |
| Total assets | `Assets` (a point in time: use frame `CY<year>Q4I`) |
| Cash | `CashAndCashEquivalentsAtCarryingValue` (frame `CY<year>Q4I`) |

**When a number is missing.** Some companies do not tag a number. GM, for example, has not tagged R&D since 2013, but its 10-K text states it. If `annual_value` returns `None`:

1. Say that the number is not in the SEC financial data.
2. Search the 10-K text for it (Item 1 or Item 7 usually).
3. If the text does not have it either, say so. Never estimate or invent a number.

**Checked values** (useful for tests): Ford R&D 2025 = 9.40 billion. Ford revenue 2024 = 185.0 billion. GM R&D 2025 = not in the SEC data; GM's 10-K Item 1 says $8.5 billion.
