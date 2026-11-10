"""Stage 0: the ingest fills Qdrant with 10,229 chunks, and a second run changes nothing."""
from backend.ingest import ingest
from backend.rag import COLLECTION, get_client, make_filter

EXPECTED = 10_229


def count(client, companies=None, years=None) -> int:
    return client.count(COLLECTION, count_filter=make_filter(companies, years), exact=True).count


def test_first_run_gives_expected_count():
    # On the very first run this embeds everything (about a minute). Later runs skip all filings.
    assert ingest() == EXPECTED


def test_second_run_keeps_the_same_count():
    assert ingest() == EXPECTED
    assert count(get_client()) == EXPECTED


def test_every_filing_is_present_and_metadata_is_right():
    client = get_client()
    for company in ["TSLA", "GM", "F", "RIVN", "LCID", "APTV", "BWA", "ALV", "HOG", "CVNA"]:
        for year in (2024, 2025):
            assert count(client, [company], [year]) > 0, f"{company} {year} has no chunks"
    # A stored chunk carries company, year, Item code, Item title, and the SEC link.
    points, _ = client.scroll(COLLECTION, scroll_filter=make_filter(["F"], [2025]), limit=1)
    meta = points[0].payload["metadata"]
    assert meta["company"] == "F" and meta["year"] == 2025
    assert meta["item"] and meta["item_title"] and meta["source"].startswith("https://www.sec.gov/")
