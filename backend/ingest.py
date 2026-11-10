"""Read the 10-K files, cut them into chunks, and store the chunks in Qdrant.

Run with:  uv run python -m backend.ingest
"""
import csv
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from backend.rag import COLLECTION, get_client, get_embeddings, make_filter

DATA = Path(__file__).resolve().parent.parent / "data"
ITEM = re.compile(r"^## Item (\w+)\. (.+)$", re.M)  # headings like "## Item 1A. Risk Factors"
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)


def read_items(path: Path) -> list[dict]:
    """Return one dict per Item: code, title, and text."""
    text = path.read_text(encoding="utf-8")
    heads = list(ITEM.finditer(text))
    items = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append({"code": h.group(1), "title": h.group(2), "text": text[h.end():end].strip()})
    return items


def chunk_id(company: str, year: int, item: str, n: int) -> str:
    """A stable ID: the same chunk always gets the same ID, so running twice never makes copies."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{year}/{item}/{n}"))


def chunks_for(row: dict) -> tuple[list[Document], list[str]]:
    """Split one filing into chunks. Each Item is split on its own, so a chunk never crosses Items."""
    company, year = row["ticker"], int(row["fiscal_year"])
    docs, ids = [], []
    for item in read_items(DATA / row["file"]):
        for n, piece in enumerate(splitter.split_text(item["text"])):
            docs.append(Document(
                page_content=piece,
                metadata={"company": company, "year": year, "item": item["code"],
                          "item_title": item["title"], "source": row["source_url"]},
            ))
            ids.append(chunk_id(company, year, item["code"], n))
    return docs, ids


def create_collection_if_needed(client) -> None:
    """Make the collection once. Indexes let Qdrant filter fast on company, year, and Item."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
    client.create_payload_index(COLLECTION, "metadata.company", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.item", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.year", models.PayloadSchemaType.INTEGER)


def ingest() -> int:
    """Store every filing. Returns the number of chunks in the collection at the end."""
    client = get_client()
    create_collection_if_needed(client)
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=get_embeddings())

    with open(DATA / "filings.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        company, year = row["ticker"], int(row["fiscal_year"])
        docs, ids = chunks_for(row)
        stored = client.count(COLLECTION, count_filter=make_filter([company], [year]), exact=True).count
        if stored == len(docs):
            print(f"{company} {year}: skipped ({stored} chunks already stored)")
            continue
        # Same IDs overwrite any partial earlier run.
        store.add_documents(docs, ids=ids, batch_size=200)
        print(f"{company} {year}: stored {len(docs)} chunks")

    total = client.count(COLLECTION, exact=True).count
    print(f"Collection '{COLLECTION}' now holds {total} chunks.")
    return total


if __name__ == "__main__":
    ingest()
