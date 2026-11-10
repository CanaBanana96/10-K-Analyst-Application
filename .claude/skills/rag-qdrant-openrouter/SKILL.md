---
name: rag-qdrant-openrouter
description: Use when building the RAG parts of this project - chunking the 10-K Items, embedding with OpenRouter, storing and searching in Qdrant with company, year, and Item filters, reranking with FlashRank, or splitting a question into parts. Use this instead of FAISS, Chroma, or direct OpenAI examples.
---

# RAG with Qdrant and OpenRouter

This project uses **Qdrant Cloud** for vectors and **OpenRouter** for all model calls, including embeddings. Do not use FAISS, Chroma, or the OpenAI API directly.

Keys come from environment variables: `OPENROUTER_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`.

## Models

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

OPENROUTER = "https://openrouter.ai/api/v1"

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",       # 1536 numbers per vector
    base_url=OPENROUTER,
    api_key=os.environ["OPENROUTER_API_KEY"],
    check_embedding_ctx_length=False,           # required for OpenRouter
)

llm = ChatOpenAI(
    model=os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5"),
    base_url=OPENROUTER,
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
    max_tokens=1500,
)
```

## Chunking

Split each Item into chunks of about 1,500 characters with 150 characters of overlap. Never let a chunk cross from one Item into the next. Put the company, year, and Item in the metadata of every chunk.

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
docs = [
    Document(page_content=chunk, metadata={"company": ticker, "year": year, "item": item["code"],
                                           "item_title": item["title"], "source": source_url})
    for item in items for chunk in splitter.split_text(item["text"])
]
```

With these settings, the 20 filings give exactly 10,229 chunks. Embedding them takes about one minute.

## Qdrant collection and filters

Create the collection once, with an index on each metadata field that you filter on. Use one collection for all companies and years.

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tenk")

if not client.collection_exists(COLLECTION):
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
    client.create_payload_index(COLLECTION, "metadata.company", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.item", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.year", models.PayloadSchemaType.INTEGER)

store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
store.add_documents(docs, batch_size=200)
```

LangChain stores metadata under `metadata.`, so filters use keys such as `metadata.company`.

```python
def make_filter(companies=None, years=None, items=None):
    must = []
    if companies:
        must.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies)))
    if years:
        must.append(models.FieldCondition(key="metadata.year", match=models.MatchAny(any=years)))
    if items:
        must.append(models.FieldCondition(key="metadata.item", match=models.MatchAny(any=items)))
    return models.Filter(must=must) if must else None

hits = store.similarity_search("battery cell suppliers", k=8, filter=make_filter(["F"], [2025]))
```

To make ingesting safe to run twice, give each chunk a stable ID, and skip a filing that is already complete.

```python
import uuid

def chunk_id(company: str, year: int, item: str, n: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{year}/{item}/{n}"))

def points_for(company: str, year: int) -> int:
    flt = make_filter([company], [year])
    return client.count(COLLECTION, count_filter=flt, exact=True).count

# Skip the filing when points_for(company, year) == number of chunks it splits into.
# Otherwise add its chunks with ids=[chunk_id(...) for each chunk]; the same IDs overwrite a partial earlier run.
```

## Reranking

Rerank with FlashRank. Its model is about 3 MB and runs on the CPU, so it fits on a small server. Search for more chunks than you need (for example 20), then keep the best few (for example 5).

```python
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")   # load once, at start-up

def rerank(question: str, docs: list, top_n: int = 5) -> list:
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = ranker.rerank(RerankRequest(query=question, passages=passages))
    return [docs[r["id"]] for r in ranked[:top_n]]
```

## Splitting a question into parts (decomposition)

Ask the LLM for a short list of sub-questions, each with the companies and years it is about. Search each sub-question with its own filter, then answer from all results together. In LangGraph, `Send` runs the searches in parallel (see the `langgraph-fundamentals` skill).

## Answers

The answer must use only the retrieved chunks and the SEC data. Every answer cites its sources: company, fiscal year, and Item (or the SEC data field). If the sources do not contain the answer, the answer says so.
