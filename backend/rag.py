"""Shared RAG code: models, the Qdrant client, and the search filter.

Every stage imports from this file, so we write it once.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient, models

# Read the keys from the .env file into environment variables.
load_dotenv()

OPENROUTER = "https://openrouter.ai/api/v1"
COLLECTION = os.environ.get("QDRANT_COLLECTION") or "tenk"


def get_embeddings() -> OpenAIEmbeddings:
    """Embeddings turn text into a list of 1536 numbers. Similar text gets similar numbers."""
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url=OPENROUTER,
        api_key=os.environ["OPENROUTER_API_KEY"],
        check_embedding_ctx_length=False,  # required for OpenRouter
    )


def get_llm() -> ChatOpenAI:
    """The chat model, through OpenRouter. temperature=0 makes answers as repeatable as possible."""
    return ChatOpenAI(
        model=os.environ.get("CHAT_MODEL") or "anthropic/claude-haiku-5.5",
        base_url=OPENROUTER,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_tokens=1500,
    )


def get_client() -> QdrantClient:
    """Connect to the Qdrant Cloud cluster."""
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])


def make_filter(companies=None, years=None, items=None):
    """Build a Qdrant filter. An empty or missing list means 'do not filter on this'.

    LangChain stores metadata under 'metadata.', so the keys start with that.
    """
    must = []
    if companies:
        must.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies)))
    if years:
        must.append(models.FieldCondition(key="metadata.year", match=models.MatchAny(any=years)))
    if items:
        must.append(models.FieldCondition(key="metadata.item", match=models.MatchAny(any=items)))
    return models.Filter(must=must) if must else None


_store = None


def get_store() -> QdrantVectorStore:
    """The vector store, made once and reused. It embeds the question and searches Qdrant."""
    global _store
    if _store is None:
        _store = QdrantVectorStore(
            client=get_client(), collection_name=COLLECTION, embedding=get_embeddings())
    return _store


@traceable(name="qdrant_search", run_type="retriever")  # shows up as its own step in LangSmith
def search(question: str, k: int = 5, companies=None, years=None, items=None) -> list:
    """Return the k chunks closest in meaning to the question. Lists are optional filters."""
    return get_store().similarity_search(
        question, k=k, filter=make_filter(companies, years, items))


def chunk_to_source(chunk) -> dict:
    """Turn a chunk into the source dictionary that the API returns."""
    m = chunk.metadata
    return {
        "type": "10-K",
        "company": m["company"],
        "year": m["year"],
        "item": m["item"],
        "item_title": m["item_title"],
        "excerpt": chunk.page_content[:300],
        "url": m["source"],
    }


_ranker = None


def get_ranker() -> Ranker:
    """Load the reranking model once. The first time, it downloads about 3 MB into .flashrank_cache/."""
    global _ranker
    if _ranker is None:
        cache = Path(__file__).resolve().parent.parent / ".flashrank_cache"
        _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=str(cache))
    return _ranker


@traceable(name="flashrank_rerank")  # shows up as its own step in LangSmith
def rerank(question: str, docs: list, top_n: int = 5) -> list:
    """Re-read the chunks next to the question and keep the top_n best, best first."""
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = get_ranker().rerank(RerankRequest(query=question, passages=passages))
    return [docs[r["id"]] for r in ranked[:top_n]]
