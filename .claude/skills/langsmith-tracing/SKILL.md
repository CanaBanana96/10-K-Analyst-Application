---
name: langsmith-tracing
description: Use when turning on LangSmith tracing for this project, adding the pipeline tag and the thread ID to each run, showing our own functions (the Qdrant search and the rerank) as steps with @traceable, or checking that traces arrive. LangChain steps need only settings; our own functions need @traceable.
---

# Tracing with LangSmith

LangSmith records every run of the app: each LangGraph step, each tool call, each model call, the tokens, and the time. In the LangSmith website, a run is a tree of steps. Clicking a step shows the data that went in and came out.

## Turning it on

LangChain and LangGraph send traces to LangSmith by themselves when these environment variables are set. No tracing code is needed.

| Variable | Value |
|---|---|
| `LANGSMITH_TRACING` | `true` |
| `LANGSMITH_API_KEY` | the student's key. Never print it. |
| `LANGSMITH_PROJECT` | the project name, for example `tenk-analyst` |

If the variables are not set, the app runs without tracing. Do not write code that requires them.

## The tag and the thread ID

Put two things in the config of each run:

- the **pipeline name as a tag**, so that pipelines can be filtered and compared;
- the **chat ID as the metadata field `thread_id`**, so that one conversation appears as one thread.

```python
def run_config(pipeline: str, chat_id: str) -> dict:
    return {
        "configurable": {"thread_id": chat_id},   # also used by the Neon checkpointer
        "tags": [pipeline],
        "metadata": {"thread_id": chat_id},
    }

result = graph.invoke(state, run_config(pipeline, chat_id))
```

## Seeing inside your own functions

LangSmith records LangChain and LangGraph steps by itself: model calls, chains, graph nodes, and tools. It does not record plain Python functions, such as the Qdrant search and the FlashRank rerank inside `search_10k`. Mark the functions you want to see with `@traceable`. Each call then appears as a step, with its input, output, and time. When tracing is off, the decorator does nothing.

```python
from langsmith import traceable

@traceable(run_type="retriever", name="qdrant_search")
def search(question: str, k: int = 5, flt=None) -> list:
    ...  # unchanged

@traceable(name="flashrank_rerank")
def rerank(question: str, docs: list, top_n: int = 5) -> list:
    ...  # unchanged
```

`run_type="retriever"` makes LangSmith show the returned chunks as a list of documents. Put the decorator only on the few functions that matter, so that the trace stays readable.

## Short scripts and tests

The traces are sent in the background. A long-running back end needs nothing more. A short script or a test should wait before it ends, or the last traces may be lost:

```python
from langchain_core.tracers.langchain import wait_for_all_tracers

wait_for_all_tracers()
```

## Good to know

- **A "Tracing Only" key is safer on Render.** It can send traces but cannot read them, so a leaked key exposes nothing. If you use one, check in the LangSmith website that the deployed app's traces arrive. The Personal Access Token from the setup guide works everywhere.
- **Do not read traces back from the app.** The app only sends traces. Reading runs through older SDK methods calls an endpoint that LangSmith is retiring (`POST /runs/query`).
- **Cost shows $0** for OpenRouter model names, because LangSmith does not know their prices. The token counts are correct.
- Traces can take a few seconds to appear in the website.
