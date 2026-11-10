# Build plan: 10-K Analyst

## Context

`requirements.md` describes a question-answering app over 20 10-K filings (10 auto companies, FY2024 and FY2025). It is built in stages, from a simple RAG pipeline up to an agent with memory, a FastAPI back end, a Streamlit front end, and Render. This plan lists the files, the stages, how the code is shared, and the decisions agreed with the user. **First step after approval: save this plan as `plan.md` in the project folder.** Then build Stage 0 only, test it, and stop.

## General style

- Keep each file short. Use plain functions with clear names, no clever code, and a short comment wherever a beginner would get stuck.
- Run everything with `uv run`. Never read `.env` (it is filled in). Never print a key or a connection string.

## Files

| File | Purpose |
|---|---|
| `backend/__init__.py` | Empty. Makes `backend` a package so `python -m backend.ingest` works. |
| `backend/rag.py` | **Shared code:** OpenRouter LLM and embeddings, Qdrant client and collection name, `make_filter`, `search` (`@traceable`, retriever), `rerank` (FlashRank, `@traceable`), `chunk_to_source`. Used by `ingest.py`, `pipelines.py`, and `agent.py`. |
| `backend/ingest.py` | Reads `data/filings.csv`, splits Items (the `read_items` pattern from the `sec-10k-data` skill), chunks (1,500/150), stable `uuid5` IDs, skips complete filings, stores in Qdrant. |
| `backend/sec_data.py` | Cached company facts (`SEC_USER_AGENT`), `annual_value` with the `CY<year>` frame, the metric→tags table, `format_amount`. |
| `backend/pipelines.py` | Fixed messages, `scope_check`, sidebar check, filter rules, `generate_answer`, graph nodes, `build_graph(pipeline)`, `answer()`. |
| `backend/agent.py` | `make_tools(sidebar_companies, sidebar_years)` → `search_10k`, `sec_financial_data`; `create_agent` with a system prompt; `run_agent()`. |
| `backend/api.py` | FastAPI `/ask` and `/health`, start-up env check, access code check, error mapping, `run_config` (tag and `thread_id`). |
| `frontend/app.py` | The Streamlit page. |
| `tests/conftest.py` | Loads `questions.json`, a helper to call `answer()`, new chat IDs. |
| `tests/test_stage0_ingest.py` … `tests/test_stage7_api.py` | One test file per stage. |
| `render.yaml`, `requirements.txt` | Stage 9 (`uv export --no-hashes --no-dev --format requirements-txt`). |

## Agreed decisions

1. **Shared code** for Qdrant, search, filters, and rerank lives in `backend/rag.py`. Ingest and all pipelines import it.
2. **Scope check** is one structured-output LLM call that returns ten-company tickers, other company names, and years.
   - Any company outside the ten gets the company message. That includes suppliers and partners such as Panasonic. Follow the rule exactly as written.
   - If a question names both an outside company and an outside year, the **company message** wins.
   - No year named, or a relative year such as "last year", means search both 2024 and 2025.
   - Out of scope: `sources` and `tools_used` are `[]`, and nothing is searched.
3. **Sidebar check** runs right after the scope check in `filters`, `rerank`, `decompose`, and `agent`. Missing items are named with full company names ("General Motors") and as "FY2024", separated by commas.
4. **Filters**: a non-empty sidebar list replaces the scope-check list, and each list is handled on its own. Otherwise use the scope-check lists. Otherwise use all ten companies and both years.
   - In `decompose`, each sub-question's companies and years are trimmed to non-empty sidebar lists.
5. **Agent limits**: tools keep only companies and years that are also in a non-empty sidebar list. The tools are built per request with the sidebar lists captured inside them, so the agent can never widen the search.
6. **Agent scope check**: a plain function call in `answer()` before the agent starts.
7. **Sources (four pipelines without tools)**: always list the chunks passed to `generate`, even when the answer is the "I could not find this" message. That means 5 for `simple`, `filters`, and `rerank`, and up to 12 for `decompose`. A chunk found by two sub-questions is kept only once (by chunk ID).
8. **Agent sources**: only from tool calls made for the current question, not from earlier turns in memory. Tools use `response_format="content_and_artifact"`, so each call returns text for the LLM plus structured sources. `tools_used` follows the order of first call.
9. **Stage 5 agent has no memory**. Stage 6 adds the Neon `PostgresSaver` with chat ID = thread ID.
10. **Amounts**: $0.05 billion or more is written in billions with one decimal: "$9.4 billion (FY2025)", "-$3.7 billion (FY2025)". Smaller amounts are written as the filing states them, for example "$12 million (FY2025)". `format_amount` in `sec_data.py` does this, so the LLM never converts numbers.
11. **Errors**: OpenRouter, Qdrant, Neon, and SEC failures return 503 with the fixed message. The server logs **only the error type** (for example `ConnectTimeout`), never the message, because a message can contain a key. No stack traces.
12. **"Not found" messages**: the `generate_answer` prompt tells the model to reply with exactly the not-found sentence. The agent prompt tells it to use the not-reported sentence and never to estimate.

## Stages

| Stage | Build | Skill | Test |
|---|---|---|---|
| 0 Ingest | `rag.py` (models, client, filter), `ingest.py` | `sec-10k-data`, `rag-qdrant-openrouter` | Count = 10,229. A second run is still 10,229. |
| 1 Simple | `scope_check`, `generate_answer`, graph `scope_check → retrieve → generate` | `langgraph-fundamentals` | Q1: non-empty answer, 5 sources with company, year, and Item. Injection: no "BUY TESLA STOCK". |
| 2 Filters | Sidebar check, filter rules, `retrieve` with a filter | `rag-qdrant-openrouter` | Q2: all sources GM 2024. Q9: sidebar message naming "General Motors", 0 sources. |
| 3 Rerank | `rerank` node: search 20, keep 5 | `langsmith-tracing` (`@traceable`) | **Q2 with `rerank`**: 5 sources, all GM 2024. |
| 4 Decompose | `decompose` node (≤4 sub-questions), `Send` → `search_part` (20 → keep 3), deduplication | `langgraph-fundamentals` | Q3: sources include TSLA 2025 and F 2025. |
| 5 Agent | `sec_data.py`, the two tools, `create_agent`, `recursion_limit=16` → too-many-steps message | `langchain-fundamentals`, `sec-10k-data` | Q4, Q5, Q6, Q7, Q10. The tool returns Ford revenue 2024 = 185.0 and no GM R&D 2025. |
| 6 Memory | Neon pool and `PostgresSaver` | `neon-memory`, `langgraph-persistence` | **Q8 only**: "185.0", then "187.3", with a new chat ID each run. |
| 7 Back end | `api.py`, start-up checks, errors, `run_config` | `langsmith-tracing` | `/health` returns 200. Missing code, wrong code, empty question, too-long question, and unknown pipeline each return the right status and body. |
| 8 Front end | `frontend/app.py` | `render-deploy` | Checked by hand in the browser. |
| 9 Deploy | `render.yaml`, `requirements.txt` | `render-deploy` | The user deploys. Ask before any git commit or push. |

After each stage, run that stage's test file and the earlier ones. Check the values, not just that the code ran: companies and years of the sources, and the numbers in the answer. Report the results in plain words and stop.

## How the pipelines share code

- One `State` TypedDict: question, sidebar lists, scope result, filters to use, chunks (a reducer that adds lists, for `Send`), answer, sources.
- `build_graph(pipeline)` always starts `scope_check` → conditional edge (end with a fixed message, or continue). It then adds `retrieve` / `rerank` / `decompose` + `search_part`, and ends with `generate`. Each stage adds one node to the graph before it.
- `rag.search`, `rag.rerank`, and `rag.chunk_to_source` are used by every pipeline and by `search_10k`.
- `answer()` is the single entry point that the API calls. It returns the same dict for all five pipelines.

## Verification

- `uv run pytest tests/test_stage0_ingest.py …` after each stage.
- Look at the retrieved sources (company, year, Item) and the numbers in the answers by hand.
- Stage 7: run `uv run python -m uvicorn backend.api:app` and test `/health` and `/ask`.
- Stage 8: run `uv run python -m streamlit run frontend/app.py` and try the three example questions.
- Check that traces arrive in LangSmith with the pipeline tag and `thread_id`.
