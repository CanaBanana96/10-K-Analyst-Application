# Instructions for Claude

This folder is a Python project called **10-K Analyst**. It answers questions about the 10-K filings of ten US auto companies. The user builds it by writing prompts to you. You write the code, run it, and test it.

## About the user

The user is a business analytics graduate student. They know some Python but are new to RAG, LangGraph, and web apps. Help them understand each step, and do not just finish the work for them. Explain things in short sentences and plain words. When you use a technical term for the first time, say what it means.

## The setup

- `requirements.md` is the specification. Build what it says, in the order of its stages.
- `data/` already contains the 20 filings and `data/filings.csv`. Do not download 10-Ks from SEC.
- Python packages are listed in `pyproject.toml` and installed with uv. Run Python with `uv run`, for example `uv run pytest`. Start servers as modules, with `uv run python -m uvicorn ...` and `uv run python -m streamlit run ...`, because Windows can block the `streamlit` program itself. Add a package only with `uv add`, and only if the user agrees.
- Keys are in the `.env` file. The code reads them from environment variables (load them with `python-dotenv`). You may not read `.env`.
- The skills in this folder explain how to use LangGraph, Qdrant, OpenRouter, Neon, LangSmith, SEC data, and Render in this project. Load the matching skill before you write that part of the code.
- The `docs-langchain` and `reference-langchain` tools search the current LangChain and LangGraph documentation. Use them when you are not sure about an API. Look for the Python pages, not JavaScript.

## Rules

1. Build in small steps. Build one stage of the plan, test it, and stop. Then tell the user what now works, which test question to try, and what the next stage adds.

2. Before you create or change a file, explain in one or two plain sentences what you are about to do and why. The user approves each change, so they need to understand it first.

3. Keep the code simple. Use few files, plain functions, and clear names. Add a short comment where a beginner would get stuck. Do not add features that `requirements.md` does not ask for.

4. Use this project's stack: LangGraph, Qdrant, and OpenRouter for both the LLM and the embeddings. Do not use FAISS, Chroma, or the OpenAI API directly.

5. Never ask for an API key or a password, never read `.env`, and never print or log a key or a connection string. If a key is missing, tell the user which line of `.env` to fill in.

6. Test after every stage with `uv run pytest`. Then check values, not just that something ran: look at the retrieved sources, the companies and years they come from, and the numbers in the answer. Report what you checked in plain words.

7. Answers from the app must come only from the 10-K text and the SEC financial data, with sources. If the sources do not contain the answer, the app says so. It never invents a number.

8. Ask before you commit to git, push to GitHub, or change anything outside this folder. Never commit `.env`.

9. If a fix fails three times, stop. Explain what you tried and what you think the problem is, and let the user decide what to do next.
