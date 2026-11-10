"""Check your setup before class. It never prints a key.

Run it in this folder:  uv run python check_setup.py
You can run it after each account you create. Values you have not filled in yet are skipped.
"""
import json
import os
import re
import shutil
import subprocess
import urllib.request

from dotenv import load_dotenv

load_dotenv()
# Until the LangSmith key is filled in, do not try to send traces.
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"

results = []  # "ok", "fail", or "todo"


def report(name: str, ok: bool, detail: str) -> None:
    results.append("ok" if ok else "fail")
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: {detail}")


def not_yet(name: str, keys: list) -> bool:
    """Print a 'not filled in yet' line and return True if any of these .env values is empty."""
    empty = [k for k in keys if not os.environ.get(k)]
    if empty:
        results.append("todo")
        print(f"[ -- ] {name}: not filled in yet ({', '.join(empty)})")
    return bool(empty)


# 1. OpenRouter: one tiny model call. With LangSmith on, this call is also your first trace.
if not not_yet("OpenRouter", ["OPENROUTER_API_KEY"]):
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5"),
                         base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                         max_tokens=10, temperature=0)
        reply = llm.invoke("Reply with the single word READY.", config={"tags": ["setup-check"]}).content
        report("OpenRouter", "READY" in reply.upper(), f"the model replied: {reply.strip()[:30]}")
    except Exception as e:
        report("OpenRouter", False, f"{type(e).__name__}. Check OPENROUTER_API_KEY and your credit.")

# 2. Qdrant Cloud
if not not_yet("Qdrant", ["QDRANT_URL", "QDRANT_API_KEY"]):
    try:
        from qdrant_client import QdrantClient

        names = [c.name for c in QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                                              timeout=30).get_collections().collections]
        report("Qdrant", True, f"connected; {len(names)} collection(s)")
    except Exception as e:
        report("Qdrant", False, f"{type(e).__name__}. Check QDRANT_URL and QDRANT_API_KEY.")

# 3. Neon Postgres
if not not_yet("Neon", ["NEON_DATABASE_URL"]):
    try:
        import psycopg

        with psycopg.connect(os.environ["NEON_DATABASE_URL"], connect_timeout=30) as conn:
            version = conn.execute("select version()").fetchone()[0].split(" on ")[0]
        report("Neon", True, f"connected; {version}")
    except Exception as e:
        report("Neon", False, f"{type(e).__name__}. Check NEON_DATABASE_URL (the direct connection string).")

# 4. LangSmith: the OpenRouter call above was traced if tracing is on.
if not not_yet("LangSmith", ["LANGSMITH_API_KEY"]):
    if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
        report("LangSmith", True, "tracing is on; look for a run tagged 'setup-check' in project "
                                  f"'{os.environ.get('LANGSMITH_PROJECT', 'default')}'")
    else:
        report("LangSmith", False, "set LANGSMITH_TRACING=true in .env")

# 5. SEC financial data (free, but SEC wants your name and email)
agent = os.environ.get("SEC_USER_AGENT", "")
if "example.com" in agent:
    results.append("todo")
    print("[ -- ] SEC data: not filled in yet (put your own name and email in SEC_USER_AGENT)")
elif not not_yet("SEC data", ["SEC_USER_AGENT"]):
    try:
        req = urllib.request.Request("https://data.sec.gov/api/xbrl/companyfacts/CIK0000037996.json",
                                     headers={"User-Agent": agent})
        with urllib.request.urlopen(req, timeout=60) as r:
            name = json.load(r)["entityName"]
        report("SEC data", True, f"read the data for {name}")
    except Exception as e:
        report("SEC data", False, f"{type(e).__name__}. Check SEC_USER_AGENT (your name and email).")

# 6. The access code for your app
if not not_yet("Access code", ["ACCESS_CODE"]):
    report("Access code", True, "filled in")

# 7. The GitHub CLI, for deploying. It never prints the token.
if not shutil.which("gh"):
    report("GitHub CLI", False, "not installed. See the setup guide.")
else:
    status = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    user = re.search(r"(?:account|as) (\S+)", status.stdout + status.stderr)
    if status.returncode == 0:
        report("GitHub CLI", True, f"signed in as {user.group(1) if user else 'you'}")
    else:
        results.append("todo")
        print("[ -- ] GitHub CLI: not signed in yet (run gh auth login, then gh auth setup-git)")

print()
if "fail" in results:
    print("Fix the lines marked FAIL, then run this again.")
elif "todo" in results:
    print("So far so good. Fill in the lines marked --, then run this again.")
else:
    print("Everything is ready.")
