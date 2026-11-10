"""The front end: a Streamlit page that talks to the back end.

Start it with:  uv run python -m streamlit run frontend/app.py

Streamlit runs this whole file again from top to bottom every time the user clicks something.
So anything that must survive (the chat, the chat ID) lives in st.session_state.
"""
import hmac
import os
import sys
import uuid
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

# `streamlit run frontend/app.py` only puts the frontend/ folder on the import path.
# Add the project folder, so we can reuse the number format from backend/sec_data.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.sec_data import format_amount  # noqa: E402

load_dotenv()

API_URL = (os.environ.get("API_URL") or "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 90
WAKING_MESSAGE = "The server is waking up. This takes about 30 seconds."
UNAVAILABLE_MESSAGE = "The answer service is not available right now. Try again in a minute."

COMPANIES = {"TSLA": "Tesla", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian", "LCID": "Lucid",
             "APTV": "Aptiv", "BWA": "BorgWarner", "ALV": "Autoliv", "HOG": "Harley-Davidson",
             "CVNA": "Carvana"}
YEARS = [2024, 2025]
PIPELINES = {  # the first one is the default
    "agent": "Agent: picks its own tools, reads exact SEC numbers, and remembers the chat.",
    "simple": "Simple: searches all filings and answers from the top 5 passages.",
    "filters": "Filters: searches only the companies and years in the question.",
    "rerank": "Rerank: searches 20 passages and keeps the best 5.",
    "decompose": "Decompose: splits the question into parts and searches each part.",
}
EXAMPLES = ["What risks does Tesla report about battery supply?",
            "Compare GM and Ford R&D in 2025",
            "How did Carvana's revenue change from 2024 to 2025?"]

st.set_page_config(page_title="10-K Analyst", page_icon="📈")


def safe(text: str) -> str:
    """Streamlit reads $...$ as a maths formula. Escape every $ so that amounts show as text."""
    return text.replace("$", "\\$")


def new_chat() -> None:
    st.session_state.chat_id = str(uuid.uuid4())  # a new ID means the agent starts with a fresh memory
    st.session_state.messages = []


# ---- 1. The access code screen ----
if not st.session_state.get("authed"):
    st.title("10-K Analyst")
    expected = os.environ.get("ACCESS_CODE", "")
    if not expected:
        st.error("ACCESS_CODE is not set. Add it to the .env file and restart the page.")
        st.stop()
    with st.form("login"):  # inside a form, pressing Enter submits it
        entered = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong access code.")
    st.stop()  # nothing below runs until the code is right

if "chat_id" not in st.session_state:
    new_chat()

# ---- 2. The sidebar ----
with st.sidebar:
    st.header("Filters")
    chosen_companies = st.multiselect(
        "Companies", options=list(COMPANIES), default=list(COMPANIES),
        format_func=lambda t: f"{COMPANIES[t]} ({t})",
        help="With everything (or nothing) selected, the app searches all ten companies.")
    chosen_years = st.multiselect("Fiscal years", options=YEARS, default=YEARS,
                                  help="With everything (or nothing) selected, the app searches both years.")
    show_sources = st.toggle("Show sources", value=True)
    st.button("New chat", on_click=new_chat)
    with st.expander("Learning mode"):
        pipeline = st.radio("How should the app answer?", options=list(PIPELINES), index=0,
                            format_func=lambda p: PIPELINES[p])

# The back end treats an empty list as "not set". So when everything is selected, send an empty list.
sent_companies = [] if len(chosen_companies) in (0, len(COMPANIES)) else chosen_companies
sent_years = [] if len(chosen_years) in (0, len(YEARS)) else chosen_years


# ---- 3. Showing a message ----
def show_source(source: dict) -> None:
    name = COMPANIES.get(source["company"], source["company"])
    if source["type"] == "sec":
        st.markdown(safe(f"**{name}, SEC data**: {source['metric']} is "
                         f"{format_amount(source['value'], source['year'])} (tag `{source['tag']}`)"))
    else:
        st.markdown(safe(f"**{name}, FY{source['year']} 10-K, Item {source['item']}**: "
                         f"{source['item_title']} ([SEC filing]({source['url']}))"))
        st.caption(safe(source["excerpt"] + "..."))


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        st.markdown(safe(message["content"]))
        if message["role"] != "assistant":
            return
        tools = message.get("tools_used", [])  # the tool labels are always shown
        st.markdown(" ".join(f":blue-badge[{t}]" for t in tools) if tools else ":gray-badge[no tools used]")
        sources = message.get("sources", [])
        if show_sources and sources:
            with st.expander(f"Sources ({len(sources)})"):
                for source in sources:
                    show_source(source)


# ---- 4. Asking the back end ----
def call_api(question: str) -> dict:
    """Send one question. Returns the chat message to show (an answer, or an error)."""
    payload = {"question": question, "chat_id": st.session_state.chat_id, "pipeline": pipeline,
               "companies": sent_companies, "years": sent_years}
    headers = {"X-Access-Code": os.environ.get("ACCESS_CODE", "")}
    response = None
    for attempt in (1, 2):  # a sleeping free server needs one more try
        try:
            response = httpx.post(f"{API_URL}/ask", json=payload, headers=headers, timeout=TIMEOUT_SECONDS)
            break
        except httpx.TransportError:  # a timeout or a failed connection
            if attempt == 1:
                st.info(WAKING_MESSAGE)
    if response is None:
        return {"role": "assistant", "content": UNAVAILABLE_MESSAGE, "error": True}
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.status_code != 200:
        return {"role": "assistant", "content": data.get("error") or UNAVAILABLE_MESSAGE, "error": True}
    return {"role": "assistant", "content": data["answer"], "tools_used": data["tools_used"],
            "sources": data["sources"]}


# ---- 5. The page ----
st.title("10-K Analyst")
st.caption("Ask about the 10-K filings of ten US auto companies, fiscal years 2024 and 2025.")

for message in st.session_state.messages:
    show_message(message)


def pick_example(text: str) -> None:
    st.session_state.example = text


if not st.session_state.messages:  # the examples disappear once the chat has started
    for column, text in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        column.button(text, on_click=pick_example, args=(text,), use_container_width=True)

question = st.chat_input("Ask a question about the filings")
question = question or st.session_state.pop("example", None)

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message)
    with st.chat_message("assistant"):
        with st.spinner("Searching the filings..."):
            reply = call_api(question)
    st.session_state.messages.append(reply)
    st.rerun()  # draw the whole chat again, so the new answer looks like the others
