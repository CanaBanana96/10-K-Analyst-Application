"""The back end: a FastAPI web server with /health and /ask.

Start it with:  uv run python -m uvicorn backend.api:app
"""
import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.agent import close_pool
from backend.pipelines import answer

log = logging.getLogger("tenk")

REQUIRED_VARIABLES = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
                      "NEON_DATABASE_URL", "SEC_USER_AGENT", "ACCESS_CODE"]
# LangSmith is optional, but if its key is set it must not contain a line break either.
LINE_BREAK_CHECKED = REQUIRED_VARIABLES + ["LANGSMITH_API_KEY"]

PIPELINES = ("simple", "filters", "rerank", "decompose", "agent")
MAX_QUESTION_LENGTH = 1000
UNAVAILABLE_MESSAGE = "The answer service is not available right now. Try again in a minute."


def check_environment(env=None) -> list[str]:
    """Return a list of problems with the settings. It names variables, and never shows a value."""
    env = os.environ if env is None else env
    problems = [f"{name} is not set" for name in REQUIRED_VARIABLES if not env.get(name)]
    for name in LINE_BREAK_CHECKED:
        value = env.get(name) or ""
        # A key pasted with an extra line would fail later, and the library's error would print the key.
        if "\n" in value or "\r" in value:
            problems.append(f"{name} contains a line break")
    return problems


@asynccontextmanager
async def lifespan(app: FastAPI):
    problems = check_environment()
    if problems:
        for problem in problems:
            print(f"Start-up check failed: {problem}")
        raise SystemExit(1)
    yield
    close_pool()  # close the connections to Neon when the server stops


app = FastAPI(title="10-K Analyst", lifespan=lifespan)


class ApiError(Exception):
    """An error with a status code and a fixed message."""
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, error: ApiError):
    return JSONResponse(status_code=error.status, content={"error": error.message})


def require_access_code(x_access_code: str | None = Header(default=None)) -> None:
    """Runs before /ask. The header X-Access-Code must equal the ACCESS_CODE setting."""
    expected = os.environ.get("ACCESS_CODE", "")
    # compare_digest takes the same time whether the code is nearly right or wrong.
    if not expected or not x_access_code or not hmac.compare_digest(x_access_code.encode(), expected.encode()):
        raise ApiError(401, "Wrong or missing access code.")


class AskRequest(BaseModel):
    question: str
    chat_id: str
    pipeline: str
    companies: list[str] = []
    years: list[int] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}  # calls nothing, so it costs nothing


@app.post("/ask", dependencies=[Depends(require_access_code)])
def ask(request: AskRequest) -> dict:
    if not request.question.strip():
        raise ApiError(400, "The question is empty.")
    if len(request.question) > MAX_QUESTION_LENGTH:
        raise ApiError(400, "The question is too long. Use at most 1,000 characters.")
    if request.pipeline not in PIPELINES:
        raise ApiError(400, "Unknown pipeline. Use simple, filters, rerank, decompose, or agent.")
    try:
        return answer(request.question, request.pipeline, request.companies, request.years, request.chat_id)
    except Exception as error:
        # Log only the type of the error (for example ConnectTimeout). Its message could contain a key.
        log.error("Request failed: %s", type(error).__name__)
        raise ApiError(503, UNAVAILABLE_MESSAGE)
