"""Stage 9: checks of render.yaml and requirements.txt. Nothing is deployed and nothing costs money."""
import re
import subprocess
from pathlib import Path

import yaml

from backend.api import LINE_BREAK_CHECKED, app

ROOT = Path(__file__).resolve().parent.parent
BLUEPRINT_TEXT = (ROOT / "render.yaml").read_text()
SERVICES = {s["name"]: s for s in yaml.safe_load(BLUEPRINT_TEXT)["services"]}
API, UI = SERVICES["tenk-analyst-api"], SERVICES["tenk-analyst-ui"]
OPTIONAL_SETTINGS = {"CHAT_MODEL"}  # the code has a default for these, so Render does not need them


def env_keys(service) -> set[str]:
    return {e["key"] for e in service["envVars"]}


# ---- render.yaml ----
def test_there_are_two_free_python_web_services():
    assert set(SERVICES) == {"tenk-analyst-api", "tenk-analyst-ui"}
    for service in SERVICES.values():
        assert service["type"] == "web" and service["plan"] == "free" and service["runtime"] == "python"
        assert service["buildCommand"] == "pip install -r requirements.txt"


def test_the_back_end_start_command_points_at_real_code():
    command = API["startCommand"]
    assert "backend.api:app" in command and "--host 0.0.0.0" in command and "--port $PORT" in command
    assert app.title == "10-K Analyst"          # the name after the colon really is the FastAPI app
    assert API["healthCheckPath"] == "/health"
    assert "/health" in {route.path for route in app.routes}


def test_the_front_end_start_command_points_at_real_code():
    command = UI["startCommand"]
    assert "frontend/app.py" in command and (ROOT / "frontend" / "app.py").exists()
    assert "--server.port $PORT" in command and "--server.address 0.0.0.0" in command
    assert "--server.headless true" in command


def test_the_back_end_declares_every_setting_the_start_up_check_needs():
    assert set(LINE_BREAK_CHECKED) <= env_keys(API)  # the six required ones, and LANGSMITH_API_KEY
    assert {"QDRANT_COLLECTION", "LANGSMITH_TRACING", "LANGSMITH_PROJECT"} <= env_keys(API)


def test_the_front_end_declares_its_settings_and_shares_the_access_code():
    assert {"API_URL", "ACCESS_CODE"} <= env_keys(UI)
    assert "ACCESS_CODE" in env_keys(API)


def test_every_setting_the_code_reads_is_declared_somewhere():
    read = set()
    for path in [*(ROOT / "backend").glob("*.py"), ROOT / "frontend" / "app.py"]:
        read |= set(re.findall(r'environ(?:\.get)?\(\s*"([A-Z_]+)"', path.read_text()))
        read |= set(re.findall(r'environ\["([A-Z_]+)"\]', path.read_text()))
    declared = env_keys(API) | env_keys(UI) | OPTIONAL_SETTINGS
    print("\nSettings read by the code:", sorted(read))
    assert read, "the scan found nothing, so the test itself is broken"
    assert read <= declared, f"not declared in render.yaml: {sorted(read - declared)}"


def test_no_secret_is_written_in_render_yaml():
    for service in SERVICES.values():
        for env in service["envVars"]:
            if env["key"] in ("PYTHON_VERSION", "LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
                assert "value" in env  # plain settings, safe to commit
            else:
                assert env.get("sync") is False and "value" not in env, f"{env['key']} must be typed in the dashboard"
    # Nothing shaped like a key or a connection string may appear anywhere in the file.
    for pattern in ["sk-", "postgres://", "postgresql://", "lsv2_", "ls__"]:
        assert pattern not in BLUEPRINT_TEXT, pattern


def test_the_python_version_fits_the_project():
    for service in SERVICES.values():
        version = next(e["value"] for e in service["envVars"] if e["key"] == "PYTHON_VERSION")
        assert re.fullmatch(r"3\.12\.\d+", version)  # pyproject.toml asks for >=3.12,<3.13


# ---- requirements.txt ----
def package_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith(("#", " "))]


def test_requirements_match_the_lock_file_and_leave_out_test_tools():
    fresh = subprocess.run(
        ["uv", "export", "--no-hashes", "--no-dev", "--format", "requirements-txt"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    committed = (ROOT / "requirements.txt").read_text()
    assert package_lines(committed) == package_lines(fresh), "requirements.txt is out of date: run uv export again"
    names = {re.split(r"[=<>;\[ ]", line)[0].lower() for line in package_lines(committed)}
    for needed in ["fastapi", "uvicorn", "streamlit", "langgraph", "langchain", "langchain-openai",
                   "langchain-qdrant", "qdrant-client", "flashrank", "psycopg", "psycopg-pool",
                   "langgraph-checkpoint-postgres", "httpx", "python-dotenv"]:
        assert needed in names, f"{needed} is missing from requirements.txt"
    assert "pytest" not in names
    assert not [line for line in package_lines(committed) if line.startswith("-e") or "file://" in line]
    assert all("==" in line for line in package_lines(committed))  # every package is pinned


def test_env_file_is_ignored_by_git():
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignored and ".env.*" in ignored
