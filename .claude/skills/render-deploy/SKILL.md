---
name: render-deploy
description: Use when preparing this project to run on Render - the render.yaml file with two web services (FastAPI back end and Streamlit front end), start commands, environment variables, the access code, and handling the free plan's sleep and wake-up.
---

# Deploying to Render

The app runs on Render as two free web services built from the same GitHub repository:

1. `<name>-api`: the FastAPI back end.
2. `<name>-ui`: the Streamlit front end. It calls the back end's public URL.

The student creates the services in the Render dashboard with **New, then Blueprint**, and picks their GitHub repository. Render reads `render.yaml` from the repository. The student types each secret into the dashboard. You never handle a key.

## Files Render needs

- `requirements.txt` with every package the app needs. Make it from the project with `uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt`, and commit it.
- `render.yaml` at the top of the repository.
- The `data/` folder is not needed on Render. The vectors are already in Qdrant.

## render.yaml

```yaml
services:
  - type: web
    name: tenk-analyst-api
    runtime: python
    plan: free
    region: ohio                 # close to Neon (AWS US East 2)
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.api:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: "3.12.8"
      - key: OPENROUTER_API_KEY
        sync: false              # sync: false means the student types the value in the dashboard
      - key: QDRANT_URL
        sync: false
      - key: QDRANT_API_KEY
        sync: false
      - key: QDRANT_COLLECTION
        sync: false              # the same collection name as in .env (default tenk)
      - key: NEON_DATABASE_URL
        sync: false
      - key: SEC_USER_AGENT
        sync: false
      - key: LANGSMITH_TRACING
        value: "true"
      - key: LANGSMITH_API_KEY
        sync: false              # a "Tracing Only" key: it can send traces but cannot read them
      - key: LANGSMITH_PROJECT
        value: tenk-analyst
      - key: ACCESS_CODE
        sync: false              # the same value as on the front end
  - type: web
    name: tenk-analyst-ui
    runtime: python
    plan: free
    region: ohio
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
    envVars:
      - key: PYTHON_VERSION
        value: "3.12.8"
      - key: API_URL
        sync: false              # the back end's URL, for example https://tenk-analyst-api.onrender.com
      - key: ACCESS_CODE
        sync: false
```

Adjust the module paths in the start commands to match the project's folders.

## The access code

The front end asks for the access code before it shows the chat. It compares the entry with its own `ACCESS_CODE` environment variable. Keep the result in `st.session_state`, so the user enters it once per browser session. The front end sends the code in the `X-Access-Code` header with every `/ask`.

The back end must check that header against its own `ACCESS_CODE` and return status 401 when it is wrong or missing. Otherwise anyone who finds the back-end URL could spend the student's OpenRouter credit. Both services therefore need `ACCESS_CODE`, with the same value.

## Sleep and wake-up

Free services sleep after a period without visitors. The first request then takes about 30 seconds.

- The back end has a quick `GET /health` endpoint that returns `{"status": "ok"}`.
- The front end calls the back end with a timeout of at least 90 seconds. If the call times out or the connection fails, it shows "The server is waking up. This takes about 30 seconds." and tries again once.

## Good to know

- Each free service has 512 MB of memory. This app fits, because embeddings run on OpenRouter and the reranker model is 3 MB.
- Render does not need the `.env` file. Never commit `.env`.
- A deploy takes a few minutes. Follow it in the service's Logs tab.
