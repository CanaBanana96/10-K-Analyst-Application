---
name: neon-memory
description: Use when adding conversation memory to the LangGraph agent with Neon Postgres - the connection pool, the PostgresSaver checkpointer, thread IDs for chats, and "New chat". Read together with the langgraph-persistence skill.
---

# Conversation memory in Neon Postgres

The agent saves each conversation in Neon with LangGraph's checkpointer. A conversation is identified by its **thread ID**. "New chat" in the front end simply means a new thread ID.

The connection string comes from the environment variable `NEON_DATABASE_URL`. Use the direct connection string (its host does not contain `-pooler`).

## The connection pool and the checkpointer

A web server needs a pool of connections. Neon pauses the database when it is idle, so the pool must check a connection before it uses it.

```python
import os
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo=os.environ["NEON_DATABASE_URL"],
    max_size=5,
    check=ConnectionPool.check_connection,        # replaces connections that Neon closed while idle
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)
checkpointer = PostgresSaver(pool)
checkpointer.setup()                               # creates the tables once; safe to run again
```

Create the pool and the checkpointer once, when the back end starts, not on every request. Close the pool when the back end stops.

## Using it

Pass the checkpointer when you build the agent or graph. Pass the thread ID with every call.

```python
from langchain.agents import create_agent

agent = create_agent(llm, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": chat_id}}
result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
```

Send only the new question. The checkpointer adds the earlier messages of the same thread.

## Good to know

- The first request after Neon has been idle takes about one second longer.
- Memory lives in the database, so a conversation survives a restart of the back end.
- Never print or log the connection string. It contains the password.
