# Simple Chatbot

A FastAPI-based chatbot with a custom HTML/CSS/JS UI, powered by Hugging Face Router (OpenAI-compatible API). It supports session memory and optional database tool-calling.

## Features

- Web chat UI at `/`
- REST API endpoint at `/chat`
- Hugging Face Router integration via OpenAI SDK
- Per-session in-memory conversation history
- Optional tool-calling for PostgreSQL queries
- Structured, color-capable logging

## Tech Stack

- Python 3.12+
- FastAPI + Uvicorn
- OpenAI Python SDK (`base_url=https://router.huggingface.co/v1`)
- PostgreSQL (`psycopg2`) for tool-backed DB queries
- Vanilla HTML/CSS/JavaScript frontend

## Project Structure

```text
simple_chatbot/
├─ main.py                     # FastAPI app + routes
├─ core/
│  └─ logger.py                # Central logger config
├─ service/
│  ├─ llm.py                   # HF Router chat + tool flow
│  ├─ tools.py                 # Tool wrappers
│  └─ db.py                    # PostgreSQL query execution
├─ static/
│  ├─ index.html               # Chat UI
│  ├─ styles.css               # UI styles
│  └─ app.js                   # UI behavior
├─ .env-example
├─ pyproject.toml
└─ README.md
```

## Prerequisites

1. Python `3.12+`
2. A Hugging Face token with inference access
3. PostgreSQL running locally (for tool-calling), default expected:
   - host: `localhost`
   - port: `5432`
   - db: `csv_db`
   - user: `csv_user`
   - password: `csv_pass`

## Setup

```bash
cd /Users/vaibhavpandey/Documents/projects/simple_chatbot
cp .env-example .env
```

Edit `.env` and set at least:

```env
HF_MODEL=openai/gpt-oss-20b:groq
HF_TOKEN=hf_your_real_token_here
```

Install dependencies:

```bash
uv sync
```

## Run

```bash
uv run uvicorn main:app --reload
```

Open in browser:

- UI: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`

## API Usage

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Chat

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session-1",
    "user_input": "Show top 5 customers by order count"
  }'
```

## Environment Variables

### LLM

- `HF_MODEL` default: `openai/gpt-oss-20b:groq`
- `HF_TOKEN` required
- `HF_TEMPERATURE` default: `0.2`
- `HF_TOP_P` default: `0.9`
- `MAX_HISTORY_MESSAGES` default: `16`
- `EPISODIC_TOP_K` default: `3` (how many relevant episodes are injected)
- `ENABLE_TOOLS` default: `true`

### Logging

- `LOG_LEVEL` default: `INFO`
- `LOGGER` default: `true`
- `LOGGER_COLOR` default: `true`

## Tool-Calling Notes

- Current tool: `get_data_from_db(query: str)`
- Intended for safe `SELECT` queries
- Tool output is JSON-serialized with datetime-safe conversion
- If model/provider does not support tools, app retries without tools

## Troubleshooting

- `HF_TOKEN is missing`: set token in `.env` and restart server.
- `does not support tools`: set `ENABLE_TOOLS=false` or use a tool-capable model/provider route.
- DB errors: verify PostgreSQL credentials in `service/db.py` and ensure database is running.

## Security

- Never commit real `HF_TOKEN` to git.
- If token is exposed, rotate it immediately in Hugging Face settings.


## Semantic Search Upgrade

This project now supports semantic product search through Qdrant.

### What changed

- New tool: `semantic_search_products(query, limit=5)`
- New tool: `reindex_products_semantic()`
- Product vectors are indexed from PostgreSQL `products` table into Qdrant collection `products`
- DB and Qdrant are now configurable using `.env`

### First-time setup for semantic search

1. Ensure PostgreSQL and Qdrant are running.
2. Update `.env` with DB and Qdrant variables from `.env-example`.
3. Install deps: `uv sync`
4. Start app: `uv run uvicorn main:app --reload`
5. Ask a product intent query, e.g. `show aesthetic shirts for summer`.

The first semantic query auto-indexes products if the Qdrant collection is empty.

Manual indexing command:

```bash
uv run python scripts/index_products_qdrant.py --recreate
```

## Episodic Memory (LangChain + LangGraph)

- Memory graph uses `langgraph` `StateGraph(MessagesState)` + `MemorySaver` checkpointer.
- Each completed turn is stored as an episode: timestamp, user input, assistant reply, summary, tags.
- On each new query, chatbot injects:
  - recent conversation memory
  - top relevant episodic memories (`EPISODIC_TOP_K`)
- SQL self-correction still runs independently for DB-query retries.
