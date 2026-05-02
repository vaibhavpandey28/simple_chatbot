# Simple Chatbot

A FastAPI-based chatbot with a custom HTML/CSS/JS UI, powered by Hugging Face Router (OpenAI-compatible API). It includes routing, tool-calling, SQL self-correction, episodic memory persistence, and Langfuse monitoring.

## Features

- Web chat UI at `/`
- REST API endpoint at `/chat`
- Hugging Face Router integration via OpenAI SDK
- Routing supervisor (`semantic_search` / `sql` / `direct_answer`)
- SQL self-correction loop (retry with repaired query)
- Episodic memory persisted in PostgreSQL
- Semantic search via Qdrant
- Langfuse monitoring (optional)
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
│  ├─ llm.py                   # Router + tool flow + SQL self-correction + memory injection
│  ├─ memory.py                # Episodic memory manager (PostgreSQL-backed)
│  ├─ monitoring.py            # Langfuse monitoring wrapper
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

- Tools:
  - `get_data_from_db(query: str)`
  - `semantic_search_products(query: str, limit: int = 5)`
  - `reindex_products_semantic()`
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

## Episodic Memory

- Each completed turn is stored as an episode in PostgreSQL:
  - `episodic_memory` table (summary + tags + full turn)
  - `chat_messages_memory` table (recent chat messages)
- On each new query, chatbot injects:
  - recent conversation memory
  - top relevant episodic memories (`EPISODIC_TOP_K`)
- Memory persists across server restarts.

## Langfuse Monitoring

- Enable with `.env`:
  - `LANGFUSE_ENABLED=true`
  - `LANGFUSE_PUBLIC_KEY=...`
  - `LANGFUSE_SECRET_KEY=...`
  - Optional `LANGFUSE_HOST` (default cloud)
- Captures trace per chat request and events for:
  - router decision
  - semantic search runs
  - SQL attempts and self-correction
  - final response/error

## What Is Done

- Routing supervisor for query decision (`semantic_search` / `sql` / `direct_answer`)
- SQL self-correction loop with retry attempts
- Schema-aware SQL repair guidance
- Semantic product search with Qdrant + reindex tool
- Episodic memory persistence in PostgreSQL
- Memory context injection into model prompt
- Langfuse instrumentation hooks for traces/events
- Multi-step execution loop (`plan -> act -> observe -> re-plan`)
- Multi-tool chaining in a single turn (step-wise loop)
- Completion critic pass before final answer
- SQL policy guardrails with risk scoring and fallback

## Pending (For Fully Agentic V2)

- Optional human approval checkpoint for risky actions
- Session persistence in frontend `localStorage` (stable `session_id` across refresh)
