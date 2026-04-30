# Simple Chatbot

A lightweight local chatbot built with FastAPI, Ollama, and a custom HTML/CSS/JS chat UI.

## Features

- FastAPI backend with `/chat` endpoint
- Browser chat UI served at `/`
- Local LLM inference via Ollama (OpenAI-compatible API)
- Session-based in-memory conversation history
- Configurable logging with optional colors

## Tech Stack

- Python 3.12+
- FastAPI + Uvicorn
- Ollama (local model runtime)
- OpenAI Python SDK (pointing to Ollama base URL)

## Project Structure

```text
simple_chatbot/
├─ main.py                 # FastAPI app + routes
├─ service/
│  └─ llm.py               # LLM client + chat session logic
├─ core/
│  └─ logger.py            # Central logger config
├─ static/
│  ├─ index.html           # Chat UI
│  ├─ styles.css           # UI styles
│  └─ app.js               # UI interactions
├─ pyproject.toml
└─ README.md
```

## Prerequisites

1. Install and run Ollama
2. Pull at least one model (example):

```bash
ollama pull tinyllama:latest
```

3. Make sure Ollama is reachable at `http://localhost:11434`

## Run Locally

```bash
cd /Users/username/Documents/projects/simple_chatbot
uv sync
uv run uvicorn main:app --reload
```

Open:

- UI: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`

## API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Chat

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session-1",
    "user_input": "Hello, how are you?"
  }'
```

## Environment Variables

Set in `.env` (or export in shell):

- `OLLAMA_MODEL` (default: `tinyllama:latest`)
- `LOG_LEVEL` (default: `INFO`)
- `LOGGER` (`true`/`false`) to enable/disable logs
- `LOGGER_COLOR` (`true`/`false`) for colored logs

Example:

```env
OLLAMA_MODEL=tinyllama:latest
LOG_LEVEL=INFO
LOGGER=true
LOGGER_COLOR=true
```

## Notes

- Chat history is stored in memory and resets when server restarts.
- If a model fails due to memory limits, switch to a smaller model (like `tinyllama:latest`).
