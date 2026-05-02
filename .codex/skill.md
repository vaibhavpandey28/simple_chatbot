# Codex Execution Skill: simple_chatbot

Use this file as the default execution policy for Codex work in this repository.

## Mission

Deliver safe, verifiable improvements quickly for:
- Routing quality (`semantic_search`, `sql`, `direct_answer`)
- Tool-calling reliability
- SQL safety/self-correction
- Episodic memory quality
- API and frontend chat behavior

## Fast Context

Primary files:
- `service/llm.py` (core orchestration)
- `service/tools.py` (tool interfaces)
- `service/db.py` (SQL execution)
- `service/semantic_search.py` (Qdrant retrieval/index)
- `service/memory.py` (episode capture/retrieval)
- `service/monitoring.py` (Langfuse events)
- `main.py` and `static/*` (API/UI)

## Default Commands

Setup:
```bash
cp .env-example .env
uv sync
```

Run app:
```bash
uv run uvicorn main:app --reload
```

Syntax check:
```bash
uv run python -m py_compile main.py service/*.py
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Basic chat check:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"codex-smoke","user_input":"hello"}'
```

Reindex semantic search:
```bash
uv run python scripts/index_products_qdrant.py --recreate
```

## Test Matrix (Minimum Per Behavior Change)

Run these 3 requests after editing routing/tools logic:

1. Semantic intent:
```json
{"session_id":"t1","user_input":"suggest aesthetic shirts for summer"}
```
Expected: route uses semantic search or graceful fallback if index/data unavailable.

2. Structured SQL intent:
```json
{"session_id":"t2","user_input":"show black shoes under 3000 in stock"}
```
Expected: SQL path or safe correction path; never unsafe SQL statements.

3. Direct answer intent:
```json
{"session_id":"t3","user_input":"what can you do"}
```
Expected: no forced product tool call unless needed.

## Playbooks

### A) Router misclassification

- Inspect router prompt in `service/llm.py` (`ROUTER_SYSTEM_PROMPT`).
- Tighten route rules with explicit examples for price/category/stock constraints.
- Keep output JSON schema unchanged.
- Re-run full test matrix.

### B) SQL tool errors

- Confirm only `SELECT` queries pass execution layer.
- Ensure correction prompt references only real schema columns.
- Add/adjust defensive parsing for malformed model JSON.
- Validate constrained query test again.

### C) Empty semantic search results

- Confirm index exists and embeddings pipeline is healthy.
- Improve fallback response with category hints + rephrase suggestions.
- Optionally trigger `reindex_products_semantic` flow.

### D) Memory quality issues

- Verify episode write/read paths in `service/memory.py`.
- Tune retrieval count (`EPISODIC_TOP_K`) conservatively.
- Avoid injecting noisy memory blocks when irrelevant.

### E) Monitoring gaps

- Ensure each chat request has trace lifecycle start/end.
- Add events for route decision, tool call, correction attempt, final response.
- Do not log secrets.

## Constraints

- Never commit real tokens/keys.
- Preserve JSON contracts used by planner/router/critic parsing.
- Keep SQL read-only and guarded.
- Prefer small, reviewable diffs over broad refactors.
- Update `README.md` when behavior changes are user-visible.

## Done Criteria

- App starts.
- `/health` is OK.
- Test matrix passes.
- Edited files compile.
- No regression in tool fallback behavior.
