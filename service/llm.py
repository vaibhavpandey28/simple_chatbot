import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from core.logger import get_logger
from service.memory import EpisodicMemoryManager
from service.monitoring import LangfuseMonitor
from service.tools import (
    get_data_from_db,
    reindex_products_semantic,
    semantic_search_products,
)

load_dotenv()
logger = get_logger(__name__)

# Environment-driven runtime settings.
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("HF_MODEL", "openai/gpt-oss-20b:groq")
TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("HF_TOP_P", "0.9"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "16"))
ENABLE_TOOLS = os.getenv("ENABLE_TOOLS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EPISODIC_TOP_K = int(os.getenv("EPISODIC_TOP_K", "3"))


DEFAULT_SYSTEM_PROMPT = """
You are Vaibhav's personal AI assistant.
- Be concise and practical.
- Avoid unnecessary explanations.
- Stay on-topic and answer only what the user asked.
- Do not invent names, stories, or context.
- If unsure, say so briefly and ask for clarification.
- For product recommendations, prefer semantic_search_products tool before raw SQL.
- If semantic search returns no matches, provide concrete fallback help:
  include likely categories from the database and suggest 2-3 specific rephrases.
"""

ROUTER_SYSTEM_PROMPT = """
You are a routing supervisor for an ecommerce assistant.
Select exactly one route for the user query:
- semantic_search: Use for fuzzy discovery, style/taste intent, broad recommendations.
- sql: Use for explicit structured constraints such as size, color, price limits/ranges, stock checks, exact category filters.
- direct_answer: Use for non-product chat or when no tool is needed.

Return ONLY valid JSON with shape:
{
  "route": "semantic_search" | "sql" | "direct_answer",
  "reason": "short reason",
  "semantic_query": "string",
  "sql_query": "SELECT ..." 
}

Rules:
- If route != semantic_search, semantic_query can be empty string.
- If route != sql, sql_query can be empty string.
- For sql route, output a SAFE SELECT query only (never INSERT/UPDATE/DELETE/DROP).
- For product lookups, prefer selecting: product_id, product_name, description, category, price, stock_qty.
- Detect constraints like size/color/under $X/between prices and route to sql.
"""

SQL_SELF_CORRECTION_PROMPT = """
You are a SQL self-correction layer for a PostgreSQL ecommerce assistant.
Fix failed SQL queries using the user intent, SQL error, and available schema.

Return ONLY valid JSON:
{
  "fixed_sql": "SELECT ...",
  "reason": "short reason"
}

Rules:
- Output only SAFE SELECT SQL.
- Never use columns outside available_columns.
- Keep selected columns: product_id, product_name, description, category, price, stock_qty.
- Preserve user constraints where possible (price/category/stock/etc).
"""

PLANNER_SYSTEM_PROMPT = """
You are an execution planner in a plan-act-observe-replan loop.
Given the user_input and current observation, choose exactly one next action.

Return ONLY JSON:
{
  "action": "semantic_search" | "sql" | "respond",
  "reason": "short reason",
  "semantic_query": "string",
  "sql_query": "SELECT ...",
  "response_hint": "string"
}

Rules:
- Use semantic_search for broad/fuzzy product discovery.
- Use sql for exact filters, ranges, stock, pricing constraints.
- Use respond only when enough evidence is available.
- If action != semantic_search then semantic_query can be empty.
- If action != sql then sql_query can be empty.
"""

CRITIC_SYSTEM_PROMPT = """
You are a completion critic.
Check whether the draft answer fully addresses the user query and matches tool observations.
If good, return {"needs_revision": false, "improved_answer": ""}.
If weak/incomplete, return {"needs_revision": true, "improved_answer": "...better final answer..."}.
Return ONLY JSON.
"""


# API client (Hugging Face Router speaks OpenAI-compatible format).
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN or None,
)

# Stores chat history in memory: {session_id: [messages...]}
chat_sessions: dict[str, list[dict[str, str]]] = {}
_products_columns_cache: list[str] | None = None
episodic_memory = EpisodicMemoryManager()
monitor = LangfuseMonitor()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search_products",
            "description": "Semantic search over products using meaning-based matching. Use this first for product discovery queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language product intent, e.g. aesthetic shirt for summer.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of product matches to return.",
                        "default": 5,
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reindex_products_semantic",
            "description": "Rebuild the semantic index from current PostgreSQL products table.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_from_db",
            "description": "Execute a SELECT SQL query on the PostgreSQL ecommerce database. Use when exact SQL-style data is requested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A safe SELECT SQL query to execute.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _trim_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only recent messages so history does not grow forever."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    logger.info(
        "Trimming history from %s to %s messages",
        len(messages),
        MAX_HISTORY_MESSAGES,
    )
    return [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1) :]


def _completion(messages: list[dict[str, str]], use_tools: bool):
    """Send one request to the model, with tools if enabled."""
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if use_tools:
        kwargs["tools"] = TOOLS
    return client.chat.completions.create(**kwargs)


def _inject_memory_context(
    session_id: str, user_input: str, messages: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Inject relevant episodic memory as a temporary system message."""
    memory_context = episodic_memory.build_memory_context(
        session_id=session_id, query=user_input, top_k=EPISODIC_TOP_K
    )
    if not memory_context:
        return list(messages)

    enriched = list(messages)
    enriched.append(
        {
            "role": "system",
            "content": (
                "Use this memory context only when relevant. "
                "Do not reveal internal memory mechanics.\n"
                f"{memory_context}"
            ),
        }
    )
    return enriched


def _get_or_create_session(session_id: str) -> list[dict[str, str]]:
    """Create a new session with system prompt, or return existing one."""
    if session_id not in chat_sessions:
        logger.info("Creating new session: %s", session_id)
        chat_sessions[session_id] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    return chat_sessions[session_id]


def _parse_tool_args(raw_args: str) -> dict[str, Any]:
    """Parse tool-call arguments from JSON; return empty dict if invalid."""
    try:
        return json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        logger.warning("Invalid tool arguments JSON: %s", raw_args)
        return {}


def _escape_sql_text(value: str) -> str:
    """Escape single quotes for SQL string literals."""
    return value.replace("'", "''")


def _fallback_category_hints() -> list[str]:
    """Get common categories to help users rephrase failed searches."""
    rows = get_data_from_db(
        """
        SELECT category, COUNT(*) AS total
        FROM products
        WHERE category IS NOT NULL AND category <> ''
        GROUP BY category
        ORDER BY total DESC, category ASC
        LIMIT 8
        """
    )
    if isinstance(rows, dict) and rows.get("error"):
        return []
    return [str(row.get("category")) for row in rows if row.get("category")]


def _fallback_shirt_examples(user_query: str, limit: int = 5) -> list[dict[str, Any]]:
    """If semantic search is empty, try a simple text match for shirts."""
    query = _escape_sql_text(user_query)
    rows = get_data_from_db(
        f"""
        SELECT product_id, product_name, description, category, price, stock_qty
        FROM products
        WHERE
          LOWER(COALESCE(product_name, '')) LIKE '%shirt%'
          OR LOWER(COALESCE(category, '')) LIKE '%shirt%'
          OR LOWER(COALESCE(description, '')) LIKE '%shirt%'
        ORDER BY
          CASE
            WHEN LOWER(COALESCE(product_name, '')) LIKE '%{query.lower()}%' THEN 0
            WHEN LOWER(COALESCE(description, '')) LIKE '%{query.lower()}%' THEN 1
            ELSE 2
          END,
          product_id ASC
        LIMIT {max(1, min(limit, 10))}
        """
    )
    if isinstance(rows, dict) and rows.get("error"):
        return []
    return rows


def _execute_tool(function_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one supported tool and return JSON-safe output."""
    if function_name == "get_data_from_db":
        query = args.get("query")
        if not query:
            return {"error": "Missing required argument: query"}
        logger.info("Executing tool get_data_from_db")
        return {"result": get_data_from_db(query=query)}

    if function_name == "semantic_search_products":
        query = args.get("query")
        if not query:
            return {"error": "Missing required argument: query"}
        limit = int(args.get("limit", 5))
        logger.info("Executing tool semantic_search_products")
        result = semantic_search_products(query=query, limit=limit)
        payload: dict[str, Any] = {"result": result}
        if not result:
            payload["hints"] = {
                "available_categories": _fallback_category_hints(),
                "shirt_examples": _fallback_shirt_examples(query, limit=limit),
                "retry_tip": "Try style + color + category, e.g. 'minimal white linen shirt for summer'.",
            }
        return payload

    if function_name == "reindex_products_semantic":
        logger.info("Executing tool reindex_products_semantic")
        return {"result": reindex_products_semantic()}

    logger.warning("Unsupported tool requested: %s", function_name)
    return {"error": f"Unsupported tool: {function_name}"}


def _get_products_columns() -> list[str]:
    """Read products table columns once and cache them for router guidance."""
    global _products_columns_cache
    if _products_columns_cache is not None:
        return _products_columns_cache

    rows = get_data_from_db(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'products'
        ORDER BY ordinal_position
        """
    )
    if isinstance(rows, dict) and rows.get("error"):
        logger.warning("Could not fetch products columns: %s", rows["error"])
        _products_columns_cache = [
            "product_id",
            "product_name",
            "description",
            "category",
            "price",
            "stock_qty",
        ]
        return _products_columns_cache

    _products_columns_cache = [
        str(row.get("column_name")) for row in rows if row.get("column_name")
    ]
    if not _products_columns_cache:
        _products_columns_cache = [
            "product_id",
            "product_name",
            "description",
            "category",
            "price",
            "stock_qty",
        ]
    return _products_columns_cache


def _fallback_sql_for_user_query(user_input: str) -> str:
    """Generate a guaranteed-safe SQL fallback that only uses known baseline columns."""
    safe_q = _escape_sql_text(user_input.lower())
    return f"""
    SELECT product_id, product_name, description, category, price, stock_qty
    FROM products
    WHERE
      LOWER(COALESCE(product_name, '')) LIKE '%{safe_q}%'
      OR LOWER(COALESCE(description, '')) LIKE '%{safe_q}%'
      OR LOWER(COALESCE(category, '')) LIKE '%{safe_q}%'
    ORDER BY price ASC, product_id ASC
    LIMIT 10
    """.strip()


def _is_safe_select_sql(sql: str) -> bool:
    text = (sql or "").strip().lower()
    if not text.startswith("select"):
        return False
    banned = ("insert", "update", "delete", "drop", "alter", "truncate")
    return not any(re.search(rf"\b{kw}\b", text) for kw in banned)


def _assess_sql_risk(sql: str) -> dict[str, Any]:
    text = (sql or "").strip().lower()
    risk_score = 0
    reasons: list[str] = []
    if not text.startswith("select"):
        risk_score += 90
        reasons.append("not_select")
    if " from products " not in f" {text} ":
        risk_score += 30
        reasons.append("non_products_table")
    if any(kw in text for kw in [" join ", " union ", ";", "--", " pg_", " information_schema"]):
        risk_score += 40
        reasons.append("suspicious_pattern")
    if " limit " not in f" {text} ":
        risk_score += 20
        reasons.append("missing_limit")
    return {"risk_score": min(100, risk_score), "reasons": reasons}


def _guard_sql_policy(sql: str) -> dict[str, Any]:
    risk = _assess_sql_risk(sql)
    if not _is_safe_select_sql(sql):
        return {"allowed": False, "risk": risk, "blocked_reason": "unsafe_sql"}
    if risk["risk_score"] >= 70:
        return {"allowed": False, "risk": risk, "blocked_reason": "high_risk_sql"}
    return {"allowed": True, "risk": risk, "blocked_reason": ""}


def _try_execute_text_tool_call(raw_text: str) -> dict[str, Any] | None:
    """
    Some models return a JSON tool-call in plain text instead of `tool_calls`.
    Example: {"function":"semantic_search_products","arguments":{...}}
    """
    text = (raw_text or "").strip()
    if not text.startswith("{") or '"function"' not in text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    function_name = payload.get("function")
    args = payload.get("arguments", {})
    if not isinstance(function_name, str) or not isinstance(args, dict):
        return None

    logger.info("Executing text-based tool call: %s args=%s", function_name, args)
    return _execute_tool(function_name, args)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _repair_sql_query(
    user_input: str,
    failed_sql: str,
    error_text: str,
    available_columns: list[str],
) -> tuple[str | None, str]:
    """Use LLM to repair SQL after a DB error."""
    repair_messages = [
        {"role": "system", "content": SQL_SELF_CORRECTION_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_input": user_input,
                    "failed_sql": failed_sql,
                    "error": error_text,
                    "available_columns": available_columns,
                }
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=repair_messages,
            temperature=0,
            top_p=1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _extract_first_json_object(raw) or {}
    except Exception:
        logger.exception("SQL repair step failed")
        return None, "repair_call_failed"

    fixed_sql = str(parsed.get("fixed_sql", "")).strip()
    reason = str(parsed.get("reason", "")).strip() or "no_reason"
    if not _is_safe_select_sql(fixed_sql):
        return None, f"unsafe_or_invalid_sql:{reason}"
    return fixed_sql, reason


def _run_sql_with_self_correction(
    user_input: str,
    initial_sql: str,
    max_attempts: int = 3,
    trace: Any = None,
) -> dict[str, Any]:
    """
    Self-correction loop:
    1) Run SQL
    2) On error, inspect error + schema
    3) Repair SQL and retry
    """
    available_columns = _get_products_columns()
    sql = initial_sql if _is_safe_select_sql(initial_sql) else _fallback_sql_for_user_query(user_input)
    attempts: list[dict[str, Any]] = []
    last_run: dict[str, Any] = {"result": {"error": "No SQL executed"}}

    for attempt in range(1, max_attempts + 1):
        last_run = _execute_tool("get_data_from_db", {"query": sql})
        result = last_run.get("result")
        error_text = result.get("error") if isinstance(result, dict) else None

        attempts.append({"attempt": attempt, "sql": sql, "error": error_text})
        monitor.event(
            trace,
            name="sql_attempt",
            metadata={"attempt": attempt, "sql": sql},
            output={"error": error_text},
        )
        if not error_text:
            return {
                "tool_result": last_run,
                "final_sql": sql,
                "attempts": attempts,
                "self_corrected": attempt > 1,
            }

        logger.warning("SQL attempt %s failed: %s", attempt, error_text)
        if attempt == max_attempts:
            break

        fixed_sql, reason = _repair_sql_query(
            user_input=user_input,
            failed_sql=sql,
            error_text=str(error_text),
            available_columns=available_columns,
        )
        if not fixed_sql:
            logger.warning("Repair failed (%s). Using baseline fallback SQL.", reason)
            fixed_sql = _fallback_sql_for_user_query(user_input)
        sql = fixed_sql

    return {
        "tool_result": last_run,
        "final_sql": sql,
        "attempts": attempts,
        "self_corrected": len(attempts) > 1,
    }


def _route_user_query(user_input: str) -> dict[str, Any]:
    """Supervisor step that routes user intent to semantic search, SQL, or direct answer."""
    lowered = user_input.lower()
    has_structured_filters = any(
        token in lowered
        for token in ["size", "under", "below", "less than", "price", "stock", "in stock", "between", "color"]
    )

    available_columns = ", ".join(_get_products_columns())
    router_messages = [
        {
            "role": "system",
            "content": (
                f"{ROUTER_SYSTEM_PROMPT}\n"
                f"Available products columns: {available_columns}\n"
                "Never reference columns not present in this list."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    try:
        route_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=router_messages,
            temperature=0,
            top_p=1,
        )
        raw = (route_resp.choices[0].message.content or "").strip()
        parsed = _extract_first_json_object(raw) or {}
    except Exception:
        logger.exception("Router step failed, falling back to heuristic route")
        parsed = {}

    route = str(parsed.get("route", "")).strip().lower()
    semantic_query = str(parsed.get("semantic_query", user_input)).strip() or user_input
    sql_query = str(parsed.get("sql_query", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    if route not in {"semantic_search", "sql", "direct_answer"}:
        route = "sql" if has_structured_filters else "semantic_search"

    if route == "sql":
        if not sql_query or not sql_query.lower().startswith("select"):
            sql_query = _fallback_sql_for_user_query(user_input)

    return {
        "route": route,
        "reason": reason,
        "semantic_query": semantic_query,
        "sql_query": sql_query,
    }


def _planner_next_action(user_input: str, observation: dict[str, Any], step: int) -> dict[str, Any]:
    available_columns = ", ".join(_get_products_columns())
    planner_messages = [
        {
            "role": "system",
            "content": (
                f"{PLANNER_SYSTEM_PROMPT}\n"
                f"Available products columns: {available_columns}\n"
                "Prefer safe SQL with LIMIT."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "step": step,
                    "user_input": user_input,
                    "observation": observation,
                },
                default=str,
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=planner_messages,
            temperature=0,
            top_p=1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _extract_first_json_object(raw) or {}
    except Exception:
        logger.exception("Planner step failed; fallback to route planner.")
        route = _route_user_query(user_input)
        return {
            "action": "sql" if route["route"] == "sql" else ("semantic_search" if route["route"] == "semantic_search" else "respond"),
            "reason": route.get("reason", "fallback_route"),
            "semantic_query": route.get("semantic_query", user_input),
            "sql_query": route.get("sql_query", ""),
            "response_hint": "",
        }

    action = str(parsed.get("action", "")).strip().lower()
    if action not in {"semantic_search", "sql", "respond"}:
        action = "respond" if step > 1 else "semantic_search"
    return {
        "action": action,
        "reason": str(parsed.get("reason", "")).strip(),
        "semantic_query": str(parsed.get("semantic_query", user_input)).strip() or user_input,
        "sql_query": str(parsed.get("sql_query", "")).strip(),
        "response_hint": str(parsed.get("response_hint", "")).strip(),
    }


def _run_agentic_loop(user_input: str, trace: Any = None, max_steps: int = 3) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    observation: dict[str, Any] = {"status": "start"}
    for step in range(1, max_steps + 1):
        plan = _planner_next_action(user_input=user_input, observation=observation, step=step)
        action = plan["action"]
        step_payload: dict[str, Any] = {"step": step, "plan": plan}

        if action == "respond":
            step_payload["observation"] = {"status": "enough_context"}
            steps.append(step_payload)
            monitor.event(trace, name="agent_step", metadata={"step": step, "action": action}, output=step_payload)
            break

        if action == "semantic_search":
            tool_result = _execute_tool("semantic_search_products", {"query": plan["semantic_query"], "limit": 5})
            observation = {"status": "semantic_done", "result_count": len(tool_result.get("result", [])) if isinstance(tool_result.get("result"), list) else 0}
            step_payload["tool"] = {"function": "semantic_search_products", "result": tool_result}
            steps.append(step_payload)
            monitor.event(trace, name="agent_step", metadata={"step": step, "action": action}, output=observation)
            continue

        sql_query = plan["sql_query"] or _fallback_sql_for_user_query(user_input)
        policy = _guard_sql_policy(sql_query)
        if not policy["allowed"]:
            sql_query = _fallback_sql_for_user_query(user_input)
        sql_run = _run_sql_with_self_correction(
            user_input=user_input,
            initial_sql=sql_query,
            max_attempts=3,
            trace=trace,
        )
        observation = {
            "status": "sql_done",
            "self_corrected": sql_run.get("self_corrected"),
            "risk": policy["risk"],
        }
        step_payload["tool"] = {
            "function": "get_data_from_db",
            "policy": policy,
            "result": sql_run["tool_result"],
            "attempts": sql_run["attempts"],
            "final_sql": sql_run["final_sql"],
        }
        steps.append(step_payload)
        monitor.event(trace, name="agent_step", metadata={"step": step, "action": action}, output=observation)
    return steps


def _critic_revise_answer(user_input: str, draft_reply: str, execution_steps: list[dict[str, Any]]) -> str:
    critic_messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_input": user_input,
                    "draft_reply": draft_reply,
                    "execution_steps": execution_steps,
                },
                default=str,
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=critic_messages,
            temperature=0,
            top_p=1,
        )
        parsed = _extract_first_json_object((resp.choices[0].message.content or "").strip()) or {}
    except Exception:
        logger.exception("Critic pass failed; using draft reply.")
        return draft_reply

    needs_revision = bool(parsed.get("needs_revision", False))
    improved = str(parsed.get("improved_answer", "")).strip()
    if needs_revision and improved:
        return improved
    return draft_reply


def get_response(session_id: str, user_input: str) -> str:
    """Main flow: user message -> router -> optional tool -> final reply."""
    if not HF_TOKEN:
        logger.error("HF_TOKEN is missing; cannot call Hugging Face Router.")
        return "HF_TOKEN is missing. Add it in .env and restart the server."

    trace = monitor.start_trace(
        name="chat_request",
        session_id=session_id,
        input_payload={"user_input": user_input},
    )
    messages = _get_or_create_session(session_id)
    messages.append({"role": "user", "content": user_input})
    model_messages = _inject_memory_context(
        session_id=session_id, user_input=user_input, messages=messages
    )
    logger.info("User message appended. session_id=%s total_messages=%s", session_id, len(messages))

    execution_steps: list[dict[str, Any]] = []
    if ENABLE_TOOLS:
        execution_steps = _run_agentic_loop(user_input=user_input, trace=trace, max_steps=3)
        messages.append(
            {
                "role": "assistant",
                "content": f"Execution context: {json.dumps({'steps': execution_steps}, default=str)}",
            }
        )
        model_messages = _inject_memory_context(session_id=session_id, user_input=user_input, messages=messages)

    try:
        logger.info("Calling model=%s tools_enabled=%s", MODEL_NAME, ENABLE_TOOLS)
        response = _completion(model_messages, use_tools=ENABLE_TOOLS)
    except BadRequestError as exc:
        # Some routed models/providers reject tools support.
        if "does not support tools" in str(exc).lower():
            logger.warning("Model/provider does not support tools. Retrying without tools.")
            response = _completion(model_messages, use_tools=False)
        else:
            logger.exception("LLM completion failed for session_id=%s", session_id)
            monitor.event(trace, name="llm_error", output={"error": str(exc)})
            monitor.end_trace(trace, output={"error": str(exc)})
            raise
    except Exception:
        logger.exception("LLM completion failed for session_id=%s", session_id)
        monitor.event(trace, name="llm_error", output={"error": "unexpected_exception"})
        monitor.end_trace(trace, output={"error": "unexpected_exception"})
        raise

    msg = response.choices[0].message

    if ENABLE_TOOLS and getattr(msg, "tool_calls", None):
        # Keep this simple: handle only the first tool call.
        tool_call = msg.tool_calls[0]
        function_name = tool_call.function.name
        args = _parse_tool_args(tool_call.function.arguments or "{}")
        logger.info("Tool call received: %s args=%s", function_name, args)

        # Important: add assistant tool-call message before tool response.
        # Some providers validate that each tool message references a prior tool_call id.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call.function.arguments or "{}",
                        },
                    }
                ],
            }
        )

        tool_result = _execute_tool(function_name, args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, default=str),
            }
        )
        logger.info("Tool result appended. Requesting follow-up assistant response.")
        followup = _completion(messages, use_tools=False)
        msg = followup.choices[0].message

    reply = (msg.content or "").strip()

    # Fallback: if provider returned tool-call JSON as plain text,
    # run it and ask model for a normal user-facing answer.
    text_tool_result = _try_execute_text_tool_call(reply)
    if ENABLE_TOOLS and text_tool_result is not None:
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "assistant", "content": f"Executed text tool call result: {json.dumps(text_tool_result, default=str)}"})
        followup = _completion(messages, use_tools=False)
        msg = followup.choices[0].message
        reply = (msg.content or "").strip()

    reply = _critic_revise_answer(user_input=user_input, draft_reply=reply, execution_steps=execution_steps)
    messages.append({"role": "assistant", "content": reply})
    episodic_memory.append_turn(
        session_id=session_id, user_input=user_input, assistant_reply=reply
    )
    chat_sessions[session_id] = _trim_history(messages)
    monitor.end_trace(
        trace,
        output={"reply": reply},
        metadata={"reply_chars": len(reply), "tools_enabled": ENABLE_TOOLS},
    )
    logger.info("Response generated for session_id=%s reply_chars=%s", session_id, len(reply))
    return reply
