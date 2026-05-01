import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from core.logger import get_logger
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


# API client (Hugging Face Router speaks OpenAI-compatible format).
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN or None,
)

# Stores chat history in memory: {session_id: [messages...]}
chat_sessions: dict[str, list[dict[str, str]]] = {}


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


def get_response(session_id: str, user_input: str) -> str:
    """Main flow: user message -> model -> optional tool -> final reply."""
    if not HF_TOKEN:
        logger.error("HF_TOKEN is missing; cannot call Hugging Face Router.")
        return "HF_TOKEN is missing. Add it in .env and restart the server."

    messages = _get_or_create_session(session_id)
    messages.append({"role": "user", "content": user_input})
    logger.info("User message appended. session_id=%s total_messages=%s", session_id, len(messages))

    try:
        logger.info("Calling model=%s tools_enabled=%s", MODEL_NAME, ENABLE_TOOLS)
        response = _completion(messages, use_tools=ENABLE_TOOLS)
    except BadRequestError as exc:
        # Some routed models/providers reject tools support.
        if "does not support tools" in str(exc).lower():
            logger.warning("Model/provider does not support tools. Retrying without tools.")
            response = _completion(messages, use_tools=False)
        else:
            logger.exception("LLM completion failed for session_id=%s", session_id)
            raise
    except Exception:
        logger.exception("LLM completion failed for session_id=%s", session_id)
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
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(text_tool_result, default=str),
            }
        )
        followup = _completion(messages, use_tools=False)
        msg = followup.choices[0].message
        reply = (msg.content or "").strip()

    messages.append({"role": "assistant", "content": reply})
    chat_sessions[session_id] = _trim_history(messages)
    logger.info("Response generated for session_id=%s reply_chars=%s", session_id, len(reply))
    return reply
