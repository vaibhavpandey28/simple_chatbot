import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from core.logger import get_logger
from service.tools import get_data_from_db

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
"""


# OpenAI-compatible client for Hugging Face Router.
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN or None,
)

# In-memory per-session history.
chat_sessions: dict[str, list[dict[str, str]]] = {}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_data_from_db",
            "description": "Execute a SELECT SQL query on the PostgreSQL ecommerce database.",
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
    """Keep system prompt + latest conversation turns."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    logger.debug(
        "Trimming history from %s to %s messages",
        len(messages),
        MAX_HISTORY_MESSAGES,
    )
    return [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1) :]


def _completion(messages: list[dict[str, str]], use_tools: bool):
    """Call LLM once. Optionally attach tool definitions."""
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
    """Initialize session with system prompt if missing."""
    if session_id not in chat_sessions:
        logger.info("Creating new session: %s", session_id)
        chat_sessions[session_id] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    return chat_sessions[session_id]


def _parse_tool_args(raw_args: str) -> dict[str, Any]:
    """Parse tool-call JSON args safely."""
    try:
        return json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        logger.warning("Invalid tool arguments JSON: %s", raw_args)
        return {}


def _execute_tool(function_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute supported tool and return serializable response."""
    if function_name != "get_data_from_db":
        logger.warning("Unsupported tool requested: %s", function_name)
        return {"error": f"Unsupported tool: {function_name}"}

    query = args.get("query")
    if not query:
        return {"error": "Missing required argument: query"}

    logger.info("Executing tool get_data_from_db")
    return {"result": get_data_from_db(query=query)}


def get_response(session_id: str, user_input: str) -> str:
    """Main chat flow: user message -> LLM -> optional tool -> final assistant reply."""
    if not HF_TOKEN:
        logger.error("HF_TOKEN is missing; cannot call Hugging Face Router.")
        return "HF_TOKEN is missing. Add it in .env and restart the server."

    messages = _get_or_create_session(session_id)
    messages.append({"role": "user", "content": user_input})
    logger.debug("User message appended. session_id=%s total_messages=%s", session_id, len(messages))

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
        # Handle only first tool-call for simplicity.
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
    messages.append({"role": "assistant", "content": reply})
    chat_sessions[session_id] = _trim_history(messages)
    logger.info("Response generated for session_id=%s reply_chars=%s", session_id, len(reply))
    return reply
