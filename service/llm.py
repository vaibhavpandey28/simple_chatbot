from openai import OpenAI
from core.logger import get_logger
import os

logger = get_logger(__name__)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

chat_sessions ={}
MODEL_NAME = os.getenv("OLLAMA_MODEL", "tinyllama:latest")

DEFAULT_SYSTEM_PROMPT = """
You are Vaibhav's personal AI assistant.
- Be concise and practical
- Prefer code examples when relevant
- Assume user is a developer learning LLMs
- Avoid unnecessary explanations
"""

def get_response(session_id: str, user_input: str):
    if session_id not in chat_sessions:
        logger.info("Creating new session: %s", session_id)
        chat_sessions[session_id] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]

    messages = chat_sessions[session_id]
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
    except Exception:
        logger.exception("LLM completion failed for session_id=%s", session_id)
        raise

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    # optional: limit memory
    if len(messages) > 20:
        logger.debug("Trimming chat history for session_id=%s", session_id)
        chat_sessions[session_id] = [messages[0]] + messages[-18:]

    logger.info("Generated response for session_id=%s", session_id)
    return reply
