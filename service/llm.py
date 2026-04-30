from openai import OpenAI
from core.logger import get_logger
import os
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

chat_sessions ={}
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3:mini")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "16"))

# - Prefer code examples when relevant
# - Assume user is a developer learning LLMs

DEFAULT_SYSTEM_PROMPT = """
You are Vaibhav's personal AI assistant.
- Be concise and practical
- Avoid unnecessary explanations.
- Stay on-topic and answer only what the user asked.
- Do not invent names, stories, or context.
- If unsure, say so briefly and ask for clarification.
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
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
    except Exception:
        logger.exception("LLM completion failed for session_id=%s", session_id)
        raise

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    # Keep memory bounded to reduce drift from old context.
    if len(messages) > MAX_HISTORY_MESSAGES:
        logger.debug("Trimming chat history for session_id=%s", session_id)
        chat_sessions[session_id] = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

    logger.info("Generated response for session_id=%s", session_id)
    return reply
