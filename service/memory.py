from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.logger import get_logger
from service.db import get_connection

logger = get_logger(__name__)


@dataclass
class Episode:
    timestamp_iso: str
    user_input: str
    assistant_reply: str
    summary: str
    tags: list[str]


class EpisodicMemoryManager:
    """Persistent episodic memory backed by PostgreSQL."""

    def __init__(self, max_episodes_per_session: int = 200) -> None:
        self.max_episodes_per_session = max_episodes_per_session
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_input TEXT NOT NULL,
                    assistant_reply TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags TEXT[] NOT NULL DEFAULT '{}'
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages_memory (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_session_created
                ON episodic_memory (session_id, created_at DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_memory_session_created
                ON chat_messages_memory (session_id, created_at DESC)
                """
            )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Failed creating memory schema")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

    def _episode_summary(self, user_input: str, assistant_reply: str) -> str:
        combined = f"User asked: {user_input.strip()} | Assistant answered: {assistant_reply.strip()}"
        return combined[:400]

    def append_turn(self, session_id: str, user_input: str, assistant_reply: str) -> None:
        summary = self._episode_summary(user_input, assistant_reply)
        tags = sorted(self._tokenize(user_input))[:12]

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO chat_messages_memory (session_id, role, content)
                VALUES (%s, 'user', %s), (%s, 'assistant', %s)
                """,
                (session_id, user_input, session_id, assistant_reply),
            )
            cur.execute(
                """
                INSERT INTO episodic_memory (session_id, user_input, assistant_reply, summary, tags)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, user_input, assistant_reply, summary, tags),
            )

            cur.execute(
                """
                DELETE FROM episodic_memory
                WHERE session_id = %s
                  AND id NOT IN (
                    SELECT id FROM episodic_memory
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                  )
                """,
                (session_id, session_id, self.max_episodes_per_session),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Failed appending episodic memory")
        finally:
            cur.close()
            conn.close()

    def get_recent_messages(self, session_id: str, limit: int = 6) -> list[BaseMessage]:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages_memory
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, max(1, limit)),
            )
            rows = cur.fetchall()
        except Exception:
            logger.exception("Failed reading recent memory messages")
            return []
        finally:
            cur.close()
            conn.close()

        rows = list(reversed(rows))
        out: list[BaseMessage] = []
        for role, content in rows:
            if role == "user":
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
        return out

    def get_relevant_episodes(self, session_id: str, query: str, top_k: int = 3) -> list[Episode]:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT created_at, user_input, assistant_reply, summary, tags
                FROM episodic_memory
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, max(20, self.max_episodes_per_session)),
            )
            rows = cur.fetchall()
        except Exception:
            logger.exception("Failed reading episodic memory")
            return []
        finally:
            cur.close()
            conn.close()

        if not rows:
            return []

        q_tokens = self._tokenize(query)
        episodes: list[Episode] = []
        for created_at, user_input, assistant_reply, summary, tags in rows:
            ts = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)
            episodes.append(
                Episode(
                    timestamp_iso=ts,
                    user_input=user_input,
                    assistant_reply=assistant_reply,
                    summary=summary,
                    tags=list(tags or []),
                )
            )

        if not q_tokens:
            return episodes[:top_k]

        scored: list[tuple[float, Episode]] = []
        for idx, ep in enumerate(episodes):
            overlap = len(q_tokens & set(ep.tags))
            recency_bonus = max(0.0, 1.0 - (idx * 0.02))
            scored.append((overlap + recency_bonus, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def build_memory_context(self, session_id: str, query: str, top_k: int = 3) -> str:
        recent = self.get_recent_messages(session_id=session_id, limit=4)
        episodes = self.get_relevant_episodes(session_id=session_id, query=query, top_k=top_k)

        lines: list[str] = []
        if recent:
            lines.append("Recent conversation memory:")
            for msg in recent:
                role = "user" if msg.type == "human" else "assistant"
                lines.append(f"- {role}: {str(msg.content)[:200]}")

        if episodes:
            lines.append("Relevant episodic memory:")
            for ep in episodes:
                lines.append(f"- [{ep.timestamp_iso}] {ep.summary}")

        return "\n".join(lines).strip()
