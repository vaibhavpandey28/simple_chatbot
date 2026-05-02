from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


@dataclass
class Episode:
    timestamp_iso: str
    user_input: str
    assistant_reply: str
    summary: str
    tags: list[str]


class EpisodicMemoryManager:
    """Session memory with LangGraph checkpointer + episodic retrieval layer."""

    def __init__(self, max_episodes_per_session: int = 200) -> None:
        self.max_episodes_per_session = max_episodes_per_session
        self._episodes: dict[str, list[Episode]] = {}

        builder = StateGraph(MessagesState)
        builder.add_node("append_messages", self._append_messages)
        builder.add_edge(START, "append_messages")
        builder.add_edge("append_messages", END)
        self._graph = builder.compile(checkpointer=MemorySaver())

    @staticmethod
    def _append_messages(state: MessagesState) -> dict[str, Any]:
        return {"messages": state["messages"]}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

    def _episode_summary(self, user_input: str, assistant_reply: str) -> str:
        # Keep this deterministic and cheap; this is the raw episodic record.
        combined = f"User asked: {user_input.strip()} | Assistant answered: {assistant_reply.strip()}"
        return combined[:400]

    def append_turn(self, session_id: str, user_input: str, assistant_reply: str) -> None:
        config = {"configurable": {"thread_id": session_id}}
        self._graph.invoke(
            {"messages": [HumanMessage(content=user_input), AIMessage(content=assistant_reply)]},
            config=config,
        )

        summary = self._episode_summary(user_input, assistant_reply)
        tags = sorted(self._tokenize(user_input))[:12]
        episode = Episode(
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            user_input=user_input,
            assistant_reply=assistant_reply,
            summary=summary,
            tags=tags,
        )
        session_episodes = self._episodes.setdefault(session_id, [])
        session_episodes.append(episode)
        if len(session_episodes) > self.max_episodes_per_session:
            del session_episodes[0 : len(session_episodes) - self.max_episodes_per_session]

    def get_recent_messages(self, session_id: str, limit: int = 6) -> list[BaseMessage]:
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self._graph.get_state(config)
        values = snapshot.values or {}
        messages = values.get("messages", [])
        if not isinstance(messages, list):
            return []
        return messages[-max(1, limit) :]

    def get_relevant_episodes(self, session_id: str, query: str, top_k: int = 3) -> list[Episode]:
        episodes = self._episodes.get(session_id, [])
        if not episodes:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return episodes[-top_k:]

        scored: list[tuple[float, Episode]] = []
        for ep in episodes:
            ep_tokens = set(ep.tags)
            overlap = len(q_tokens & ep_tokens)
            recency_bonus = 0.05
            score = overlap + recency_bonus
            scored.append((score, ep))

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
