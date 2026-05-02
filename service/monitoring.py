import os
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


class LangfuseMonitor:
    """Best-effort Langfuse monitor. No-op when not configured."""

    def __init__(self) -> None:
        self.enabled = os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._client = None

        if not self.enabled:
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if not public_key or not secret_key:
            logger.warning("Langfuse enabled but keys are missing; disabling Langfuse")
            self.enabled = False
            return

        try:
            from langfuse import Langfuse  # type: ignore

            self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            logger.info("Langfuse monitoring enabled")
        except Exception:
            logger.exception("Failed to initialize Langfuse; disabling")
            self.enabled = False

    def start_trace(self, *, name: str, session_id: str, input_payload: Any):
        if not self.enabled or self._client is None:
            return None
        try:
            # Langfuse SDK v2 style.
            if hasattr(self._client, "trace"):
                return self._client.trace(name=name, session_id=session_id, input=input_payload)

            # Langfuse SDK v3 style: create a trace id and emit events with trace_context.
            if hasattr(self._client, "create_trace_id") and hasattr(self._client, "create_event"):
                trace_id = self._client.create_trace_id(seed=f"{session_id}:{name}")
                trace_handle = {"trace_id": trace_id, "session_id": session_id, "name": name}
                self._client.create_event(
                    trace_context={"trace_id": trace_id, "session_id": session_id},
                    name=f"{name}.start",
                    input=input_payload,
                    metadata={"session_id": session_id},
                )
                return trace_handle

            return None
        except Exception:
            logger.exception("Failed creating Langfuse trace")
            return None

    def event(self, trace: Any, *, name: str, metadata: dict[str, Any] | None = None, output: Any = None) -> None:
        if not trace:
            return
        try:
            # Preferred path: event if available.
            if hasattr(trace, "event"):
                trace.event(name=name, metadata=metadata or {}, output=output)
                return
            # SDK v3 style handle.
            if isinstance(trace, dict) and self._client and hasattr(self._client, "create_event"):
                self._client.create_event(
                    trace_context={
                        "trace_id": trace.get("trace_id"),
                        "session_id": trace.get("session_id"),
                    },
                    name=name,
                    output=output,
                    metadata=metadata or {},
                )
                return
            # Fallback: span to annotate step.
            if hasattr(trace, "span"):
                span = trace.span(name=name, metadata=metadata or {})
                if hasattr(span, "end"):
                    span.end(output=output)
        except Exception:
            logger.exception("Failed recording Langfuse event")

    def end_trace(self, trace: Any, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        if not trace:
            return
        try:
            if isinstance(trace, dict) and self._client and hasattr(self._client, "create_event"):
                self._client.create_event(
                    trace_context={
                        "trace_id": trace.get("trace_id"),
                        "session_id": trace.get("session_id"),
                    },
                    name=f"{trace.get('name', 'chat_request')}.end",
                    output=output,
                    metadata=metadata or {},
                )

            if hasattr(trace, "update"):
                trace.update(output=output, metadata=metadata or {})
            if self._client and hasattr(self._client, "flush"):
                self._client.flush()
        except Exception:
            logger.exception("Failed finalizing Langfuse trace")
