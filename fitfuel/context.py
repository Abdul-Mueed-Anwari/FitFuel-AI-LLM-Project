from __future__ import annotations

import contextvars

trace_ctx: contextvars.ContextVar[list | None] = contextvars.ContextVar("ff_trace", default=None)
agent_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("ff_agent", default="Coach Agent")
profile_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("ff_profile", default=None)
user_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("ff_user", default=None)


def log_tool(name: str, detail: str) -> None:
    bucket = trace_ctx.get()
    if bucket is None:
        return
    text = " ".join(str(detail).split())
    if len(text) > 180:
        text = text[:177] + "..."
    bucket.append({"agent": agent_ctx.get(), "tool": name, "detail": text})
