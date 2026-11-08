from __future__ import annotations

from fitfuel.context import profile_ctx, trace_ctx, user_ctx
from fitfuel.graph import GRAPH
from fitfuel.llm import llm_configured, load_env
from fitfuel import store


def _recent_chat(user_id: str, message: str) -> str:
    lines = []
    for row in store.list_messages(user_id):
        role = row.get("role")
        if role not in {"user", "assistant"}:
            continue
        lines.append(f"{role}: {row.get('content', '')}")
    current = f"user: {message.strip()}"
    if lines and lines[-1] == current:
        lines.pop()
    kept = lines[-6:]
    return "\n".join(kept) if kept else "None"


def run_turn(
    user_id: str,
    message: str,
    *,
    force_intent: str | None = None,
    use_llm: bool | None = None,
    allow_web: bool = False,
) -> dict:
    load_env()
    store.init_db()
    profile = store.get_profile(user_id)
    current = store.get_current_plan(user_id)
    if use_llm is None:
        use_llm = llm_configured()
    store.clear_pending(user_id)
    trace: list[dict] = []
    trace_token = trace_ctx.set(trace)
    user_token = user_ctx.set(user_id)
    profile_token = profile_ctx.set(profile)
    initial = {
        "user_id": user_id,
        "message": message.strip(),
        "has_plan": current is not None,
        "force_intent": force_intent,
        "allow_web": allow_web,
        "use_llm": use_llm,
        "profile": profile,
        "workouts": (current or {}).get("workouts") or [],
        "meals": (current or {}).get("meals") or [],
        "macros": (current or {}).get("macros") or {},
        "meal_totals": (current or {}).get("meal_totals") or {},
        "change_notes": [],
        "lookups": (current or {}).get("lookups") or {},
        "repaired": False,
        "history": _recent_chat(user_id, message),
    }
    try:
        final = GRAPH.invoke(initial, config={"recursion_limit": 25})
    except Exception as exc:
        trace.append({"agent": "Coach Agent", "tool": "error", "detail": exc.__class__.__name__})
        final = None
        error = exc
    else:
        error = None
    finally:
        trace_ctx.reset(trace_token)
        user_ctx.reset(user_token)
        profile_ctx.reset(profile_token)
    if error is not None:
        return {
            "answer": "The coach stopped before changing your plan. Please try that message again.",
            "plan": current,
            "trace": trace,
            "intent": "error",
        }
    intent = final.get("intent")
    plan = current if intent in {"medical", "question", "error"} else store.get_current_plan(user_id)
    return {
        "answer": final.get("answer") or "Tell me what you want to change in the plan.",
        "plan": plan,
        "trace": trace,
        "intent": intent,
    }
