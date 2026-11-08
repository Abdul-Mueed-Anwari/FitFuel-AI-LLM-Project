from __future__ import annotations

import json
import math
import re
import threading

from langchain_core.tools import tool

from fitfuel.catalog import exercise_by_id, get_bm25, meal_by_id, search_knowledge
from fitfuel.catalog_data import NOTES
from fitfuel.context import log_tool, profile_ctx, user_ctx
from fitfuel.rules import validate
from fitfuel import store

_retriever_tool = None
_wiki_tool = None
_ddg_tool = None

WIKI_AGENT = "FitFuelCoach/1.0 (Abdul Mueed Anwari; educational fitness project) Python-requests"


def _run_timed(fn, seconds: float) -> str:
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        return "unavailable: timeout"
    if "error" in box:
        return f"unavailable: {box['error'].__class__.__name__}: {box['error']}"
    value = box.get("value")
    return str(value).strip() if value else "unavailable: empty"


def _hit(item_id: str) -> dict | None:
    exercise = exercise_by_id(item_id)
    if exercise:
        return {
            "id": exercise["id"],
            "kind": "exercise",
            "name": exercise["name"],
            "text": f"{exercise['name']}. {exercise['cue']}",
        }
    meal = meal_by_id(item_id)
    if meal:
        return {
            "id": meal["id"],
            "kind": "meal",
            "name": meal["name"],
            "text": f"{meal['name']}. Ingredients: {', '.join(meal['ingredients'])}.",
        }
    note = next((item for item in NOTES if item["id"] == item_id), None)
    if note:
        return {"id": note["id"], "kind": "note", "name": note["title"], "text": note["text"]}
    return None


def _official_retriever():
    global _retriever_tool
    if _retriever_tool is None:
        from langchain_core.tools.retriever import create_retriever_tool

        retriever = get_bm25()
        if retriever is None:
            return None
        _retriever_tool = create_retriever_tool(
            retriever,
            "knowledge_retriever",
            "Find exercises, meals, and nutrition notes.",
        )
    return _retriever_tool


@tool
def calculator(expression: str) -> str:
    """Calculate a calorie expression."""
    import numexpr

    value = numexpr.evaluate(
        expression.strip(),
        global_dict={},
        local_dict={"pi": math.pi, "e": math.e},
    )
    shown = round(float(value), 2)
    log_tool("calculator", f"{expression} = {shown}")
    return str(shown)


@tool
def knowledge_retriever(query: str, kind: str = "any") -> str:
    """Find exercises, meals, or notes."""
    hits: list[dict] = []
    official = _official_retriever()
    if official is not None:
        content = str(official.invoke(query))
        for item_id in re.findall(r"id=([A-Za-z0-9_]+)", content):
            hit = _hit(item_id)
            if hit and (kind == "any" or hit["kind"] == kind) and hit["id"] not in {item["id"] for item in hits}:
                hits.append(hit)
    if len(hits) < 4:
        for hit in search_knowledge(query, kind=kind, k=6):
            if hit["id"] not in {item["id"] for item in hits}:
                hits.append(hit)
    names = ", ".join(hit["name"] for hit in hits[:3]) or "nothing"
    log_tool("knowledge_retriever", f"{kind} · {query} · {names}")
    return json.dumps(hits[:6])


@tool
def get_user_profile(user_id: str = "") -> str:
    """Load the saved profile."""
    profile = profile_ctx.get()
    if profile is None:
        profile = store.get_profile(user_id or user_ctx.get() or "")
    log_tool(
        "get_user_profile",
        f"{profile.get('name', 'user')} · {profile.get('goal')} · {profile.get('days_per_week')} days",
    )
    return json.dumps(profile)


@tool
def validate_plan(plan_json: str) -> str:
    """Check the plan."""
    profile = profile_ctx.get() or {}
    result = validate(profile, json.loads(plan_json))
    if result["ok"]:
        log_tool("validate_plan", "passed")
    else:
        log_tool("validate_plan", "; ".join(result["issues"]))
    return json.dumps(result)


@tool
def save_plan(plan_json: str) -> str:
    """Save the plan."""
    user_id = user_ctx.get() or ""
    plan = json.loads(plan_json)
    plan_id = store.save_plan(user_id, plan)
    store.update_profile(user_id, profile_ctx.get() or store.get_profile(user_id))
    log_tool("save_plan", plan.get("summary", plan_id))
    return plan_id


@tool
def update_plan(plan_json: str) -> str:
    """Update the plan."""
    user_id = user_ctx.get() or ""
    plan = json.loads(plan_json)
    plan_id = store.update_plan(user_id, plan)
    store.update_profile(user_id, profile_ctx.get() or store.get_profile(user_id))
    log_tool("update_plan", plan.get("summary", plan_id))
    return plan_id


def _wiki():
    global _wiki_tool
    if _wiki_tool is None:
        import wikipedia
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        wikipedia.set_user_agent(WIKI_AGENT)
        _wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(wiki_client=wikipedia, top_k_results=1, doc_content_chars_max=500)
        )
    return _wiki_tool


def _ddg():
    global _ddg_tool
    if _ddg_tool is None:
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

        _ddg_tool = DuckDuckGoSearchRun(
            api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3, time=None, backend="html")
        )
    return _ddg_tool


def wikipedia_lookup(query: str) -> str:
    text = _run_timed(lambda: str(_wiki().invoke(query)), 15)
    log_tool("wikipedia", text)
    return text


def duckduckgo_search(query: str) -> str:
    def fetch() -> str:
        try:
            return str(_ddg().invoke(query))
        except Exception:
            return str(_ddg().invoke("beginner high protein meals"))

    text = _run_timed(fetch, 15)
    log_tool("duckduckgo_search", text)
    return text
