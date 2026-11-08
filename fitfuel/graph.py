from __future__ import annotations

import json
import re

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from fitfuel.context import agent_ctx, log_tool, profile_ctx, user_ctx
from fitfuel.intent import _concrete, apply_updates, classify
from fitfuel.planning import make_meals, make_workout
from fitfuel.profiles import EQUIPMENT_LABELS, GOAL_LABELS
from fitfuel.prompts import CHAT_PROMPT, CONVERSATION_PROMPT, RESULT_PROMPT, SAFETY_PROMPT
from fitfuel.tools import (
    duckduckgo_search,
    get_user_profile,
    knowledge_retriever,
    save_plan,
    update_plan,
    validate_plan,
    wikipedia_lookup,
)

GENERIC_NOTES = {
    "Built the weekly workout from the exercise list.",
    "Chose meals from the approved list.",
}


class CoachState(TypedDict, total=False):
    user_id: str
    message: str
    has_plan: bool
    force_intent: str | None
    allow_web: bool
    use_llm: bool
    profile: dict
    updates: dict
    intent: str
    workouts: list
    meals: list
    macros: dict
    meal_totals: dict
    validation: dict
    repaired: bool
    change_notes: list
    lookups: dict
    answer: str
    history: str
    model_say: str
    prior_workouts: list
    prior_meals: list
    applied: bool
    custom_id: str | None


def _bind(state: CoachState, agent: str) -> None:
    agent_ctx.set(agent)
    user_ctx.set(state.get("user_id"))
    profile_ctx.set(state.get("profile"))


def _json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _profile_text(profile: dict) -> str:
    avoid = ", ".join(profile.get("avoid") or []) or "none"
    allergies = ", ".join(profile.get("allergies") or []) or "none"
    notes = (profile.get("notes") or "").strip() or "none"
    return (
        f"{profile.get('name', 'User')}, goal {profile.get('goal')}, "
        f"{profile.get('days_per_week')} days, {profile.get('session_minutes')} minutes, "
        f"equipment {profile.get('equipment')}, diet {profile.get('diet')}, "
        f"cuisine {profile.get('cuisine')}, avoid {avoid}, allergies {allergies}, notes {notes}"
    )


def _week_text(workouts: list | None, meals: list | None) -> str:
    days = []
    for day in workouts or []:
        names = ", ".join(item.get("name", "") for item in day.get("exercises") or [])
        days.append(f"{day.get('title', 'Day')}: {names}")
    foods = [
        f"{meal.get('slot')}: {meal.get('name')} ({meal.get('calories')} kcal)"
        for meal in meals or []
    ]
    if not days and not foods:
        return "No week yet."
    return "Training: " + " | ".join(days) + ". Meals: " + "; ".join(foods) + "."


def _model_change(data: dict) -> bool:
    if data.get("diet") or data.get("cuisine") or data.get("equipment") or data.get("goal"):
        return True
    if data.get("session_minutes") in {30, 45, 60}:
        return True
    days = data.get("days_per_week")
    if isinstance(days, int) and 2 <= days <= 6:
        return True
    cap = data.get("dinner_calorie_cap")
    if isinstance(cap, int) and 250 <= cap <= 900:
        return True
    swap_from = data.get("swap_from")
    return isinstance(swap_from, str) and bool(swap_from.strip())


def coach_node(state: CoachState) -> dict:
    _bind(state, "Coach Agent")
    get_user_profile.invoke({"user_id": state.get("user_id", "")})
    parsed = classify(state.get("message", ""), bool(state.get("has_plan")), state.get("force_intent"))
    from fitfuel.dishes import dish_in_text, wants_without

    message = state.get("message", "")
    other_change = any(
        [
            parsed.get("diet"),
            parsed.get("cuisine"),
            parsed.get("dinner_calorie_cap"),
            parsed.get("session_minutes"),
            parsed.get("days_per_week"),
            parsed.get("equipment"),
            parsed.get("goal"),
            parsed.get("swap_requested"),
            parsed.get("prefer_easy"),
            parsed.get("regenerate"),
        ]
    )
    refers = bool(re.search(r"\b(it|that|this|them|those)\b", message, re.I))
    if wants_without(message) and parsed.get("intent") != "medical" and not other_change and (refers or dish_in_text(message)):
        parsed["add_avoid"] = []
        parsed["intent"] = "add"
        if not parsed.get("add_name"):
            parsed["add_from_history"] = True
    model_say = ""
    if state.get("use_llm") and not state.get("force_intent"):
        from fitfuel.llm import complete, last_error, provider_name

        raw = complete(
            CONVERSATION_PROMPT.format(
                profile=_profile_text(state.get("profile") or {}),
                week=_week_text(state.get("workouts") or [], state.get("meals") or []),
                history=state.get("history") or "None",
                message=state.get("message", ""),
            )
        )
        data = _json_object(raw or "")
        if data:
            model_say = str(data.get("say") or "").strip()
            _apply_model(parsed, data)
            log_tool("prompt", f"{provider_name()} read the message with the profile and the week")
        else:
            detail = f"{provider_name()} did not answer"
            if last_error and last_error != "no-model":
                detail += f" ({last_error})"
            log_tool("prompt", detail + ", used routing rules")
    if parsed["intent"] != "medical":
        profile = apply_updates(state["profile"], parsed)
    else:
        profile = state["profile"]
    if parsed["intent"] == "generate" and state.get("has_plan"):
        parsed["regenerate"] = True
        used = []
        for day in state.get("workouts") or []:
            for exercise in day.get("exercises") or []:
                if exercise.get("id"):
                    used.append(exercise["id"])
        for meal in state.get("meals") or []:
            if meal.get("id"):
                used.append(meal["id"])
        parsed["avoid_ids"] = used
    log_tool("route", parsed["intent"].replace("_", " "))
    return {
        "intent": parsed["intent"],
        "updates": parsed,
        "profile": profile,
        "change_notes": [],
        "model_say": model_say,
        "prior_workouts": list(state.get("workouts") or []),
        "prior_meals": list(state.get("meals") or []),
    }


def route_coach(state: CoachState) -> str:
    return state.get("intent") or "question"


def workout_node(state: CoachState) -> dict:
    _bind(state, "Workout Agent")
    days, notes = make_workout(state["profile"], state.get("updates") or {}, state.get("workouts") or [])
    return {"workouts": days, "change_notes": list(state.get("change_notes") or []) + notes}


def after_workout(state: CoachState) -> str:
    if state.get("intent") == "update_workout":
        return "tool_lookup"
    return "nutrition"


def nutrition_node(state: CoachState) -> dict:
    _bind(state, "Nutrition Agent")
    first = state.get("intent") == "generate"
    meals, totals, macros, notes = make_meals(state["profile"], state.get("updates") or {}, first)
    return {
        "meals": meals,
        "meal_totals": totals,
        "macros": macros,
        "change_notes": list(state.get("change_notes") or []) + notes,
    }


def _pack(state: CoachState) -> dict:
    profile = state["profile"]
    summary = f"{profile['days_per_week']}-day {GOAL_LABELS[profile['goal']]} plan"
    return {
        "summary": summary,
        "macros": state.get("macros") or {},
        "meal_totals": state.get("meal_totals") or {},
        "workouts": state.get("workouts") or [],
        "meals": state.get("meals") or [],
        "lookups": state.get("lookups") or {},
        "safety_notes": [
            "FitFuel Coach is not a doctor or dietitian. These are general beginner ideas, not medical advice."
        ],
        "validation": state.get("validation") or {"ok": False, "issues": ["The plan was not checked."]},
    }


def tool_lookup_node(state: CoachState) -> dict:
    if not state.get("allow_web"):
        return {}
    exercise = "squat exercise"
    days = state.get("workouts") or []
    if days and days[0].get("exercises"):
        exercise = f"{days[0]['exercises'][0]['name']} exercise"
    diet = (state.get("profile") or {}).get("diet") or "beginner"
    if diet == "none":
        diet = "balanced"
    meal_query = f"simple {diet} high protein meals for beginners"
    lookups = dict(state.get("lookups") or {})
    wiki_box: dict = {}
    search_box: dict = {}

    import contextvars
    import threading

    wiki_ctx = contextvars.copy_context()
    search_ctx = contextvars.copy_context()

    def run_wiki() -> None:
        def inner() -> None:
            _bind(state, "Workout Agent")
            wiki_box["text"] = wikipedia_lookup(exercise)

        wiki_ctx.run(inner)

    def run_search() -> None:
        def inner() -> None:
            _bind(state, "Nutrition Agent")
            search_box["text"] = duckduckgo_search(meal_query)

        search_ctx.run(inner)

    wiki_thread = threading.Thread(target=run_wiki, daemon=True)
    search_thread = threading.Thread(target=run_search, daemon=True)
    wiki_thread.start()
    search_thread.start()
    wiki_thread.join(16)
    search_thread.join(16)
    if wiki_box.get("text"):
        lookups["wikipedia"] = wiki_box["text"]
    if search_box.get("text"):
        lookups["duckduckgo"] = search_box["text"]
    return {"lookups": lookups}


def validator_node(state: CoachState) -> dict:
    _bind(state, "Validator Agent")
    raw = validate_plan.invoke({"plan_json": json.dumps(_pack(state))})
    return {"validation": json.loads(raw)}


def after_validate(state: CoachState) -> str:
    if state.get("validation", {}).get("ok"):
        return "save"
    if state.get("repaired") or state.get("intent") == "add":
        return "hold"
    return "repair"


def add_node(state: CoachState) -> dict:
    _bind(state, "Coach Agent")
    from fitfuel.suggest import place_request

    updates = state.get("updates") or {}
    current = {
        "workouts": state.get("workouts") or [],
        "meals": state.get("meals") or [],
        "meal_totals": state.get("meal_totals") or {},
        "macros": state.get("macros") or {},
    }
    placed = place_request(
        state.get("profile") or {},
        current,
        state.get("message", ""),
        bool(state.get("use_llm")),
        name=updates.get("add_name"),
        slot=updates.get("add_slot"),
        history=state.get("history") or "",
    )
    notes = list(state.get("change_notes") or []) + list(placed.get("notes") or [])
    log_tool("suggestion", notes[-1] if notes else "Checked the new item.")
    if not placed.get("applied"):
        reason = list(placed.get("notes") or ["It was not added."])
        return {
            "applied": False,
            "change_notes": notes,
            "validation": {"ok": False, "issues": reason},
            "custom_id": None,
        }
    return {
        "applied": True,
        "workouts": placed.get("workouts") or [],
        "meals": placed.get("meals") or [],
        "meal_totals": placed.get("meal_totals") or {},
        "change_notes": notes,
        "custom_id": placed.get("custom_id"),
    }


def route_after_add(state: CoachState) -> str:
    return "validator" if state.get("applied") else "explain"


def repair_node(state: CoachState) -> dict:
    _bind(state, "Validator Agent")
    issues = " ".join(state.get("validation", {}).get("issues") or []).lower()
    notes = list(state.get("change_notes") or [])
    workouts = state.get("workouts") or []
    meals = state.get("meals") or []
    totals = state.get("meal_totals") or {}
    macros = state.get("macros") or {}
    fresh = dict(state.get("updates") or {})
    fresh["regenerate"] = True
    if any(word in issues for word in ("equipment", "exercise", "training", "minute", "unknown exercise")):
        workouts, extra = make_workout(state["profile"], fresh, [])
        notes.extend(extra)
    if any(word in issues for word in ("diet", "food", "dinner", "meal", "calorie", "unknown meal", "missing")):
        meals, totals, macros, extra = make_meals(state["profile"], fresh, False)
        notes.extend(extra)
    notes.append("Validator Agent sent the plan back for a correction.")
    return {
        "workouts": workouts,
        "meals": meals,
        "meal_totals": totals,
        "macros": macros,
        "change_notes": notes,
        "repaired": True,
    }


def save_node(state: CoachState) -> dict:
    _bind(state, "Coach Agent")
    payload = json.dumps(_pack(state))
    if state.get("has_plan") and state.get("intent") != "generate":
        update_plan.invoke({"plan_json": payload})
    else:
        save_plan.invoke({"plan_json": payload})
    return {}


def explain_node(state: CoachState) -> dict:
    _bind(state, "Coach Agent")
    validation = state.get("validation") or {}
    ok = bool(validation.get("ok"))
    if not ok and state.get("custom_id"):
        from fitfuel import store

        store.delete_custom(state["custom_id"])
    answer = _template(state)
    if state.get("use_llm"):
        from fitfuel.llm import complete, last_error, provider_name

        issues = validation.get("issues") or []
        week = _week_text(
            state.get("workouts") if ok else state.get("prior_workouts"),
            state.get("meals") if ok else state.get("prior_meals"),
        )
        written = complete(
            RESULT_PROMPT.format(
                profile=_profile_text(state.get("profile") or {}),
                week=week,
                message=state.get("message", ""),
                notes="; ".join(_visible_notes(state)) or "None",
                ok="yes" if ok else "no",
                issues="; ".join(issues) or "none",
            )
        )
        if written:
            log_tool("prompt", f"{provider_name()} wrote the reply after the check")
            answer = written
            if not ok and not _says_refused(written):
                answer = written.strip() + " " + _refusal(state)
        else:
            detail = f"{provider_name()} did not answer"
            if last_error and last_error != "no-model":
                detail += f" ({last_error})"
            log_tool("prompt", detail + ", used the tool summary")
    return {"answer": answer}


def safety_node(state: CoachState) -> dict:
    _bind(state, "Coach Agent")
    kind = (state.get("updates") or {}).get("medical_kind")
    if kind == "extreme":
        answer = (
            "I will not build a very low-calorie diet. FitFuel Coach is not a doctor or dietitian, "
            "and extreme diets are outside this app. I can keep a moderate plan and change one meal, "
            "such as dinners under 500 calories."
        )
    else:
        answer = (
            "I can't help with injuries, diagnosis, or treatment. FitFuel Coach is not a doctor or dietitian. "
            "Please talk to a qualified professional about pain, injuries, pregnancy, or medical conditions. "
            "I can still build a general beginner plan when you want that."
        )
    if state.get("use_llm"):
        from fitfuel.llm import complete, provider_name

        written = complete(
            SAFETY_PROMPT.format(
                profile=_profile_text(state.get("profile") or {}),
                message=state.get("message", ""),
            )
        )
        if written and _says_refused(written):
            log_tool("prompt", f"{provider_name()} refused the medical request")
            answer = written
    return {"answer": answer}


def _fill_model(parsed: dict, data: dict) -> None:
    choices = {
        "diet": {"vegan", "vegetarian", "pescatarian"},
        "cuisine": {"pakistani", "chinese", "indian", "mediterranean", "mexican", "italian"},
        "equipment": {"dumbbells", "bands", "bodyweight", "home", "gym"},
        "goal": {"fat_loss", "muscle_gain", "maintain"},
    }
    for key, ok in choices.items():
        value = data.get(key)
        if value in ok and not parsed.get(key):
            parsed[key] = value
    minutes = data.get("session_minutes")
    if minutes in {30, 45, 60} and not parsed.get("session_minutes"):
        parsed["session_minutes"] = minutes
    days = data.get("days_per_week")
    if isinstance(days, int) and 2 <= days <= 6 and not parsed.get("days_per_week"):
        parsed["days_per_week"] = days
    cap = data.get("dinner_calorie_cap")
    if isinstance(cap, int) and 250 <= cap <= 900 and not parsed.get("dinner_calorie_cap"):
        parsed["dinner_calorie_cap"] = cap
    swap_from = data.get("swap_from")
    if isinstance(swap_from, str) and swap_from.strip() and not parsed.get("swap_from"):
        parsed["swap_requested"] = True
        parsed["swap_from"] = swap_from.strip().lower()
    add_name = data.get("add_name")
    if isinstance(add_name, str) and add_name.strip() and not parsed.get("add_name"):
        from fitfuel.dishes import is_reference

        if not is_reference(add_name):
            parsed["add_name"] = add_name.strip()
    slot = data.get("add_slot")
    if slot in {"breakfast", "lunch", "dinner", "snack"} and not parsed.get("add_slot"):
        parsed["add_slot"] = slot
    kind = data.get("add_kind")
    if kind in {"meal", "exercise"} and not parsed.get("add_kind"):
        parsed["add_kind"] = kind


def _apply_model(parsed: dict, data: dict) -> None:
    if data.get("intent") == "medical" or data.get("medical") is True:
        parsed["intent"] = "medical"
        parsed["medical"] = True
        parsed["medical_kind"] = parsed.get("medical_kind") or "injury"
        return
    intent = data.get("intent")
    allowed = {"generate", "update_workout", "update_meals", "update_both", "question", "add"}
    if intent not in allowed:
        intent = None
    if parsed.get("intent") == "add" or _concrete(parsed):
        _fill_model(parsed, data)
        return
    if intent == "add" and (data.get("add_name") or parsed.get("add_name")):
        parsed["intent"] = "add"
        _fill_model(parsed, data)
        return
    if intent in {"update_workout", "update_meals", "update_both", "generate"} and _model_change(data):
        parsed["intent"] = intent
        _fill_model(parsed, data)
        return
    if intent == "question":
        parsed["intent"] = "question"


def answer_node(state: CoachState) -> dict:
    _bind(state, "Coach Agent")
    message = state.get("message", "")
    from fitfuel.dishes import question_block

    blocked = question_block(message, state.get("history") or "", state.get("profile") or {})
    if blocked:
        log_tool("suggestion", "The dish includes something on the user's list.")
        return {"answer": blocked}
    knowledge_retriever.invoke({"query": message, "kind": "any"})
    notes = _web_notes(state, message) if state.get("allow_web") else {}
    brief = _search_brief(notes)
    if state.get("use_llm") and brief:
        from fitfuel.llm import complete, provider_name

        written = complete(
            CHAT_PROMPT.format(
                profile=_profile_text(state.get("profile") or {}),
                week=_week_text(state.get("workouts") or [], state.get("meals") or []),
                history=state.get("history") or "None",
                notes=brief,
                message=message,
            )
        )
        if written:
            log_tool("prompt", f"{provider_name()} wrote the reply from the search notes")
            return {"answer": written.strip()}
    written = str(state.get("model_say") or "").strip()
    if written:
        from fitfuel.llm import provider_name

        log_tool("prompt", f"{provider_name()} wrote the reply")
        return {"answer": written}
    if state.get("use_llm"):
        from fitfuel.llm import complete, last_error, provider_name

        written = complete(
            CHAT_PROMPT.format(
                profile=_profile_text(state.get("profile") or {}),
                week=_week_text(state.get("workouts") or [], state.get("meals") or []),
                history=state.get("history") or "None",
                notes=brief or "None",
                message=message,
            )
        )
        if written:
            log_tool("prompt", f"{provider_name()} wrote the reply")
            return {"answer": written}
        detail = f"{provider_name()} did not answer"
        if last_error and last_error != "no-model":
            detail += f" ({last_error})"
        log_tool("prompt", detail)
    return {
        "answer": "I can talk through this week with you. Ask me to swap a move, shorten a session, or change a meal."
    }


def _search_brief(notes: dict) -> str:
    parts = []
    for label, key in (("Wikipedia", "wikipedia"), ("DuckDuckGo", "duckduckgo")):
        text = " ".join(str(notes.get(key) or "").split())
        if not text or "unavailable" in text.lower():
            continue
        parts.append(f"{label}: {text[:500]}")
    return "\n".join(parts)


def _web_notes(state: CoachState, message: str) -> dict:
    import contextvars
    import threading

    wiki_box: dict = {}
    search_box: dict = {}
    wiki_ctx = contextvars.copy_context()
    search_ctx = contextvars.copy_context()

    def run(ctx, box: dict, fn) -> None:
        def inner() -> None:
            _bind(state, "Coach Agent")
            box["text"] = fn()

        ctx.run(inner)

    wiki = threading.Thread(
        target=run,
        args=(wiki_ctx, wiki_box, lambda: wikipedia_lookup(message)),
        daemon=True,
    )
    search = threading.Thread(
        target=run,
        args=(search_ctx, search_box, lambda: duckduckgo_search(message)),
        daemon=True,
    )
    wiki.start()
    search.start()
    wiki.join(8)
    search.join(8)
    notes = {}
    if wiki_box.get("text"):
        notes["wikipedia"] = wiki_box["text"]
    if search_box.get("text"):
        notes["duckduckgo"] = search_box["text"]
    return notes


def _says_refused(text: str) -> bool:
    lowered = text.lower()
    phrases = (
        "not add",
        "not added",
        "wasn't added",
        "was not added",
        "did not",
        "didn't",
        "do not",
        "won't",
        "will not",
        "could not",
        "cannot",
        "can't",
        "left the week",
        "left your week",
        "not change",
        "professional",
        "doctor",
    )
    return any(phrase in lowered for phrase in phrases)


def _refusal(state: CoachState) -> str:
    validation = state.get("validation") or {}
    reason = " ".join(_visible_notes(state)) or "; ".join(validation.get("issues") or [])
    if reason:
        return f"I checked it and did not change the week. {reason}"
    return "I checked it and did not change the week."


def _visible_notes(state: CoachState) -> list[str]:
    return [note for note in state.get("change_notes") or [] if note not in GENERIC_NOTES]


def _template(state: CoachState) -> str:
    profile = state["profile"]
    macros = state.get("macros") or {}
    calories = macros.get("calories")
    calorie_bit = f" Meals are aimed near {calories} kcal a day." if calories else ""
    diet = profile.get("diet") or "none"
    diet_bit = "" if diet == "none" else f" The meals are {diet}."
    avoid = profile.get("avoid") or []
    avoid_bit = f" They leave out {', '.join(avoid)}." if avoid else ""
    cap = profile.get("dinner_calorie_cap")
    cap_bit = f" Dinners stay under {cap} kcal." if cap else ""
    notes = " ".join(_visible_notes(state))
    validation = state.get("validation") or {}
    if not validation.get("ok"):
        return _refusal(state)
    lead = (
        f"Your {profile['days_per_week']}-day {GOAL_LABELS[profile['goal']]} plan uses "
        f"{EQUIPMENT_LABELS[profile['equipment']]} for about {profile['session_minutes']} minutes a session."
        f"{calorie_bit}{diet_bit}{avoid_bit}{cap_bit}"
    )
    if notes:
        lead += " " + notes
    lead += " The check passed, so this is the week I kept."
    return lead.strip()


def build_graph():
    graph = StateGraph(CoachState)
    graph.add_node("coach", coach_node)
    graph.add_node("workout", workout_node)
    graph.add_node("nutrition", nutrition_node)
    graph.add_node("tool_lookup", tool_lookup_node)
    graph.add_node("validator", validator_node)
    graph.add_node("repair", repair_node)
    graph.add_node("save", save_node)
    graph.add_node("explain", explain_node)
    graph.add_node("safety", safety_node)
    graph.add_node("answer", answer_node)
    graph.add_node("add_item", add_node)
    graph.add_edge(START, "coach")
    graph.add_conditional_edges(
        "coach",
        route_coach,
        {
            "medical": "safety",
            "question": "answer",
            "generate": "workout",
            "update_both": "workout",
            "update_workout": "workout",
            "update_meals": "nutrition",
            "add": "add_item",
        },
    )
    graph.add_conditional_edges(
        "add_item",
        route_after_add,
        {"validator": "validator", "explain": "explain"},
    )
    graph.add_conditional_edges(
        "workout",
        after_workout,
        {"tool_lookup": "tool_lookup", "nutrition": "nutrition"},
    )
    graph.add_edge("nutrition", "tool_lookup")
    graph.add_edge("tool_lookup", "validator")
    graph.add_conditional_edges(
        "validator",
        after_validate,
        {"repair": "repair", "save": "save", "hold": "explain"},
    )
    graph.add_edge("repair", "validator")
    graph.add_edge("save", "explain")
    graph.add_edge("explain", END)
    graph.add_edge("safety", END)
    graph.add_edge("answer", END)
    return graph.compile()


GRAPH = build_graph()
