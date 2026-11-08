from __future__ import annotations

import json
import re
import uuid

from fitfuel.catalog import known_name, meal_calories
from fitfuel.dishes import (
    apply_omit,
    cleaned_name,
    conflicts,
    guess_slot,
    ingredient_words,
    refusal,
    resolve_dish,
)
from fitfuel.intent import extract, medical_reason
from fitfuel.profiles import EQUIPMENT_OK, MEAT_SOURCES
from fitfuel.rules import blocked_tokens, meal_allowed, validate
from fitfuel import store

YES_RE = re.compile(
    r"^(yes|yeah|yep|yup|sure|okay|ok|add it|please add(?: it)?|do it|go ahead)\b",
    re.I,
)
NO_RE = re.compile(r"^(no|nope|nah|don't|do not|not now|cancel|leave it)\b", re.I)
EXPLICIT_RE = re.compile(
    r"\b(suggest|not on the list|isn't on the list|isnt on the list|outside the list|"
    r"new exercise|new meal|new dinner|new breakfast|new lunch|new snack)\b",
    re.I,
)
REQUEST_RE = re.compile(r"\b(?:add|include|try)\s+(?:a |an |the )?(?:new )?(.+)$", re.I)
GENERIC = {
    "it",
    "this",
    "that",
    "something",
    "one",
    "exercise",
    "meal",
    "food",
    "workout",
    "dinner",
    "breakfast",
    "lunch",
    "snack",
    "new",
}
FOOD_RE = re.compile(
    r"\b(meal|dinner|breakfast|lunch|snack|food|eat|recipe|cuisine|chicken|beef|rice|fish|egg|paneer|dal|daal|biryani)\b",
    re.I,
)
TRAIN_RE = re.compile(
    r"\b(exercise|workout|lift|training|press|squat|lunge|row|plank|curl|swing|carry|raise|crunch|deadlift|push-up|pushup)\b",
    re.I,
)
MEAT_WORDS = ("chicken", "beef", "pork", "turkey", "mutton", "lamb", "meat")
STOPWORDS = {"a", "an", "the", "and", "with", "for", "new", "my", "please", "some"}


def handle_suggestion(user_id: str, message: str, profile: dict, plan: dict | None, use_llm: bool) -> dict | None:
    text = message.strip()
    pending = store.get_pending(user_id)
    if pending and YES_RE.match(text):
        return _accept(user_id, profile, plan, pending)
    if pending and NO_RE.match(text):
        store.clear_pending(user_id)
        return _reply(plan, "suggest_declined", "Okay. I left the week as it is.")
    if not _should_offer(text):
        return None
    if plan is None:
        return _reply(plan, "question", "Build a week first. Then I can ask before adding something new.")
    draft = _draft(text, profile, use_llm)
    if isinstance(draft, str):
        return _reply(plan, "suggest", draft)
    store.set_pending(user_id, draft)
    return _reply(plan, "suggest", _ask(draft), "Asked before adding a new item.")


def _should_offer(text: str) -> bool:
    if medical_reason(text):
        return False
    explicit = bool(EXPLICIT_RE.search(text))
    named = _named(text)
    if not explicit and not named:
        return False
    parsed = extract(text)
    structured = any(
        [
            parsed["diet"],
            parsed["cuisine"],
            parsed["add_avoid"],
            parsed["dinner_calorie_cap"],
            parsed["session_minutes"],
            parsed["days_per_week"],
            parsed["equipment"],
            parsed["goal"],
            parsed["swap_requested"],
            parsed["regenerate"],
        ]
    )
    if structured and not explicit:
        return False
    if named and known_name(named) and not explicit:
        return False
    if named and known_name(named) and explicit:
        return False
    return True


def _named(text: str) -> str | None:
    match = REQUEST_RE.search(text.strip())
    if not match:
        return None
    name = match.group(1)
    name = re.sub(r"\b(?:please|thanks|thank you)\b", "", name, flags=re.I)
    name = re.sub(r"\b(?:to|onto|into)\s+(?:my |the |this )?(?:plan|week|day)\b.*$", "", name, flags=re.I)
    name = re.sub(
        r"\b(?:for|as)\s+(?:a |an |my )?(?:breakfast|lunch|dinner|snack|exercise|meal)\b.*$",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"\s+", " ", name).strip(" ?.,!")
    if len(name) < 3 or name.lower() in GENERIC:
        return None
    return name


def _draft(text: str, profile: dict, use_llm: bool) -> dict | str:
    named = _named(text)
    kind = _kind(text, named or "")
    slot = _slot(text, named or "")
    item = None
    if use_llm:
        item = _from_model(text, profile, kind, slot, named)
    if item is None:
        item = _local_item(profile, kind, slot, named)
    problem = _problem(profile, item)
    if problem:
        return problem
    if item["kind"] == "meal":
        slot = item["slots"][0]
    sets, reps, rest = _prescription(profile)
    return {"kind": item["kind"], "item": item, "slot": slot, "sets": sets, "reps": reps, "rest_sec": rest}


def _kind(text: str, name: str) -> str:
    blob = f"{text} {name}"
    food = bool(FOOD_RE.search(blob))
    train = bool(TRAIN_RE.search(blob))
    if train and not food:
        return "exercise"
    return "meal"


def _slot(text: str, name: str) -> str:
    blob = f"{text} {name}".lower()
    for slot in ("breakfast", "lunch", "dinner", "snack"):
        if slot in blob:
            return slot
    return "dinner"


def _equipment(profile: dict) -> str:
    choice = profile.get("equipment") or "home"
    if choice == "home":
        return "dumbbells"
    if choice == "gym":
        return "gym"
    return choice


def _title(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split())


def _local_item(profile: dict, kind: str, slot: str, named: str | None) -> dict:
    cuisine = profile.get("cuisine") or "none"
    if kind == "exercise":
        name = _title(named) if named else "Tempo March"
        return {
            "id": "custom_" + uuid.uuid4().hex[:8],
            "kind": "exercise",
            "name": name[:60],
            "equipment": _equipment(profile),
            "focus": ["full"],
            "easy": True,
            "custom": True,
            "cue": "Move slowly and keep the reps under control.",
        }
    name = _title(named) if named else (_title(cuisine) + " Plate" if cuisine != "none" else "Coach's Rice Plate")
    words = [word for word in re.findall(r"[a-z]+", name.lower()) if word not in STOPWORDS and len(word) > 2]
    if slot in {"lunch", "dinner"} and "rice" not in words:
        words.append("rice")
    if not words:
        words = ["rice", "vegetables"]
    source = _source(words, profile)
    return {
        "id": "custom_" + uuid.uuid4().hex[:8],
        "kind": "meal",
        "name": name[:60],
        "slots": [slot],
        "diets": _diet_tags(source),
        "protein_source": source,
        "allergens": _allergens(words),
        "ingredients": words[:8],
        "protein_g": 32 if slot != "snack" else 14,
        "carbs_g": 40 if slot != "snack" else 18,
        "fat_g": 12 if slot != "snack" else 5,
        "priority": 0,
        "cuisines": [] if cuisine == "none" else [cuisine],
        "custom": True,
    }


def _source(words: list[str], profile: dict) -> str:
    blob = " ".join(words)
    for word, source in (
        ("chicken", "chicken"),
        ("beef", "beef"),
        ("pork", "pork"),
        ("turkey", "turkey"),
        ("fish", "fish"),
        ("shrimp", "shellfish"),
        ("prawn", "shellfish"),
        ("egg", "eggs"),
        ("eggs", "eggs"),
        ("tofu", "soy"),
        ("soy", "soy"),
        ("paneer", "dairy"),
        ("yogurt", "dairy"),
        ("cheese", "dairy"),
    ):
        if word in blob:
            return source
    diet = profile.get("diet") or "none"
    if diet in {"vegan", "vegetarian", "pescatarian"}:
        return "legumes"
    return "chicken"


def _diet_tags(source: str) -> list[str]:
    if source in MEAT_SOURCES or source == "chicken":
        return []
    if source in {"fish", "shellfish"}:
        return ["pescatarian"]
    if source in {"eggs", "dairy"}:
        return ["vegetarian", "pescatarian"]
    return ["vegan", "vegetarian", "pescatarian"]


def _allergens(words: list[str]) -> list[str]:
    blob = " ".join(words)
    found = []
    checks = {
        "eggs": ("egg", "eggs"),
        "gluten": ("bread", "pasta", "roti", "naan", "wheat"),
        "soy": ("soy", "tofu"),
        "dairy": ("paneer", "yogurt", "cheese", "milk"),
        "fish": ("fish",),
        "shellfish": ("shrimp", "prawn"),
        "nuts": ("almond", "peanut", "nuts"),
    }
    for allergen, tokens in checks.items():
        if any(token in blob for token in tokens):
            found.append(allergen)
    return found


def _from_model(text: str, profile: dict, kind: str, slot: str, named: str | None) -> dict | None:
    from fitfuel.llm import complete

    raw = complete(
        "Return one JSON object only. Draft a beginner gym item that is not already approved. "
        "Use an ordinary portion. Do not give medical advice.\n"
        f"User: {text}\nKind: {kind}. Name hint: {named or 'choose one'}. Slot: {slot}. "
        f"Diet: {profile.get('diet')}. Cuisine: {profile.get('cuisine')}. "
        f"Equipment: {profile.get('equipment')}. Avoid: {', '.join(profile.get('avoid') or [])}.\n"
        '{"kind":"meal","name":"","slot":"dinner","ingredients":["rice"],"protein_g":30,'
        '"carbs_g":40,"fat_g":10,"equipment":"dumbbells","focus":"full","cue":"Move slowly."}'
    )
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    model_kind = data.get("kind") if data.get("kind") in {"meal", "exercise"} else kind
    name = str(data.get("name") or named or "").strip()
    if len(name) < 3:
        return None
    if model_kind == "exercise":
        equipment = str(data.get("equipment") or _equipment(profile))
        if equipment not in EQUIPMENT_OK.get(profile.get("equipment"), set()):
            equipment = _equipment(profile)
        focus = str(data.get("focus") or "full")
        if focus not in {"lower", "push", "pull", "core", "full"}:
            focus = "full"
        return {
            "id": "custom_" + uuid.uuid4().hex[:8],
            "kind": "exercise",
            "name": _title(name)[:60],
            "equipment": equipment,
            "focus": [focus],
            "easy": True,
            "custom": True,
            "cue": str(data.get("cue") or "Move slowly and keep the reps under control.")[:160],
        }
    model_slot = data.get("slot") if data.get("slot") in {"breakfast", "lunch", "dinner", "snack"} else slot
    ingredients = [str(part).strip().lower() for part in data.get("ingredients") or [] if str(part).strip()]
    if not ingredients:
        ingredients = ["rice", "vegetables"]
    source = _source(ingredients, profile)
    return {
        "id": "custom_" + uuid.uuid4().hex[:8],
        "kind": "meal",
        "name": _title(name)[:60],
        "slots": [model_slot],
        "diets": _diet_tags(source),
        "protein_source": source,
        "allergens": _allergens(ingredients),
        "ingredients": ingredients[:8],
        "protein_g": _clamp(data.get("protein_g"), 8, 50, 30),
        "carbs_g": _clamp(data.get("carbs_g"), 0, 80, 40),
        "fat_g": _clamp(data.get("fat_g"), 0, 30, 10),
        "priority": 0,
        "cuisines": [] if (profile.get("cuisine") or "none") == "none" else [profile["cuisine"]],
        "custom": True,
    }


def _clamp(value, low: int, high: int, default: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


def _problem(profile: dict, item: dict) -> str | None:
    blob = item["name"].lower()
    if item["kind"] == "meal":
        blob += " " + " ".join(item.get("ingredients") or [])
    for token in blocked_tokens(profile):
        if re.search(rf"\b{re.escape(token)}\b", blob):
            return f"That includes {token}, which this profile leaves out, so I did not add it."
    if item["kind"] == "exercise":
        allowed = EQUIPMENT_OK[profile["equipment"]]
        if item["equipment"] not in allowed:
            return "That needs equipment you do not have, so I did not add it."
        return None
    if not meal_allowed(item, profile):
        return "That meal does not fit this diet or the foods you leave out, so I did not add it."
    return None


def _prescription(profile: dict) -> tuple[int, str, int]:
    if profile.get("goal") == "muscle_gain":
        return 4, "8-12", 90
    if profile.get("goal") == "fat_loss":
        return 3, "12-15", 50
    return 3, "10-12", 60


def _ask(draft: dict) -> str:
    item = draft["item"]
    if item["kind"] == "exercise":
        return (
            f"{item['name']} is not on the list yet. I can add it to day 1 "
            f"({item['equipment']}, {draft['sets']} x {draft['reps']}). Should I add it to this week?"
        )
    calories = meal_calories(item)
    foods = ", ".join(item["ingredients"])
    return (
        f"{item['name']} is not on the list yet. I can add it as {draft['slot']}: "
        f"about {calories} kcal, with {foods}. Should I add it to this week?"
    )


def _accept(user_id: str, profile: dict, plan: dict | None, pending: dict) -> dict:
    store.clear_pending(user_id)
    if not plan:
        return _reply(plan, "suggest", "Build a week first. Then I can add it.")
    item = pending["item"]
    updated = json.loads(json.dumps(plan))
    if item["kind"] == "exercise":
        note = _place_exercise(updated, pending)
    else:
        note = _place_meal(updated, pending, profile)
    if note is None:
        return _reply(plan, "suggest", "I could not fit that into this week, so I left the plan as it is.")
    store.save_custom(item)
    updated["validation"] = validate(profile, updated)
    if not updated["validation"].get("ok"):
        store.delete_custom(item["id"])
        issues = "; ".join(updated["validation"].get("issues") or [])
        return _reply(plan, "suggest", f"I checked it and did not add it. {issues}")
    for key in ("id", "created_at", "is_current"):
        updated.pop(key, None)
    store.update_plan(user_id, updated)
    saved = store.get_current_plan(user_id)
    return _reply(saved, "suggest_added", note, f"Added {item['name']} after a yes.")


def _place_exercise(plan: dict, pending: dict) -> str | None:
    days = plan.get("workouts") or []
    if not days or not days[0].get("exercises"):
        return None
    item = pending["item"]
    current = days[0]["exercises"][-1]
    days[0]["exercises"][-1] = {
        "id": item["id"],
        "name": item["name"],
        "equipment": item["equipment"],
        "sets": pending["sets"],
        "reps": pending["reps"],
        "rest_sec": pending["rest_sec"],
        "cue": item["cue"],
    }
    return f"Added {item['name']} to {days[0].get('title', 'day 1')}. It replaced {current['name']}."


def _place_meal(plan: dict, pending: dict, profile: dict) -> str | None:
    item = pending["item"]
    slot = pending.get("slot") or "dinner"
    served = _served(item, slot, profile)
    if served is None:
        return None
    meals = plan.get("meals") or []
    replaced = None
    for index, meal in enumerate(meals):
        if meal.get("slot") == slot:
            replaced = meal.get("name")
            meals[index] = served
            break
    if replaced is None:
        meals.append(served)
    plan["meals"] = meals
    plan["meal_totals"] = {
        "calories": sum(int(meal["calories"]) for meal in meals),
        "protein_g": sum(int(meal["protein_g"]) for meal in meals),
        "carbs_g": sum(int(meal.get("carbs_g") or 0) for meal in meals),
        "fat_g": sum(int(meal.get("fat_g") or 0) for meal in meals),
    }
    if replaced:
        return f"Added {item['name']} as {slot}. It replaced {replaced}."
    return f"Added {item['name']} as {slot}."


def _served(item: dict, slot: str, profile: dict) -> dict | None:
    from fitfuel.planning import _portion, _scaled, _slot_targets, macro_targets

    targets, _notes = macro_targets(profile)
    cap = profile.get("dinner_calorie_cap")
    portion = _portion(item, _slot_targets(targets["calories"], cap)[slot], cap, slot)
    if portion is None:
        return None
    return _scaled(item, portion, cap, slot)


def place_request(
    profile: dict,
    plan: dict | None,
    message: str,
    use_llm: bool,
    name: str | None = None,
    slot: str | None = None,
    history: str = "",
) -> dict:
    if not plan or not (plan.get("meals") or plan.get("workouts")):
        return {"applied": False, "notes": ["Build a week first. Then I can add it."], "custom_id": None}
    dish = resolve_dish(message, history, name)
    if not dish:
        return {"applied": False, "notes": ["Tell me which dish you want added."], "custom_id": None}
    if _kind(f"{message} {dish}", dish) == "exercise":
        return _place_draft(plan, profile, f"Add {dish}", use_llm, slot)
    ingredients = ingredient_words(dish)
    if use_llm:
        extra = _model_ingredients(message, history, profile, dish)
        for item in extra:
            if item not in ingredients:
                ingredients.append(item)
    kept, omitted = apply_omit(ingredients, message, profile)
    hits = conflicts(kept, profile)
    if hits:
        return {"applied": False, "notes": [refusal(dish, kept, hits)], "custom_id": None}
    if not kept:
        return {
            "applied": False,
            "notes": [f"Nothing of {dish} is left once those ingredients are removed, so I did not add it."],
            "custom_id": None,
        }
    chosen = slot or guess_slot(dish, message)
    item = _local_item(profile, "meal", chosen, cleaned_name(dish, omitted))
    item["ingredients"] = kept[:8]
    item["allergens"] = _allergens(kept)
    item["name"] = cleaned_name(dish, omitted)
    problem = _problem(profile, item)
    if problem:
        return {"applied": False, "notes": [problem], "custom_id": None}
    draft = {
        "kind": "meal",
        "item": item,
        "slot": chosen,
        "sets": _prescription(profile)[0],
        "reps": _prescription(profile)[1],
        "rest_sec": _prescription(profile)[2],
    }
    updated = json.loads(json.dumps(plan))
    note = _place_meal(updated, draft, profile)
    if note is None:
        return {
            "applied": False,
            "notes": ["I could not fit that into this week, so I left the plan as it is."],
            "custom_id": None,
        }
    if omitted:
        note += " Left out " + ", ".join(omitted) + "."
    store.save_custom(item)
    return {
        "applied": True,
        "notes": [note],
        "custom_id": item["id"],
        "workouts": updated.get("workouts") or [],
        "meals": updated.get("meals") or [],
        "meal_totals": updated.get("meal_totals") or {},
    }


def _place_draft(plan: dict, profile: dict, text: str, use_llm: bool, slot: str | None) -> dict:
    draft = _draft(text, profile, use_llm)
    if isinstance(draft, str):
        return {"applied": False, "notes": [draft], "custom_id": None}
    if slot and draft["item"]["kind"] == "meal":
        draft["slot"] = slot
    updated = json.loads(json.dumps(plan))
    if draft["item"]["kind"] == "exercise":
        note = _place_exercise(updated, draft)
    else:
        note = _place_meal(updated, draft, profile)
    if note is None:
        return {
            "applied": False,
            "notes": ["I could not fit that into this week, so I left the plan as it is."],
            "custom_id": None,
        }
    store.save_custom(draft["item"])
    return {
        "applied": True,
        "notes": [note],
        "custom_id": draft["item"]["id"],
        "workouts": updated.get("workouts") or [],
        "meals": updated.get("meals") or [],
        "meal_totals": updated.get("meal_totals") or {},
    }


def _model_ingredients(message: str, history: str, profile: dict, dish: str) -> list[str]:
    from fitfuel.llm import complete

    limits = ", ".join(list(profile.get("allergies") or []) + list(profile.get("avoid") or [])) or "none"
    raw = complete(
        "Return JSON only: {\"ingredients\":[\"oats\"]}. "
        "List the usual ingredient words for this dish. Do not drop one because of an allergy.\n"
        f"Dish: {dish}\nLimits: {limits}\nNotes: {profile.get('notes') or 'none'}\n"
        f"Recent chat:\n{history or 'None'}\nUser: {message}"
    )
    if not raw:
        return []
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    words = []
    for part in data.get("ingredients") or []:
        token = re.sub(r"[^a-z]", "", str(part).lower())
        if len(token) > 2 and token not in words:
            words.append(token)
    return words[:8]


def _reply(plan: dict | None, intent: str, answer: str, detail: str = "Suggestion only. The plan was not changed.") -> dict:
    return {
        "answer": answer,
        "plan": plan,
        "trace": [{"agent": "Coach Agent", "tool": "suggestion", "detail": detail}],
        "intent": intent,
    }
