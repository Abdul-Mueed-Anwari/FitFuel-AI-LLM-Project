from __future__ import annotations

import re

GENERIC_SWAPS = {
    "this",
    "that",
    "it",
    "one",
    "exercise",
    "this exercise",
    "that exercise",
    "my exercise",
}

MEDICAL_RE = re.compile(
    r"\b(injury|injured|pain|hurt|torn|surgery|pregnant|pregnancy|diabetes|cancer|"
    r"dizzy|dizziness|diagnosis|diagnose|disorder|medication|painkiller|laxative|"
    r"anorexia|purge|starve|chest pains?)\b",
    re.I,
)
DINNER_RE = re.compile(
    r"(?:dinner|dinners|evening meal|evening meals).{0,30}?(\d{3,4})",
    re.I,
)
DAILY_RE = re.compile(
    r"(\d{3,4})\s*(?:kcal|calories?)\s*(?:a day|per day|daily|each day|diet|plan)\b",
    re.I,
)
LOOSE_CAL_RE = re.compile(r"(\d{3,4})\s*(?:kcal|calories?)", re.I)
MINUTES_RE = re.compile(r"(\d{2,3})\s*minutes?\b", re.I)
DAYS_RE = re.compile(r"\b([2-6])\s*days?\b", re.I)
SWAP_FOR_RE = re.compile(r"(?:swap|replace)\s+(.+?)\s+(?:for|with)\s+(.+)", re.I)
SWAP_RE = re.compile(r"(?:swap|replace)\s+(?:the\s+|this\s+|my\s+)?(.+)", re.I)
AVOID_RE = re.compile(
    r"(?:no|without|avoid|allergic to|allergy to)\s+(?:any\s+)?(dairy|milk|lactose|gluten|eggs?|nuts|peanuts?|soy|shellfish|fish|pork)",
    re.I,
)
FREE_RE = re.compile(r"\b(dairy|gluten|nut|peanut|soy|shellfish|pork)[-\s]?free\b", re.I)


def medical_reason(text: str) -> str | None:
    if not text:
        return None
    if _extreme_calories(text):
        return "extreme"
    if MEDICAL_RE.search(text):
        return "injury"
    return None


def _extreme_calories(text: str) -> bool:
    stripped = DINNER_RE.sub(" ", text)
    if re.search(r"\b(dinner|breakfast|lunch|snack|meal)s?\b", text, re.I) and DINNER_RE.search(text):
        stripped = DINNER_RE.sub(" ", text)
    for match in DAILY_RE.finditer(stripped):
        if int(match.group(1)) < 1400:
            return True
    if re.search(r"\b(dinner|breakfast|lunch|snack|meal)s?\b", text, re.I):
        return False
    for match in LOOSE_CAL_RE.finditer(stripped):
        if int(match.group(1)) < 1400:
            return True
    return False


def _clean_phrase(text: str) -> str:
    return text.strip(" ?.,!").lower()


def _limit_token(word: str) -> str:
    word = word.lower().rstrip("s") if word.lower() not in {"eggs"} else word.lower()
    mapping = {
        "milk": "dairy",
        "lactose": "dairy",
        "egg": "eggs",
        "eggs": "eggs",
        "nut": "nuts",
        "nuts": "nuts",
        "peanut": "peanuts",
        "peanuts": "peanuts",
        "dairy": "dairy",
        "gluten": "gluten",
        "soy": "soy",
        "shellfish": "shellfish",
        "fish": "fish",
        "pork": "pork",
    }
    return mapping.get(word, word)


def extract(message: str) -> dict:
    text = message.strip()
    updates = {
        "diet": None,
        "cuisine": None,
        "add_avoid": [],
        "days_per_week": None,
        "session_minutes": None,
        "equipment": None,
        "goal": None,
        "dinner_calorie_cap": None,
        "swap_requested": False,
        "swap_from": None,
        "prefer_name": None,
        "prefer_easy": False,
        "medical": False,
        "medical_kind": None,
        "regenerate": False,
    }
    reason = medical_reason(text)
    if reason:
        updates["medical"] = True
        updates["medical_kind"] = reason

    dinner = DINNER_RE.search(text)
    if dinner:
        cap = int(dinner.group(1))
        if 250 <= cap <= 900:
            updates["dinner_calorie_cap"] = cap

    if re.search(r"\bvegan\b", text, re.I):
        updates["diet"] = "vegan"
    elif re.search(r"\bvegetarian\b", text, re.I):
        updates["diet"] = "vegetarian"
    elif re.search(r"\bpescatarian\b", text, re.I):
        updates["diet"] = "pescatarian"

    for cuisine, pattern in (
        ("pakistani", r"pakistani|desi"),
        ("chinese", r"chinese"),
        ("indian", r"indian"),
        ("mediterranean", r"mediterranean"),
        ("mexican", r"mexican"),
        ("italian", r"italian"),
    ):
        if re.search(rf"\b(?:{pattern})\b", text, re.I):
            updates["cuisine"] = cuisine
            break

    for match in list(AVOID_RE.finditer(text)) + list(FREE_RE.finditer(text)):
        token = _limit_token(match.group(1))
        if token not in updates["add_avoid"]:
            updates["add_avoid"].append(token)

    minutes = MINUTES_RE.search(text)
    if minutes:
        value = int(minutes.group(1))
        if 20 <= value <= 90:
            updates["session_minutes"] = 30 if value <= 37 else 45 if value <= 52 else 60
    days = DAYS_RE.search(text)
    if days:
        updates["days_per_week"] = int(days.group(1))

    lowered = text.lower()
    if "full gym" in lowered or "gym machine" in lowered:
        updates["equipment"] = "gym"
    elif "home gym" in lowered:
        updates["equipment"] = "home"
    elif re.search(r"\bdumb+ells?\b", lowered):
        updates["equipment"] = "dumbbells"
    elif "band" in lowered:
        updates["equipment"] = "bands"
    elif "bodyweight" in lowered or "no equipment" in lowered:
        updates["equipment"] = "bodyweight"

    if re.search(r"\b(fat loss|lose fat|lose weight)\b", text, re.I):
        updates["goal"] = "fat_loss"
    elif re.search(r"\b(muscle|bulk)\b", text, re.I):
        updates["goal"] = "muscle_gain"
    elif re.search(r"\b(body recomp\w*|recomp|maintain)\b", text, re.I):
        updates["goal"] = "maintain"

    if re.search(r"\b(easier|too hard|simpler)\b", text, re.I):
        updates["prefer_easy"] = True
    if re.search(r"\b(start over|new plan|regenerate|rebuild)\b", text, re.I):
        updates["regenerate"] = True

    swap_for = SWAP_FOR_RE.search(text)
    if swap_for:
        updates["swap_requested"] = True
        updates["swap_from"] = _swap_name(swap_for.group(1))
        updates["prefer_name"] = _clean_phrase(swap_for.group(2)) or None
    else:
        swap = SWAP_RE.search(text)
        if swap:
            updates["swap_requested"] = True
            updates["swap_from"] = _swap_name(swap.group(1))
    return updates


def _swap_name(raw: str) -> str | None:
    cleaned = _clean_phrase(raw)
    if cleaned in GENERIC_SWAPS:
        return None
    return cleaned or None


EQUIPMENT_WORDS = {
    "dumbbell",
    "dumbbells",
    "dumbell",
    "dumbells",
    "band",
    "bands",
    "bodyweight",
    "gym",
    "home gym",
    "no equipment",
}
REFERENCE_WORDS = {
    "it",
    "that",
    "this",
    "them",
    "those",
    "then",
    "okay",
    "ok",
    "please",
    "just",
    "now",
    "too",
    "also",
    "one",
    "thing",
}
ADD_RE = re.compile(r"\b(?:add|include|try)\s+(?:a |an |the )?(?:new )?(.+)$", re.I)
SUGGEST_RE = re.compile(
    r"\b(suggest|not on the list|isn't on the list|isnt on the list|"
    r"new exercise|new meal|new dinner|new breakfast|new lunch|new snack)\b",
    re.I,
)


def _concrete(parsed: dict) -> bool:
    return any(
        [
            parsed.get("diet"),
            parsed.get("cuisine"),
            parsed.get("add_avoid"),
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


def _add_name(raw: str) -> str:
    name = re.sub(r"\b(?:please|thanks|thank you)\b", "", raw, flags=re.I)
    name = re.sub(r"\b(?:to|onto|into)\s+(?:my |the |this )?(?:plan|week|day)\b.*$", "", name, flags=re.I)
    name = re.sub(
        r"\b(?:for|as)\s+(?:a |an |my )?(?:breakfast|lunch|dinner|snack|exercise|meal)\b.*$",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"\s+", " ", name).strip(" ?.,!")
    words = re.findall(r"[a-z']+", name.lower())
    if words and all(word in REFERENCE_WORDS for word in words):
        return ""
    return name


def add_request(message: str) -> str | None:
    text = message.strip()
    if SUGGEST_RE.search(text):
        match = ADD_RE.search(text)
        name = _add_name(match.group(1)) if match else ""
        if name.lower() in EQUIPMENT_WORDS:
            return None
        return name
    match = ADD_RE.search(text)
    if not match:
        return None
    name = _add_name(match.group(1))
    if name == "":
        return ""
    if len(name) < 3 or name.lower() in EQUIPMENT_WORDS:
        return None
    return name


def mentioned_slot(message: str) -> str | None:
    for slot in ("breakfast", "lunch", "dinner", "snack"):
        if re.search(rf"\b{slot}\b", message, re.I):
            return slot
    return None


def classify(message: str, has_plan: bool, force_intent: str | None = None) -> dict:
    parsed = extract(message)
    if parsed["medical"]:
        parsed["intent"] = "medical"
        return parsed
    if force_intent:
        parsed["intent"] = force_intent
        return parsed
    if parsed["regenerate"] or not has_plan and _wants_plan(message):
        parsed["intent"] = "generate"
        return parsed
    target = add_request(message)
    if target is not None and not _concrete(parsed):
        parsed["intent"] = "add"
        parsed["add_name"] = target or None
        parsed["add_slot"] = mentioned_slot(message)
        parsed["add_from_history"] = target == ""
        return parsed

    meal = bool(
        parsed["diet"]
        or parsed["cuisine"]
        or parsed["add_avoid"]
        or parsed["dinner_calorie_cap"]
        or re.search(r"\b(meal|dinner|breakfast|lunch|snack|calorie|calories|kcal|food)\b", message, re.I)
    )
    workout = bool(
        parsed["swap_requested"]
        or parsed["session_minutes"]
        or parsed["days_per_week"]
        or parsed["equipment"]
        or parsed["prefer_easy"]
        or re.search(r"\b(workouts?|exercises?|sessions?|training|gym)\b", message, re.I)
    )
    if parsed["goal"]:
        meal = True
        workout = True
    if meal and workout:
        intent = "update_both"
    elif meal:
        intent = "update_meals"
    elif workout:
        intent = "update_workout"
    elif re.search(r"^(what|how|why|who|when|explain|tell me)\b", message.strip(), re.I) or "?" in message:
        intent = "question"
    elif has_plan:
        intent = "question"
    else:
        intent = "generate"
    if not has_plan and intent in {"update_meals", "update_workout", "update_both"}:
        intent = "generate"
    parsed["intent"] = intent
    return parsed


def _wants_plan(message: str) -> bool:
    return bool(re.search(r"\b(plan|workout|meal|schedule)\b", message, re.I))


def apply_updates(profile: dict, updates: dict) -> dict:
    profile = dict(profile)
    profile["avoid"] = list(profile.get("avoid") or [])
    profile["allergies"] = list(profile.get("allergies") or [])
    for key in ("diet", "cuisine", "days_per_week", "session_minutes", "equipment", "goal"):
        if updates.get(key):
            profile[key] = updates[key]
    if updates.get("dinner_calorie_cap"):
        profile["dinner_calorie_cap"] = updates["dinner_calorie_cap"]
    for token in updates.get("add_avoid") or []:
        if token not in profile["avoid"]:
            profile["avoid"].append(token)
    from fitfuel.profiles import clean_profile

    return clean_profile(profile)
