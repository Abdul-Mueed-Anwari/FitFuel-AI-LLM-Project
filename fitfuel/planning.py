from __future__ import annotations

import json

from fitfuel.catalog import exercise_by_id, exercises, meal_by_id, meal_calories, meals
from fitfuel.profiles import ACTIVITY, EQUIPMENT_OK, GOAL_MULTIPLIER
from fitfuel.rules import exercise_count, meal_allowed
from fitfuel.tools import calculator, knowledge_retriever


def _calc(expression: str) -> float:
    return float(calculator.invoke({"expression": expression}))


def macro_targets(profile: dict) -> tuple[dict, list[str]]:
    weight = profile["weight_kg"]
    height = profile["height_cm"]
    age = profile["age"]
    offset = {"male": "+ 5", "female": "- 161"}.get(profile["sex"], "- 78")
    factor = ACTIVITY[int(profile["days_per_week"])]
    mult = GOAL_MULTIPLIER[profile["goal"]]
    expression = f"((10*{weight} + 6.25*{height} - 5*{age} {offset}) * {factor}) * {mult}"
    raw = round(_calc(expression))
    calories = min(3800, max(1500, raw))
    notes = []
    if calories != raw:
        notes.append(f"The formula came out at {raw} kcal, so the target stays at {calories} kcal.")
    per_kg = 1.6 if profile["goal"] == "maintain" else 1.8
    protein = round(_calc(f"{weight} * {per_kg}"))
    fat = round(_calc(f"{weight} * 0.8"))
    carb_calories = _calc(f"{calories} - ({protein} * 4) - ({fat} * 9)")
    if carb_calories < 80:
        fat = round(_calc(f"{weight} * 0.6"))
        carb_calories = _calc(f"{calories} - ({protein} * 4) - ({fat} * 9)")
    carbs = max(0, round(_calc(f"{max(carb_calories, 0)} / 4")))
    targets = {
        "calories": calories,
        "protein_g": protein,
        "carbs_g": carbs,
        "fat_g": fat,
        "method": "Mifflin-St Jeor with a light activity factor",
    }
    return targets, notes


def _day_slots(focus: str, count: int) -> tuple[str, list[str]]:
    if focus == "lower":
        return "Lower body", ["lower"] * count
    if focus == "upper":
        return "Upper body", ["push" if index % 2 == 0 else "pull" for index in range(count)]
    if focus == "core":
        return "Core", ["core"] * count
    pattern = ["lower", "push", "pull", "core", "lower"]
    return "Full body", [pattern[index % len(pattern)] for index in range(count)]


def _matches(exercise: dict, slot: str) -> bool:
    focus = set(exercise["focus"])
    if slot == "full":
        return bool(focus & {"full", "lower", "push", "pull"})
    return slot in focus


def _prescription(profile: dict) -> tuple[int, str, int]:
    if profile["session_minutes"] <= 30:
        return 3, "10-12", 45
    if profile["goal"] == "muscle_gain":
        return 4, "8-12", 90
    if profile["goal"] == "fat_loss":
        return 3, "12-15", 50
    return 3, "10-12", 60


def _fresh_workout(profile: dict, prefer_easy: bool, avoid: set[str] | None = None) -> list[dict]:
    allowed = EQUIPMENT_OK[profile["equipment"]]
    count = exercise_count(profile["session_minutes"])
    sets, reps, rest = _prescription(profile)
    focuses = {
        2: ["full", "full"],
        3: ["full", "lower", "upper"],
        4: ["lower", "upper", "lower", "upper"],
        5: ["full", "lower", "upper", "full", "core"],
        6: ["lower", "upper", "full", "lower", "upper", "core"],
    }[int(profile["days_per_week"])]
    used: set[str] = set()
    days = []
    for index, focus in enumerate(focuses, start=1):
        title, slots = _day_slots(focus, count)
        raw = knowledge_retriever.invoke({"query": f"{focus} beginner {profile['equipment']}", "kind": "exercise"})
        preferred_ids = [hit["id"] for hit in json.loads(raw)]
        exercises_today = []
        day_ids: set[str] = set()
        for slot in slots:
            pick = _pick_exercise(slot, allowed, used, day_ids, preferred_ids, prefer_easy, avoid)
            if pick is None:
                continue
            used.add(pick["id"])
            day_ids.add(pick["id"])
            exercises_today.append(
                {
                    "id": pick["id"],
                    "name": pick["name"],
                    "equipment": pick["equipment"],
                    "sets": sets,
                    "reps": reps,
                    "rest_sec": rest,
                    "cue": pick["cue"],
                }
            )
        days.append({"title": f"Day {index} · {title}", "focus": focus, "minutes": profile["session_minutes"], "exercises": exercises_today})
    return days


def _pick_exercise(slot, allowed, used, day_ids, preferred_ids, prefer_easy, avoid=None):
    avoid = avoid or set()
    pool = []
    for item_id in preferred_ids:
        item = exercise_by_id(item_id)
        if item and _matches(item, slot) and item["equipment"] in allowed:
            pool.append(item)
    for item in exercises():
        if item not in pool and _matches(item, slot) and item["equipment"] in allowed:
            pool.append(item)
    if prefer_easy:
        pool.sort(key=lambda item: (0 if item.get("easy") else 1, item["name"]))
    if avoid:
        pool = [item for item in pool if item["id"] not in avoid] + [item for item in pool if item["id"] in avoid]
    for item in pool:
        if item["id"] not in used and item["id"] not in day_ids:
            return item
    for item in pool:
        if item["id"] not in day_ids:
            return item
    return pool[0] if pool else None


def _name_matches(target: str, name: str) -> bool:
    name = name.lower()
    target = target.lower().strip()
    options = {target}
    if target.endswith("s"):
        options.add(target[:-1])
    if target.endswith("es"):
        options.add(target[:-2])
    return any(option and option in name for option in options)


def _swap(days: list[dict], profile: dict, updates: dict) -> tuple[list[dict], list[str]]:
    days = json.loads(json.dumps(days))
    target = updates.get("swap_from") or ""
    found = None
    for day_index, day in enumerate(days):
        for exercise_index, exercise in enumerate(day["exercises"]):
            if not target or _name_matches(target, exercise["name"]):
                found = (day_index, exercise_index, exercise)
                break
        if found:
            break
    if found is None:
        return days, ["I could not find that exercise in the current plan."]
    day_index, exercise_index, current = found
    day = days[day_index]
    slot = _slot_for_exercise(current["id"])
    allowed = EQUIPMENT_OK[profile["equipment"]]
    blocked = {item["id"] for item in day["exercises"]}
    prefer = (updates.get("prefer_name") or "").lower()
    raw = knowledge_retriever.invoke(
        {"query": f"{slot} beginner {profile['equipment']} {prefer}".strip(), "kind": "exercise"}
    )
    preferred = [hit["id"] for hit in json.loads(raw)]
    pool = []
    for item_id in preferred:
        item = exercise_by_id(item_id)
        if item and item["id"] not in blocked and item["equipment"] in allowed and _matches(item, slot):
            pool.append(item)
    for item in exercises():
        if item not in pool and item["id"] not in blocked and item["equipment"] in allowed and _matches(item, slot):
            pool.append(item)
    if updates.get("prefer_easy"):
        pool.sort(key=lambda item: (0 if item.get("easy") else 1, item["name"]))
    if prefer:
        named = [item for item in pool if prefer in item["name"].lower()]
        if named:
            pool = named
    if not pool:
        return days, [f"I could not find another {slot} exercise for your equipment."]
    replacement = pool[0]
    sets, reps, rest = _prescription(profile)
    day["exercises"][exercise_index] = {
        "id": replacement["id"],
        "name": replacement["name"],
        "equipment": replacement["equipment"],
        "sets": sets,
        "reps": reps,
        "rest_sec": rest,
        "cue": replacement["cue"],
    }
    return days, [f"Swapped {current['name']} for {replacement['name']} on {day['title']}."]


def _slot_for_exercise(item_id: str) -> str:
    item = exercise_by_id(item_id)
    if not item:
        return "full"
    for slot in ("lower", "push", "pull", "core"):
        if slot in item["focus"]:
            return slot
    return "full"


def make_workout(profile: dict, updates: dict, previous: list[dict] | None) -> tuple[list[dict], list[str]]:
    previous = previous or []
    schedule_changed = bool(previous) and (
        updates.get("session_minutes")
        or updates.get("days_per_week")
        or updates.get("equipment")
        or updates.get("goal")
    )
    if not previous or schedule_changed or updates.get("regenerate"):
        days = _fresh_workout(
            profile,
            bool(updates.get("prefer_easy") or profile["session_minutes"] <= 30),
            set(updates.get("avoid_ids") or []),
        )
        notes = ["Built the weekly workout from the exercise list."]
        if updates.get("avoid_ids"):
            notes.append("This week uses different sessions from the last one.")
    else:
        days = previous
        notes = []
    if updates.get("swap_requested"):
        days, swap_notes = _swap(days, profile, updates)
        notes.extend(swap_notes)
    return days, notes


def _slot_targets(calories: int, cap: int | None) -> dict[str, float]:
    shares = {"breakfast": 0.25, "lunch": 0.30, "dinner": 0.30, "snack": 0.15}
    targets = {slot: calories * share for slot, share in shares.items()}
    if cap and targets["dinner"] > cap:
        extra = targets["dinner"] - cap
        targets["dinner"] = float(cap)
        others = ["breakfast", "lunch", "snack"]
        total = sum(shares[slot] for slot in others)
        for slot in others:
            targets[slot] += extra * (shares[slot] / total)
    return targets


def _portion(meal: dict, target: float, cap: int | None, slot: str) -> float | None:
    raw = meal_calories(meal)
    if raw <= 0:
        return None
    portion = min(1.25, max(0.7, target / raw))
    if slot == "dinner" and cap:
        portion = min(portion, cap / raw)
        if portion < 0.65:
            return None
    return round(portion, 2)


def _scaled(meal: dict, portion: float, cap: int | None, slot: str) -> dict | None:
    for _ in range(4):
        protein = round(meal["protein_g"] * portion)
        carbs = round(meal["carbs_g"] * portion)
        fat = round(meal["fat_g"] * portion)
        calories = round(protein * 4 + carbs * 4 + fat * 9)
        if not (slot == "dinner" and cap and calories > cap):
            return {
                "id": meal["id"],
                "slot": slot,
                "name": meal["name"],
                "ingredients": meal["ingredients"],
                "portion": portion,
                "calories": calories,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
            }
        portion = round(portion - 0.05, 2)
        if portion < 0.65:
            return None
    return None


def make_meals(profile: dict, updates: dict, first: bool) -> tuple[list[dict], dict, dict, list[str]]:
    targets, macro_notes = macro_targets(profile)
    cap = profile.get("dinner_calorie_cap")
    slot_targets = _slot_targets(targets["calories"], cap)
    chosen = []
    used: set[str] = set()
    for slot in ("breakfast", "lunch", "dinner", "snack"):
        diet_word = profile["diet"] if profile["diet"] != "none" else ""
        cuisine_word = profile.get("cuisine") or ""
        if cuisine_word == "none":
            cuisine_word = ""
        raw = knowledge_retriever.invoke({"query": f"{slot} {diet_word} {cuisine_word} meal".strip(), "kind": "meal"})
        preferred = [hit["id"] for hit in json.loads(raw)]
        pool = []
        for item_id in preferred:
            meal = meal_by_id(item_id)
            if meal and slot in meal["slots"] and meal_allowed(meal, profile):
                pool.append(meal)
        for meal in meals():
            if meal not in pool and slot in meal["slots"] and meal_allowed(meal, profile):
                pool.append(meal)
        primary = [meal for meal in pool if meal.get("priority", 1) == 1] or pool
        cuisine = profile.get("cuisine") or "none"
        avoid = set(updates.get("avoid_ids") or [])
        ranked = sorted(
            primary,
            key=lambda meal: (
                1 if meal["id"] in avoid else 0,
                0 if cuisine not in ("", "none") and cuisine in (meal.get("cuisines") or []) else 1,
                abs((meal_calories(meal) * (_portion(meal, slot_targets[slot], cap, slot) or 1)) - slot_targets[slot]),
                meal["name"],
            ),
        )
        picked = None
        served = None
        for meal in ranked:
            if meal["id"] in used and len(ranked) > 1:
                continue
            portion = _portion(meal, slot_targets[slot], cap, slot)
            if portion is None:
                continue
            served = _scaled(meal, portion, cap, slot)
            if served:
                picked = meal
                break
        if served and picked:
            used.add(picked["id"])
            chosen.append(served)
    totals = {
        "calories": sum(meal["calories"] for meal in chosen),
        "protein_g": sum(meal["protein_g"] for meal in chosen),
        "carbs_g": sum(meal["carbs_g"] for meal in chosen),
        "fat_g": sum(meal["fat_g"] for meal in chosen),
    }
    notes = list(macro_notes)
    if updates.get("diet"):
        notes.append(f"Meals now follow a {updates['diet']} pattern.")
    if updates.get("cuisine"):
        notes.append(f"Meals now lean toward {updates['cuisine']} food.")
    if updates.get("add_avoid"):
        notes.append("Food limits now include " + ", ".join(updates["add_avoid"]) + ".")
    if first and not notes:
        notes.append("Chose meals from the approved list.")
    return chosen, totals, targets, notes
