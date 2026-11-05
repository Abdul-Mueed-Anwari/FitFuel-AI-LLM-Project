from __future__ import annotations

EQUIPMENT_OK = {
    "bodyweight": {"bodyweight"},
    "dumbbells": {"bodyweight", "dumbbells"},
    "bands": {"bodyweight", "bands"},
    "home": {"bodyweight", "dumbbells", "bands"},
    "gym": {"bodyweight", "dumbbells", "bands", "gym"},
}

GOAL_LABELS = {
    "fat_loss": "fat loss",
    "muscle_gain": "muscle gain",
    "maintain": "body recomposition",
}

CUISINE_OK = {"none", "pakistani", "chinese", "indian", "mediterranean", "mexican", "italian"}

EQUIPMENT_LABELS = {
    "bodyweight": "bodyweight only",
    "dumbbells": "dumbbells",
    "bands": "resistance bands",
    "home": "a home gym",
    "gym": "a full gym",
}

ACTIVITY = {2: 1.375, 3: 1.45, 4: 1.55, 5: 1.6, 6: 1.725}
GOAL_MULTIPLIER = {"fat_loss": 0.85, "muscle_gain": 1.1, "maintain": 1.0}
MEAT_SOURCES = {"chicken", "turkey", "beef", "pork"}
KNOWN_LIMITS = ["dairy", "eggs", "gluten", "nuts", "peanuts", "soy", "shellfish", "fish", "pork"]


def clean_profile(raw: dict) -> dict:
    goal = raw.get("goal") or "fat_loss"
    if goal not in GOAL_LABELS:
        goal = "fat_loss"
    equipment = raw.get("equipment") or "home"
    if equipment not in EQUIPMENT_OK:
        equipment = "home"
    diet = raw.get("diet") or "none"
    if diet not in {"none", "vegetarian", "vegan", "pescatarian"}:
        diet = "none"
    cuisine = raw.get("cuisine") or "none"
    if cuisine not in CUISINE_OK:
        cuisine = "none"
    sex = raw.get("sex") or "female"
    if sex not in {"female", "male", "other"}:
        sex = "female"
    days = int(raw.get("days_per_week") or 3)
    days = min(6, max(2, days))
    minutes = int(raw.get("session_minutes") or 45)
    if minutes <= 30:
        minutes = 30
    elif minutes <= 45:
        minutes = 45
    else:
        minutes = 60
    avoid = []
    for item in list(raw.get("avoid") or []):
        token = str(item).strip().lower()
        if token and token not in avoid:
            avoid.append(token)
    allergies = []
    for item in list(raw.get("allergies") or []):
        token = str(item).strip().lower()
        if token and token not in allergies:
            allergies.append(token)
    cap = raw.get("dinner_calorie_cap")
    if cap in ("", None):
        cap = None
    else:
        cap = int(cap)
        if cap < 250 or cap > 900:
            cap = None
    name = str(raw.get("name") or "Friend").strip() or "Friend"
    return {
        "name": name[:40],
        "age": int(raw.get("age") or 25),
        "sex": sex,
        "weight_kg": float(raw.get("weight_kg") or 70),
        "height_cm": float(raw.get("height_cm") or 170),
        "goal": goal,
        "days_per_week": days,
        "session_minutes": minutes,
        "equipment": equipment,
        "diet": diet,
        "cuisine": cuisine,
        "avoid": avoid,
        "allergies": allergies,
        "notes": str(raw.get("notes") or "").strip()[:400],
        "dinner_calorie_cap": cap,
    }
