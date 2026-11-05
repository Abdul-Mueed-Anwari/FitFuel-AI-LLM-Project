from __future__ import annotations

import ast
import re

from fitfuel.catalog import exercise_by_id, meal_by_id
from fitfuel.profiles import EQUIPMENT_OK, MEAT_SOURCES


def calculate(expression: str) -> float:
    expression = expression.strip()
    if not expression or len(expression) > 240:
        raise ValueError("Expression is empty or too long")
    tree = ast.parse(expression, mode="eval")
    return float(_eval(tree.body))


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("Division by zero")
        return left / right
    raise ValueError("Only simple arithmetic is allowed")


def blocked_tokens(profile: dict) -> set[str]:
    tokens = {str(item).strip().lower() for item in profile.get("avoid") or []}
    tokens.update(str(item).strip().lower() for item in profile.get("allergies") or [])
    tokens.discard("")
    if "nuts" in tokens:
        tokens.add("peanuts")
    return tokens


def meal_allowed(meal: dict, profile: dict) -> bool:
    diet = profile.get("diet") or "none"
    diets = set(meal.get("diets") or [])
    source = meal.get("protein_source")
    if diet == "vegan" and "vegan" not in diets:
        return False
    if diet == "vegetarian" and not ({"vegan", "vegetarian"} & diets):
        return False
    if diet == "pescatarian" and source in MEAT_SOURCES:
        return False
    allergens = {item.lower() for item in meal.get("allergens") or []}
    blocked = blocked_tokens(profile)
    if allergens & blocked:
        return False
    blob = " ".join(meal.get("ingredients") or []).lower()
    for token in blocked:
        if re.search(rf"\b{re.escape(token)}\b", blob):
            return False
    return True


def exercise_count(minutes: int) -> int:
    if minutes <= 30:
        return 3
    if minutes <= 45:
        return 4
    return 5


def validate(profile: dict, plan: dict) -> dict:
    issues: list[str] = []
    workouts = plan.get("workouts") or []
    if len(workouts) != int(profile["days_per_week"]):
        issues.append(f"Expected {profile['days_per_week']} training days.")
    allowed = EQUIPMENT_OK[profile["equipment"]]
    for day in workouts:
        exercises = day.get("exercises") or []
        title = day.get("title") or "A training day"
        if not 3 <= len(exercises) <= 5:
            issues.append(f"{title} should have 3 to 5 exercises.")
        if int(profile["session_minutes"]) <= 30 and len(exercises) > 3:
            issues.append(f"{title} is too long for a 30 minute session.")
        for item in exercises:
            source = exercise_by_id(item.get("id", ""))
            if source is None:
                issues.append(f"Unknown exercise {item.get('name', '')}.")
                continue
            if source["equipment"] not in allowed:
                issues.append(f"{source['name']} needs equipment that is not available.")
    meals = plan.get("meals") or []
    seen = [meal.get("slot") for meal in meals]
    for slot in ("breakfast", "lunch", "dinner", "snack"):
        if slot not in seen:
            issues.append(f"Missing {slot}.")
    daily = 0
    cap = profile.get("dinner_calorie_cap")
    for meal in meals:
        source = meal_by_id(meal.get("id", ""))
        if source is None:
            issues.append(f"Unknown meal {meal.get('name', '')}.")
            continue
        if not meal_allowed(source, profile):
            issues.append(f"{source['name']} does not match the diet or food limits.")
        calories = int(meal.get("calories") or 0)
        daily += calories
        if meal.get("slot") == "dinner" and cap and calories > int(cap):
            issues.append("Dinner is over the requested calorie cap.")
    if meals and (daily < 1200 or daily > 4200):
        issues.append("Daily meal calories are outside a moderate range.")
    return {"ok": not issues, "issues": issues}
