from __future__ import annotations

import os
import tempfile

os.environ["FITFUEL_DB"] = os.path.join(tempfile.gettempdir(), "fitfuel-eval.db")
if os.path.exists(os.environ["FITFUEL_DB"]):
    os.remove(os.environ["FITFUEL_DB"])

from fitfuel.catalog import exercise_by_id, meal_by_id
from fitfuel.catalog_data import SAMPLE_USERS
from fitfuel.pipeline import run_turn
from fitfuel.profiles import EQUIPMENT_OK, clean_profile
from fitfuel.rules import exercise_count, meal_allowed
from fitfuel import store


def check(profile: dict, plan: dict) -> list[str]:
    problems = []
    if not plan or not plan.get("validation", {}).get("ok"):
        problems.append("validation failed: " + "; ".join((plan or {}).get("validation", {}).get("issues") or ["no plan"]))
        return problems
    if len(plan["workouts"]) != profile["days_per_week"]:
        problems.append("day count")
    allowed = EQUIPMENT_OK[profile["equipment"]]
    expected = exercise_count(profile["session_minutes"])
    for day in plan["workouts"]:
        if len(day["exercises"]) != expected:
            problems.append(f"{day['title']} has {len(day['exercises'])} exercises")
        for exercise in day["exercises"]:
            source = exercise_by_id(exercise["id"])
            if source is None or source["equipment"] not in allowed:
                problems.append(f"equipment: {exercise['name']}")
    slots = [meal["slot"] for meal in plan["meals"]]
    for slot in ("breakfast", "lunch", "dinner", "snack"):
        if slot not in slots:
            problems.append(f"missing {slot}")
    for meal in plan["meals"]:
        source = meal_by_id(meal["id"])
        if source is None or not meal_allowed(source, profile):
            problems.append(f"meal rule: {meal['name']}")
    cap = profile.get("dinner_calorie_cap")
    if cap:
        dinner = next(meal for meal in plan["meals"] if meal["slot"] == "dinner")
        if dinner["calories"] > cap:
            problems.append("dinner cap")
    totals = plan.get("meal_totals") or {}
    if not 1200 <= totals.get("calories", 0) <= 4200:
        problems.append("meal calories")
    return problems


def main() -> int:
    print("FitFuel sample check")
    print(f"{'User':<10} {'Result':<6} Notes")
    failed = 0
    for raw in SAMPLE_USERS:
        profile = clean_profile(raw)
        user_id = store.create_user(profile)
        run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        if raw.get("follow_up"):
            run_turn(user_id, raw["follow_up"], use_llm=False)
            profile = store.get_profile(user_id)
        plan = store.get_current_plan(user_id)
        problems = check(profile, plan)
        if problems:
            failed += 1
            print(f"{raw['name']:<10} FAIL   {'; '.join(problems)}")
        else:
            dinner = next(meal["calories"] for meal in plan["meals"] if meal["slot"] == "dinner")
            print(
                f"{raw['name']:<10} PASS   {profile['days_per_week']} days, "
                f"{plan['meal_totals']['calories']} meal kcal, dinner {dinner}"
            )
    print(f"{len(SAMPLE_USERS) - failed}/{len(SAMPLE_USERS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
