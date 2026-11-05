from __future__ import annotations

import os
import tempfile
import unittest

os.environ["FITFUEL_DB"] = os.path.join(tempfile.gettempdir(), "fitfuel-unit.db")
if os.path.exists(os.environ["FITFUEL_DB"]):
    os.remove(os.environ["FITFUEL_DB"])

from fitfuel.catalog_data import SAMPLE_USERS
from fitfuel.intent import classify
from fitfuel.pipeline import run_turn
from fitfuel.profiles import EQUIPMENT_OK, clean_profile
from fitfuel.rules import meal_allowed, validate
from fitfuel.catalog import known_name, meal_by_id
from fitfuel import store


class IntentTests(unittest.TestCase):
    def test_proposal_phrases(self) -> None:
        dinners = classify("Make dinners under 500 calories", True)
        self.assertEqual(dinners["intent"], "update_meals")
        self.assertEqual(dinners["dinner_calorie_cap"], 500)

        minutes = classify("What if I only have 30 minutes?", True)
        self.assertEqual(minutes["intent"], "update_workout")
        self.assertEqual(minutes["session_minutes"], 30)

        diet = classify("Make my meals vegetarian.", True)
        self.assertEqual(diet["intent"], "update_meals")
        self.assertEqual(diet["diet"], "vegetarian")

        swap = classify("Can I swap this exercise?", True)
        self.assertEqual(swap["intent"], "update_workout")
        self.assertTrue(swap["swap_requested"])
        self.assertIsNone(swap["swap_from"])

        question = classify("What is a Romanian deadlift?", True)
        self.assertEqual(question["intent"], "question")

        injury = classify("I have a knee injury and need a rehab plan", True)
        self.assertEqual(injury["intent"], "medical")

        chest = classify("Can I do more chest exercises?", True)
        self.assertNotEqual(chest["intent"], "medical")
        chest_pain = classify("I have chest pain when I train", True)
        self.assertEqual(chest_pain["intent"], "medical")

        dumbbells = classify("Can you make my workouts dumbells only?", True)
        self.assertEqual(dumbbells["intent"], "update_workout")
        self.assertEqual(dumbbells["equipment"], "dumbbells")

        added = classify("Add chicken haleem for dinner", True)
        self.assertEqual(added["intent"], "add")
        self.assertEqual(added["add_name"], "chicken haleem")
        self.assertEqual(added["add_slot"], "dinner")

        extreme = classify("Put me on an 800 calorie diet", True)
        self.assertEqual(extreme["intent"], "medical")
        self.assertEqual(extreme["medical_kind"], "extreme")


class PlanTests(unittest.TestCase):
    def test_proposal_example_and_follow_up(self) -> None:
        raw = next(item for item in SAMPLE_USERS if item["name"] == "Ayesha")
        user_id = store.create_user(clean_profile(raw))
        first = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        self.assertEqual(first["intent"], "generate")
        self.assertTrue(first["plan"]["validation"]["ok"], first["plan"]["validation"])
        self.assertEqual(len(first["plan"]["workouts"]), 4)
        tools = {step["tool"] for step in first["trace"]}
        self.assertIn("knowledge_retriever", tools)
        self.assertIn("calculator", tools)
        self.assertIn("validate_plan", tools)
        self.assertIn("save_plan", tools)
        for meal in first["plan"]["meals"]:
            self.assertNotIn("dairy", meal_by_id(meal["id"])["allergens"])

        second = run_turn(user_id, "Make dinners under 500 calories", use_llm=False)
        self.assertEqual(second["intent"], "update_meals")
        dinner = next(meal for meal in second["plan"]["meals"] if meal["slot"] == "dinner")
        self.assertLessEqual(dinner["calories"], 500)
        self.assertIn("update_plan", {step["tool"] for step in second["trace"]})
        self.assertTrue(second["plan"]["validation"]["ok"], second["plan"]["validation"])

    def test_swap_and_short_session(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[0]))
        run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        before = [
            exercise["name"]
            for day in store.get_current_plan(user_id)["workouts"]
            for exercise in day["exercises"]
        ]
        self.assertTrue(any("squat" in name.lower() for name in before))
        swapped = run_turn(user_id, "Can I swap squats?", use_llm=False)
        self.assertEqual(swapped["intent"], "update_workout")
        self.assertIn("Swapped", swapped["answer"])
        after = [
            exercise["name"]
            for day in swapped["plan"]["workouts"]
            for exercise in day["exercises"]
        ]
        self.assertNotEqual(before, after)
        short = run_turn(user_id, "What if I only have 30 minutes?", use_llm=False)
        for day in short["plan"]["workouts"]:
            self.assertEqual(len(day["exercises"]), 3)
            self.assertEqual(day["minutes"], 30)

    def test_medical_does_not_replace_the_plan(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[1]))
        first = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        blocked = run_turn(user_id, "My knee injury hurts. What painkiller should I take?", use_llm=False)
        self.assertEqual(blocked["intent"], "medical")
        self.assertEqual(blocked["plan"]["id"], first["plan"]["id"])
        self.assertIn("professional", blocked["answer"].lower())

    def test_vegetarian_follow_up_removes_meat(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[-1]))
        run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        updated = run_turn(user_id, "Make my meals vegetarian.", use_llm=False)
        profile = store.get_profile(user_id)
        self.assertEqual(profile["diet"], "vegetarian")
        for meal in updated["plan"]["meals"]:
            source = meal_by_id(meal["id"])
            self.assertTrue(meal_allowed(source, profile), source["name"])

    def test_validator_rejects_a_dairy_meal_for_a_dairy_free_user(self) -> None:
        profile = clean_profile(SAMPLE_USERS[0])
        yogurt = meal_by_id("yogurt_berries")
        plan = {
            "workouts": [],
            "meals": [{"id": yogurt["id"], "slot": "breakfast", "name": yogurt["name"], "calories": 300}],
        }
        result = validate(profile, plan)
        self.assertFalse(result["ok"])
        self.assertFalse(meal_allowed(yogurt, profile))

    def test_bodyweight_plan_has_no_machines(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[2]))
        result = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        allowed = EQUIPMENT_OK["bodyweight"]
        for day in result["plan"]["workouts"]:
            self.assertEqual(len(day["exercises"]), 3)
            for exercise in day["exercises"]:
                self.assertIn(exercise["equipment"], allowed)

    def test_pakistani_meals_follow_the_cuisine(self) -> None:
        raw = dict(SAMPLE_USERS[3])
        raw["cuisine"] = "pakistani"
        user_id = store.create_user(clean_profile(raw))
        result = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        self.assertTrue(result["plan"]["validation"]["ok"], result["plan"]["validation"])
        for meal in result["plan"]["meals"]:
            source = meal_by_id(meal["id"])
            self.assertIn("pakistani", source.get("cuisines") or [], source["name"])

    def test_each_cuisine_fills_a_restricted_day(self) -> None:
        raw = next(item for item in SAMPLE_USERS if item["name"] == "Dina")
        for cuisine in ("pakistani", "chinese", "indian", "mediterranean", "mexican", "italian"):
            person = dict(raw)
            person["cuisine"] = cuisine
            user_id = store.create_user(clean_profile(person))
            result = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
            self.assertTrue(result["plan"]["validation"]["ok"], result["plan"]["validation"])
            for meal in result["plan"]["meals"]:
                source = meal_by_id(meal["id"])
                self.assertIn(cuisine, source.get("cuisines") or [], f"{cuisine}: {source['name']}")

    def test_a_new_meal_is_checked_then_added(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[0]))
        run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        added = run_turn(user_id, "Add chicken haleem for dinner", use_llm=False)
        self.assertEqual(added["intent"], "add")
        dinner = next(meal for meal in added["plan"]["meals"] if meal["slot"] == "dinner")
        self.assertIn("haleem", dinner["name"].lower())
        self.assertTrue(added["plan"]["validation"]["ok"], added["plan"]["validation"])
        self.assertIn("added", added["answer"].lower())
        self.assertIn("validate_plan", {step["tool"] for step in added["trace"]})
        self.assertIsNotNone(meal_by_id(dinner["id"]))
        store.delete_custom(dinner["id"])

    def test_a_blocked_food_is_refused(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[0]))
        first = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        blocked = run_turn(user_id, "Add yogurt for breakfast", use_llm=False)
        self.assertEqual(blocked["intent"], "add")
        self.assertEqual(blocked["plan"]["id"], first["plan"]["id"])
        self.assertIn("did not", blocked["answer"].lower())
        names = " ".join(meal["name"].lower() for meal in blocked["plan"]["meals"])
        self.assertNotIn("yogurt", names)

    def test_groq_reads_the_profile_and_the_week(self) -> None:
        import json
        from unittest.mock import patch

        user_id = store.create_user(clean_profile(SAMPLE_USERS[0]))
        run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        plan = store.get_current_plan(user_id)
        meal = plan["meals"][0]["name"]
        prompts: list[str] = []

        def fake(prompt: str) -> str:
            prompts.append(prompt)
            if "Check passed" in prompt:
                return "I switched the training to dumbbells. The check passed, so it is on the week."
            return json.dumps(
                {
                    "say": f"Your week already has {meal}.",
                    "intent": "question",
                }
            )

        with patch("fitfuel.llm.complete", fake):
            asked = run_turn(user_id, "What is on my week?", use_llm=True)
        self.assertEqual(asked["intent"], "question")
        self.assertIn("Ayesha", prompts[0])
        self.assertIn(meal, prompts[0])
        self.assertIn(meal, asked["answer"])
        self.assertNotIn("wikipedia", asked["answer"].lower())
        self.assertEqual(asked["plan"]["id"], plan["id"])

        prompts.clear()

        def fake_change(prompt: str) -> str:
            prompts.append(prompt)
            if "Check passed" in prompt:
                return "I switched the training to dumbbells. The check passed, so it is on the week."
            return json.dumps(
                {
                    "say": "I can switch that.",
                    "intent": "update_workout",
                    "equipment": "dumbbells",
                }
            )

        with patch("fitfuel.llm.complete", fake_change):
            changed = run_turn(user_id, "Please switch my training", use_llm=True)
        self.assertEqual(changed["intent"], "update_workout")
        self.assertIn("Ayesha", prompts[0])
        self.assertIn(meal, prompts[0])
        self.assertIn("check passed", changed["answer"].lower())
        self.assertTrue(changed["plan"]["validation"]["ok"], changed["plan"]["validation"])
        allowed = EQUIPMENT_OK["dumbbells"]
        for day in changed["plan"]["workouts"]:
            for exercise in day["exercises"]:
                self.assertIn(exercise["equipment"], allowed)

    def test_a_new_week_is_different(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[0]))
        first = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        second = run_turn(user_id, "Build another week for me", force_intent="generate", use_llm=False)
        self.assertTrue(second["plan"]["validation"]["ok"], second["plan"]["validation"])
        first_names = [exercise["name"] for day in first["plan"]["workouts"] for exercise in day["exercises"]]
        second_names = [exercise["name"] for day in second["plan"]["workouts"] for exercise in day["exercises"]]
        first_meals = [meal["name"] for meal in first["plan"]["meals"]]
        second_meals = [meal["name"] for meal in second["plan"]["meals"]]
        self.assertNotEqual(first_names, second_names)
        self.assertNotEqual(first_meals, second_meals)
        self.assertNotEqual(first["plan"]["id"], second["plan"]["id"])

    def test_a_short_week_also_changes(self) -> None:
        raw = next(item for item in SAMPLE_USERS if item["name"] == "Chen")
        user_id = store.create_user(clean_profile(raw))
        first = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False)
        second = run_turn(user_id, "Build another week for me", force_intent="generate", use_llm=False)
        self.assertTrue(second["plan"]["validation"]["ok"], second["plan"]["validation"])
        first_names = [exercise["name"] for day in first["plan"]["workouts"] for exercise in day["exercises"]]
        second_names = [exercise["name"] for day in second["plan"]["workouts"] for exercise in day["exercises"]]
        self.assertNotEqual(first_names, second_names)
        self.assertFalse(known_name("chicken haleem"))
        self.assertTrue(known_name("bench press"))


if __name__ == "__main__":
    unittest.main()
