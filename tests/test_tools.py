from __future__ import annotations

import json
import os
import tempfile
import unittest

os.environ["FITFUEL_DB"] = os.path.join(tempfile.gettempdir(), "fitfuel-tools.db")
if os.path.exists(os.environ["FITFUEL_DB"]):
    os.remove(os.environ["FITFUEL_DB"])

from fitfuel.catalog_data import SAMPLE_USERS
from fitfuel.pipeline import run_turn
from fitfuel.profiles import clean_profile
from fitfuel.tools import calculator, duckduckgo_search, knowledge_retriever, wikipedia_lookup
from fitfuel import store


class ToolTests(unittest.TestCase):
    def test_calculator_uses_numexpr(self) -> None:
        raw = calculator.invoke({"expression": "((10*68 + 6.25*165 - 5*24 - 161) * 1.55) * 0.85"})
        self.assertAlmostEqual(float(raw), 1884.35, places=1)

    def test_retriever_returns_approved_exercises(self) -> None:
        hits = json.loads(knowledge_retriever.invoke({"query": "lower body squat beginner", "kind": "exercise"}))
        self.assertTrue(hits)
        self.assertTrue(any(hit["kind"] == "exercise" for hit in hits))
        self.assertTrue(any("squat" in hit["name"].lower() or "lunge" in hit["name"].lower() for hit in hits))

    def test_wikipedia_and_duckduckgo_return_text(self) -> None:
        wiki = wikipedia_lookup("dumbbell squat exercise")
        self.assertNotIn("unavailable", wiki.lower())
        self.assertIn("Summary", wiki)
        search = duckduckgo_search("simple high protein meals for beginners")
        self.assertNotIn("unavailable", search.lower())
        self.assertGreater(len(search), 40)

    def test_plan_calls_every_proposal_tool(self) -> None:
        user_id = store.create_user(clean_profile(SAMPLE_USERS[0]))
        result = run_turn(user_id, "Create my weekly plan", force_intent="generate", use_llm=False, allow_web=True)
        tools = {step["tool"] for step in result["trace"]}
        for name in (
            "get_user_profile",
            "knowledge_retriever",
            "calculator",
            "wikipedia",
            "duckduckgo_search",
            "validate_plan",
            "save_plan",
        ):
            self.assertIn(name, tools, result["trace"])
        self.assertTrue(result["plan"]["validation"]["ok"])
        self.assertIn("Summary", result["plan"]["lookups"]["wikipedia"])
        self.assertNotIn("unavailable", result["plan"]["lookups"]["duckduckgo"].lower())
        self.assertGreater(len(result["plan"]["lookups"]["duckduckgo"]), 40)


if __name__ == "__main__":
    unittest.main()
