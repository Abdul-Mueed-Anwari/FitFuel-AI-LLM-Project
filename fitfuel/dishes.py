from __future__ import annotations

import re

from fitfuel.intent import REFERENCE_WORDS

PARTS = {
    "oatmeal": ["oats", "milk"],
    "porridge": ["oats", "milk"],
    "oat": ["oats"],
    "oats": ["oats"],
    "yogurt": ["yogurt", "dairy"],
    "yoghurt": ["yogurt", "dairy"],
    "omelette": ["eggs"],
    "omelet": ["eggs"],
    "pancake": ["flour", "eggs", "milk"],
    "pancakes": ["flour", "eggs", "milk"],
    "waffle": ["flour", "eggs", "milk"],
    "waffles": ["flour", "eggs", "milk"],
    "pizza": ["flour", "cheese", "dairy"],
    "pasta": ["pasta", "gluten"],
    "noodle": ["noodles", "gluten"],
    "noodles": ["noodles", "gluten"],
    "sandwich": ["bread", "gluten"],
    "burger": ["beef", "bread", "gluten"],
    "latte": ["milk", "dairy"],
    "cappuccino": ["milk", "dairy"],
    "cheese": ["cheese", "dairy"],
    "paneer": ["paneer", "dairy"],
    "butter": ["butter", "dairy"],
    "cream": ["cream", "dairy"],
    "bread": ["bread", "gluten"],
    "roti": ["roti", "gluten"],
    "naan": ["naan", "gluten"],
    "tofu": ["tofu", "soy"],
}
GROUPS = (
    {"milk", "dairy", "lactose", "yogurt", "yoghurt", "cheese", "paneer", "butter", "cream", "ghee", "whey"},
    {"egg", "eggs"},
    {"peanut", "peanuts"},
    {"nut", "nuts", "almond", "almonds", "cashew", "cashews", "walnut", "walnuts"},
    {"gluten", "wheat", "bread", "flour", "pasta", "roti", "naan"},
    {"shrimp", "prawn", "prawns", "shellfish"},
    {"soy", "tofu", "edamame"},
    {"fish", "salmon", "tuna", "cod"},
)
SKIP = {
    "a",
    "an",
    "the",
    "and",
    "with",
    "for",
    "of",
    "my",
    "some",
    "style",
    "bowl",
    "plate",
    "homemade",
    "fried",
    "grilled",
    "baked",
    "fresh",
    "hot",
    "cold",
    "please",
    "soup",
    "smoothie",
    "salad",
    "curry",
    "stew",
    "without",
    "leave",
    "leaving",
    "skip",
    "except",
    "minus",
    "remove",
    "drop",
    "hold",
}
DISH_RE = re.compile(
    r"\b(?:eat|drink|have|try|add|include|about|want)\s+(?:a |an |some |the )?(.+)",
    re.I,
)
WITHOUT_RE = re.compile(
    r"\b(?:without|leave out|leaving out|skip|except|minus|hold the|remove|drop)\b\s*(.*)$",
    re.I,
)
OMIT_FILLER = REFERENCE_WORDS | {"ingredient", "ingredients", "thing", "things", "stuff", "part", "parts", "the", "a", "an"}
SLOT_WORDS = {
    "breakfast": ("breakfast", "oatmeal", "oats", "omelette", "omelet", "pancake", "cereal", "yogurt"),
    "snack": ("snack", "smoothie", "shake"),
    "lunch": ("lunch",),
    "dinner": ("dinner", "haleem", "curry", "stew"),
}


def _stem(word: str) -> str:
    raw = re.sub(r"[^a-z]", "", word.lower())
    if len(raw) > 4 and raw.endswith("ies"):
        return raw[:-3] + "y"
    if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
        return raw[:-1]
    return raw


def forms(word: str) -> set[str]:
    base = _stem(word)
    if not base:
        return set()
    found = {base, re.sub(r"[^a-z]", "", word.lower())}
    for group in GROUPS:
        if found & group:
            found |= group
    return {item for item in found if item}


def same_food(left: str, right: str) -> bool:
    return bool(forms(left) & forms(right))


def is_reference(text: str) -> bool:
    words = re.findall(r"[a-z']+", (text or "").lower())
    return bool(words) and all(word in REFERENCE_WORDS for word in words)


def wants_without(message: str) -> bool:
    return bool(WITHOUT_RE.search(message or ""))


def _clean_phrase(raw: str) -> str:
    name = raw.strip(" ?.,!")
    name = re.sub(
        r"\b(?:for|as)\s+(?:a |an |my )?(?:breakfast|lunch|dinner|snack|exercise|meal)\b.*$",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"\s+", " ", name).strip(" ?.,!")
    return name


def dish_in_text(text: str) -> str | None:
    match = DISH_RE.search(text or "")
    if not match:
        return None
    name = _clean_phrase(match.group(1))
    name = WITHOUT_RE.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" ?.,!")
    if len(name) < 3 or is_reference(name):
        return None
    return name


def dish_from_history(history: str) -> str | None:
    lines = [line for line in (history or "").splitlines() if line.strip() and line.strip() != "None"]
    for line in reversed(lines):
        body = re.sub(r"^(user|assistant):\s*", "", line, flags=re.I)
        found = dish_in_text(body)
        if found:
            return found
    return None


def resolve_dish(message: str, history: str, hinted: str | None) -> str | None:
    if hinted and not is_reference(hinted):
        return hinted.strip()
    current = dish_in_text(message)
    if current:
        return current
    return dish_from_history(history)


def ingredient_words(name: str) -> list[str]:
    found: list[str] = []
    for word in re.findall(r"[a-z']+", name.lower()):
        if word in SKIP or len(word) < 3:
            continue
        pieces = PARTS.get(word, [word])
        for piece in pieces:
            if piece not in found:
                found.append(piece)
    return found or ["vegetables"]


def _limits(profile: dict) -> list[tuple[str, str]]:
    rows = []
    for source in ("allergies", "avoid"):
        for item in profile.get(source) or []:
            token = str(item).strip().lower()
            if token:
                rows.append((token, token))
    return rows


def conflicts(ingredients: list[str], profile: dict) -> list[tuple[str, str]]:
    notes = (profile.get("notes") or "").lower()
    hits: list[tuple[str, str]] = []
    seen = set()
    for ingredient in ingredients:
        for token, label in _limits(profile):
            if same_food(ingredient, token) and (ingredient, label) not in seen:
                hits.append((ingredient, label))
                seen.add((ingredient, label))
        if notes:
            note_forms = set()
            for word in re.findall(r"[a-z']+", notes):
                note_forms |= forms(word)
            if forms(ingredient) & note_forms and (ingredient, "your notes") not in seen:
                hits.append((ingredient, "your notes"))
                seen.add((ingredient, "your notes"))
    return hits


def omit_from(message: str) -> tuple[str, list[str]]:
    match = WITHOUT_RE.search(message or "")
    if not match:
        return "none", []
    words = [word for word in re.findall(r"[a-z']+", match.group(1).lower()) if word not in OMIT_FILLER and word not in SKIP]
    if not words:
        return "all", []
    return "some", words


def apply_omit(ingredients: list[str], message: str, profile: dict) -> tuple[list[str], list[str]]:
    mode, words = omit_from(message)
    if mode == "none":
        return ingredients, []
    if mode == "all":
        bad = [item for item in ingredients if conflicts([item], profile)]
        kept = [item for item in ingredients if item not in bad]
        return kept, bad
    omitted = []
    kept = []
    for item in ingredients:
        if any(same_food(item, word) for word in words):
            omitted.append(item)
        else:
            kept.append(item)
    return kept, omitted


def guess_slot(name: str, message: str) -> str:
    blob = f"{name} {message}".lower()
    for slot, words in SLOT_WORDS.items():
        if any(word in blob for word in words):
            return slot
    return "dinner"


def cleaned_name(dish: str, omitted: list[str]) -> str:
    if not omitted:
        return " ".join(part.capitalize() for part in dish.split())
    kept = [
        word
        for word in dish.split()
        if word.lower() not in SKIP and not any(same_food(word, item) for item in omitted)
    ]
    title = " ".join(part.capitalize() for part in kept) if kept else "Plate"
    return title[:60]


def refusal(dish: str, ingredients: list[str], hits: list[tuple[str, str]]) -> str:
    foods = ", ".join(dict.fromkeys(item for item, _label in hits))
    reasons = ", ".join(dict.fromkeys(label for _item, label in hits))
    return (
        f"{dish.capitalize()} includes {foods}, and that matches {reasons}, so I did not add it. "
        "Tell me if you will make it without that ingredient."
    )


def question_block(message: str, history: str, profile: dict) -> str | None:
    dish = resolve_dish(message, history, None)
    if not dish:
        return None
    kept, _omitted = apply_omit(ingredient_words(dish), message, profile)
    hits = conflicts(kept, profile)
    if not hits:
        return None
    return refusal(dish, kept, hits)
