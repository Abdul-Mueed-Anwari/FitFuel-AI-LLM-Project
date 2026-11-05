from __future__ import annotations

import re

from fitfuel.catalog_data import EXERCISES, MEALS, NOTES

_retriever = None
_retriever_ready = False


def meal_calories(meal: dict) -> int:
    return int(round(meal["protein_g"] * 4 + meal["carbs_g"] * 4 + meal["fat_g"] * 9))


def exercises() -> list[dict]:
    return list(EXERCISES)


def meals() -> list[dict]:
    return list(MEALS)


def _customs() -> list[dict]:
    try:
        from fitfuel import store

        return store.custom_items()
    except Exception:
        return []


def exercise_by_id(item_id: str) -> dict | None:
    found = next((item for item in EXERCISES if item["id"] == item_id), None)
    if found:
        return found
    return next((item for item in _customs() if item.get("kind") == "exercise" and item["id"] == item_id), None)


def meal_by_id(item_id: str) -> dict | None:
    found = next((item for item in MEALS if item["id"] == item_id), None)
    if found:
        return found
    return next((item for item in _customs() if item.get("kind") == "meal" and item["id"] == item_id), None)


def _name_words(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    skip = {"a", "an", "the", "for", "with", "and", "my", "please"}
    return [word for word in cleaned.split() if word not in skip and len(word) > 1]


def known_name(name: str) -> bool:
    asked = _name_words(name)
    if len(" ".join(asked)) < 3:
        return False
    options = [asked]
    if asked and asked[-1].endswith("s") and len(asked[-1]) > 3:
        options.append(asked[:-1] + [asked[-1][:-1]])
    for item in list(EXERCISES) + list(MEALS) + _customs():
        label = _name_words(item["name"])
        if not label:
            continue
        for option in options:
            if option == label:
                return True
            if len(option) >= 2 and all(word in label for word in option):
                return True
            if len(label) >= 2 and all(word in option for word in label):
                return True
    return False


def _documents():
    from langchain_core.documents import Document

    docs = []
    for item in EXERCISES:
        text = (
            f"id={item['id']} kind=exercise name={item['name']}. "
            f"Equipment: {item['equipment']}. Focus: {', '.join(item['focus'])}. "
            f"Beginner cue: {item['cue']}"
        )
        docs.append(Document(page_content=text, metadata={"id": item["id"], "kind": "exercise", "name": item["name"]}))
    for item in MEALS:
        text = (
            f"id={item['id']} kind=meal name={item['name']}. "
            f"Slots: {', '.join(item['slots'])}. "
            f"Diets: {', '.join(item['diets']) or 'omnivore'}. "
            f"Protein source: {item['protein_source']}. "
            f"Allergens: {', '.join(item['allergens']) or 'none'}. "
            f"Ingredients: {', '.join(item['ingredients'])}."
        )
        if item.get("cuisines"):
            text += f" Cuisine: {', '.join(item['cuisines'])}."
        docs.append(Document(page_content=text, metadata={"id": item["id"], "kind": "meal", "name": item["name"]}))
    for item in NOTES:
        text = f"id={item['id']} kind=note name={item['title']}. {item['text']}"
        docs.append(Document(page_content=text, metadata={"id": item["id"], "kind": "note", "name": item["title"]}))
    return docs


def get_bm25():
    return _bm25()


def _bm25():
    global _retriever, _retriever_ready
    if _retriever_ready:
        return _retriever
    _retriever_ready = True
    try:
        from langchain_community.retrievers.bm25 import BM25Retriever

        retriever = BM25Retriever.from_documents(_documents())
        retriever.k = 12
        _retriever = retriever
    except Exception:
        _retriever = None
    return _retriever


def _kind_items(kind: str) -> list[dict]:
    rows = []
    if kind in {"exercise", "any"}:
        for item in EXERCISES:
            rows.append({"id": item["id"], "kind": "exercise", "name": item["name"], "text": f"{item['name']}. {item['cue']}"})
    if kind in {"meal", "any"}:
        for item in MEALS:
            rows.append(
                {
                    "id": item["id"],
                    "kind": "meal",
                    "name": item["name"],
                    "text": f"{item['name']}. Ingredients: {', '.join(item['ingredients'])}.",
                }
            )
    if kind in {"note", "any"}:
        for item in NOTES:
            rows.append({"id": item["id"], "kind": "note", "name": item["title"], "text": item["text"]})
    return rows


def search_knowledge(query: str, kind: str = "any", k: int = 6) -> list[dict]:
    if kind not in {"exercise", "meal", "note", "any"}:
        kind = "any"
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for index, item in enumerate(_kind_items(kind)):
        words = set(re.findall(r"[a-z0-9]+", f"{item['name']} {item['text']}".lower()))
        ranked.append((len(tokens & words), index, item))
    retriever = _bm25()
    bonus = {}
    if retriever is not None:
        try:
            position = 0
            for doc in retriever.invoke(query):
                if kind != "any" and doc.metadata.get("kind") != kind:
                    continue
                bonus[doc.metadata["id"]] = max(0, 8 - position)
                position += 1
        except Exception:
            bonus = {}
    ranked.sort(key=lambda row: (-(row[0] * 10 + bonus.get(row[2]["id"], 0)), row[1]))
    return [row[2] for row in ranked[:k]]
