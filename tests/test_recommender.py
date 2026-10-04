"""Unit tests for outfit logic. They use a fake CLIP, so no model download is needed.

Run:  python -m pytest -q
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm, recommender  # noqa: E402

DIM = 16
rng = np.random.default_rng(1)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeClip:
    def text(self, texts):
        return np.stack([unit(np.ones(DIM)) for _ in texts])


def g(i, category, slot, colors, pattern="solid", formality=3, warmth=1):
    return {"id": i, "name": category, "category": category, "slot": slot, "colors": colors,
            "pattern": pattern, "formality": formality, "warmth": warmth,
            "embedding": unit(np.ones(DIM) + 0.3 * rng.standard_normal(DIM)), "image_path": ""}


WARDROBE = [
    g(1, "shirt", "top", ["white"], formality=4),
    g(2, "shirt", "top", ["light blue"], "striped", formality=4),
    g(3, "t-shirt", "top", ["red"], "printed", formality=2),
    g(4, "trousers", "bottom", ["navy"], formality=4),
    g(5, "jeans", "bottom", ["denim"], formality=2),
    g(6, "shorts", "bottom", ["olive"], "checked", formality=1),
    g(7, "formal shoes", "shoes", ["brown"], formality=4),
    g(8, "sneakers", "shoes", ["white"], formality=2),
    g(9, "hoodie", "layer", ["grey"], formality=1, warmth=3),
    g(10, "blazer", "layer", ["navy"], formality=4, warmth=2),
]
HOT = {"temp_c": 31, "rain_prob": 0.1, "description": "sunny"}


def intent(f, occasion="test"):
    return {"occasion": occasion, "formality": f, "prefer_colors": [], "avoid_colors": [], "style_notes": ""}


def ids(outfit):
    return {x["id"] for x in outfit["items"]}


def test_formal_request_gets_formal_items():
    res = recommender.rank_outfits(intent(4, "interview"), HOT, WARDROBE, FakeClip())
    assert res
    best = ids(res[0])
    assert best & {1, 2} and 4 in best and 7 in best
    assert 6 not in best and 9 not in best


def test_no_hoodie_in_heat():
    res = recommender.rank_outfits(intent(1), HOT, WARDROBE, FakeClip())
    assert all(9 not in ids(r) for r in res)


def test_avoid_colour():
    it = intent(4)
    it["avoid_colors"] = ["white"]
    res = recommender.rank_outfits(it, HOT, WARDROBE, FakeClip())
    assert all(1 not in ids(r) and 8 not in ids(r) for r in res)


def test_two_patterns_penalised():
    clash = recommender.compatibility([WARDROBE[1], WARDROBE[5]])  # striped + checked
    calm = recommender.compatibility([WARDROBE[0], WARDROBE[3]])   # white + navy solid
    assert calm > clash


def test_recently_worn_ranked_lower():
    today = date.today()
    fresh = recommender.rank_outfits(intent(4), HOT, WARDROBE, FakeClip(), day=today)
    worn = {i: today - timedelta(days=1) for i in ids(fresh[0])}
    again = recommender.rank_outfits(intent(4), HOT, WARDROBE, FakeClip(), worn=worn, day=today)
    assert ids(again[0]) != ids(fresh[0])


def test_rule_intent_parser():
    it = llm._rule_intent("Placement interview tomorrow, no black please")
    assert it["formality"] == 4 and "black" in it["avoid_colors"]
    assert llm._rule_intent("cousin's wedding")["formality"] == 5
    assert llm._rule_intent("going to the gym")["formality"] == 1


def test_choose_without_llm_returns_k_valid_outfits():
    cands = recommender.rank_outfits(intent(4), HOT, WARDROBE, FakeClip())
    chosen = llm.choose_and_explain(cands, intent(4, "interview"), HOT, k=3, use_llm=False)
    assert len(chosen) == min(3, len(cands))
    assert all(c["title"] and c["reason"] for c in chosen)
    assert len({c["_oid"] for c in chosen}) == len(chosen)