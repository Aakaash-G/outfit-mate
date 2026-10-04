"""Outfit assembly and scoring.

score = 0.45 * compatibility + 0.25 * match-to-request + 0.20 * personal style - 0.10 * recently-worn
"""
from datetime import date
from itertools import combinations

import numpy as np

from .config import W_COMPAT, W_QUERY, W_RECENCY, W_STYLE
from .vision import FORMALITY_WORDS, NEUTRALS

SLOTS = ("top", "bottom", "onepiece", "shoes", "layer")


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def query_text(intent: dict) -> str:
    return f"a photo of {FORMALITY_WORDS[intent['formality']]} clothing for {intent['occasion']}"


# ---------------------------------------------------------------- compatibility
def compatibility(items: list[dict]) -> float:
    """Rule-based + embedding-cohesion compatibility in [0, 1].

    Swap this for a trained model (e.g. a Transformer on Polyvore Outfits) later;
    keep the same signature: list of garments -> score in [0, 1].
    """
    s = 0.8
    clothes = [g for g in items if g["slot"] != "shoes"]
    patterned = sum(1 for g in clothes if g["pattern"] != "solid")
    if patterned >= 2:
        s -= 0.35  # two busy patterns clash
    bold = {c for g in items for c in g["colors"] if c not in NEUTRALS}
    if len(bold) >= 3:
        s -= 0.25  # too many strong colours
    elif not bold:
        s += 0.05  # all-neutral outfits are safe
    tops = [g for g in clothes if g["slot"] == "top"]
    bottoms = [g for g in clothes if g["slot"] == "bottom"]
    if tops and bottoms and tops[0]["colors"][0] == bottoms[0]["colors"][0] and tops[0]["colors"][0] not in NEUTRALS:
        s -= 0.1  # same bold colour head to toe
    f = [g["formality"] for g in items]
    s -= 0.1 * max(0, max(f) - min(f) - 1)  # mixing gym wear with formal wear
    embs = [g["embedding"] for g in items if g.get("embedding") is not None]
    if len(embs) > 1:
        coh = np.mean([float(a @ b) for a, b in combinations(embs, 2)])
        s += 0.2 * (_clip01((coh - 0.5) / 0.4) - 0.5)
    return _clip01(s)


# ---------------------------------------------------------------- filtering
def filter_garments(garments, intent, weather, tol=1):
    avoid = set(intent.get("avoid_colors", []))
    temp = weather["temp_c"]
    out = []
    for g in garments:
        if avoid & set(g["colors"]):
            continue
        tol_g = tol + 1 if g["slot"] == "shoes" else tol
        if abs(g["formality"] - intent["formality"]) > tol_g:
            continue
        if g["warmth"] >= 3 and temp >= 24:
            continue  # no hoodies / sweaters in the heat
        if g["category"] == "sandals" and weather["rain_prob"] > 0.7 and intent["formality"] >= 3:
            continue
        out.append(g)
    return out


def _recency_penalty(items, worn: dict, day: date) -> float:
    pens = []
    for g in items:
        d = worn.get(g["id"])
        if d is None:
            pens.append(0.0)
            continue
        gap = abs((day - d).days)
        pens.append(1 - gap / 4 if gap <= 3 else 0.0)
    return float(np.mean(pens))


# ---------------------------------------------------------------- ranking
def rank_outfits(intent, weather, garments, clip, style_vec=None, worn=None, day=None,
                 top_n=10, per_slot=6, avoid_ids=None) -> list[dict]:
    """avoid_ids: garments to leave out when the same slot still has alternatives (used by the planner)."""
    worn = worn or {}
    if avoid_ids:
        kept = []
        for g in garments:
            if g["id"] in avoid_ids:
                alternatives = [x for x in garments if x["slot"] == g["slot"] and x["id"] not in avoid_ids
                                and abs(x["formality"] - intent["formality"]) <= 1]
                if alternatives:
                    continue
            kept.append(g)
        garments = kept
    day = day or date.today()
    q = clip.text([query_text(intent)])[0]
    prefer = set(intent.get("prefer_colors", []))

    def qsim(g):
        s = _clip01((float(g["embedding"] @ q) - 0.15) / 0.2)
        return s + (0.1 if prefer & set(g["colors"]) else 0.0)

    by = {}
    for tol in (1, 2):  # relax formality if nothing fits
        pool = filter_garments(garments, intent, weather, tol)
        by = {s: sorted([g for g in pool if g["slot"] == s], key=qsim, reverse=True)[:per_slot] for s in SLOTS}
        bases = [(t, b) for t in by["top"] for b in by["bottom"]] + [(o,) for o in by["onepiece"]]
        if bases:
            break
    if not bases:
        return []

    shoes = by["shoes"] or [None]
    use_layer = weather["temp_c"] < 22 or intent["formality"] >= 4
    layers = [None] + (by["layer"] if use_layer else [])

    results = []
    for base in bases:
        for sh in shoes:
            for ly in layers:
                items = [g for g in (*base, ly, sh) if g is not None]
                comp = compatibility(items)
                qs = _clip01(np.mean([qsim(g) for g in items]))
                if style_vec is not None:
                    ss = _clip01((np.mean([float(g["embedding"] @ style_vec) for g in items]) - 0.5) / 0.4)
                else:
                    ss = 0.5
                rp = _recency_penalty(items, worn, day)
                score = W_COMPAT * comp + W_QUERY * qs + W_STYLE * ss - W_RECENCY * rp
                results.append({"items": items, "score": round(score, 4),
                                "parts": {"compat": round(comp, 3), "query": round(qs, 3),
                                          "style": round(ss, 3), "recency_penalty": round(rp, 3)}})
    results.sort(key=lambda r: r["score"], reverse=True)

    # diversity: don't return ten outfits built around the same top
    picked, seen = [], {}
    for r in results:
        key = r["items"][0]["id"]
        if seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        picked.append(r)
        if len(picked) == top_n:
            break
    return picked