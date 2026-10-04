"""Stylist agent: orchestrates weather -> intent -> retrieval/scoring -> LLM choice -> storage."""
import time
from datetime import date, timedelta

import numpy as np

from . import db, llm, recommender, weather


def recommend(prompt: str, city: str, day: date, clip, k: int = 3, extra_worn: dict | None = None,
              save: bool = True, intent: dict | None = None, use_llm: bool = True,
              avoid_ids: set | None = None) -> dict:
    timings = {}
    garments = db.list_garments()
    if not garments:
        raise ValueError("Your wardrobe is empty. Add some clothes on the Wardrobe tab first.")

    t = time.perf_counter()
    wx = weather.get_weather(city, day)
    timings["weather"] = time.perf_counter() - t

    t = time.perf_counter()
    intent = intent or llm.parse_intent(prompt, wx)
    timings["intent"] = time.perf_counter() - t

    worn = db.last_worn()
    for gid, d in (extra_worn or {}).items():
        worn[gid] = d

    t = time.perf_counter()
    cands = recommender.rank_outfits(intent, wx, garments, clip, db.get_style_vector(), worn, day,
                                     avoid_ids=avoid_ids)
    timings["retrieve_and_score"] = time.perf_counter() - t

    t = time.perf_counter()
    outfits = llm.choose_and_explain(cands, intent, wx, k=k, use_llm=use_llm) if cands else []
    timings["explain"] = time.perf_counter() - t

    if save:
        for o in outfits:
            o["outfit_id"] = db.save_outfit([g["id"] for g in o["items"]], intent["occasion"],
                                            o["score"], o["title"], o["reason"])
    return {"intent": intent, "weather": wx, "outfits": outfits, "timings": timings,
            "n_candidates": len(cands), "n_garments": len(garments)}


def feedback(outfit: dict, action: str, day: date | None = None):
    """Record like / skip / worn and nudge the user's style vector (report 6.2, FR-7)."""
    if outfit.get("outfit_id") is None:
        outfit["outfit_id"] = db.save_outfit([g["id"] for g in outfit["items"]], "manual",
                                             outfit.get("score", 0), outfit.get("title", ""), outfit.get("reason", ""))
    db.add_feedback(outfit["outfit_id"], action, day)
    mean = np.mean([g["embedding"] for g in outfit["items"]], axis=0)
    mean = mean / (np.linalg.norm(mean) or 1)
    style = db.get_style_vector()
    if action in ("like", "worn"):
        new = mean if style is None else 0.85 * style + 0.15 * mean
    elif action == "skip" and style is not None:
        new = style - 0.05 * mean
    else:
        return
    db.set_style_vector(new / (np.linalg.norm(new) or 1))


def plan_week(prompt: str, city: str, start: date, clip, days: int = 7) -> list[dict]:
    """One outfit per day, avoiding repeats by treating planned days as 'worn'."""
    wx0 = weather.get_weather(city, start)
    intent = llm.parse_intent(prompt, wx0)
    planned_worn, plan = {}, []
    shoe_ids = {g["id"] for g in db.list_garments() if g["slot"] == "shoes"}
    for i in range(days):
        day = start + timedelta(days=i)
        # don't repeat tops / bottoms / dresses worn in the previous 2 days if there is an alternative
        recent = {gid for gid, d in planned_worn.items() if 0 < (day - d).days <= 2 and gid not in shoe_ids}
        res = recommend(prompt, city, day, clip, k=1, extra_worn=planned_worn, save=False,
                        intent=intent, use_llm=False, avoid_ids=recent)
        o = res["outfits"][0] if res["outfits"] else None
        if o:
            for g in o["items"]:
                planned_worn[g["id"]] = day
        plan.append({"day": day, "outfit": o, "weather": res["weather"]})
    return plan