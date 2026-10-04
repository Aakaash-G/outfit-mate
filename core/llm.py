"""LLM stylist: intent parsing and outfit explanation.

Works with Gemini or any OpenAI-compatible API (OpenAI, Groq, ...). If no key is configured,
or a call fails, rule-based fallbacks keep the app working.
"""
import json
import re

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER
from .vision import FORMALITY_WORDS, PALETTE, describe

LAST_ERROR = None  # shown in the sidebar for debugging

SYSTEM_PROMPT = """You are Outfit Mate, a friendly personal stylist.
Rules:
- Recommend ONLY outfits from the candidate list you are given, using their ids.
- Respect the weather, the occasion's formality, the user's colour preferences and cultural dress norms.
- Never comment negatively on the user's body.
- Reply with JSON only."""


def enabled() -> bool:
    return bool(LLM_API_KEY)


def _call(user: str) -> str | None:
    global LAST_ERROR
    if not enabled():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        r = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}
            ],
        )
        return r.choices[0].message.content
    except Exception as ex:
        LAST_ERROR = f"{type(ex).__name__}: {ex}"
        return None


def _json(text: str | None):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0) if m else text)
    except Exception:
        return None


# ---------------------------------------------------------------- intent parsing
KEYWORDS = [
    (5, ["wedding", "reception", "engagement", "festival", "diwali", "pongal", "onam", "pooja", "puja", "temple"]),
    (4, ["interview", "office", "presentation", "meeting", "viva", "seminar", "conference", "formal", "placement"]),
    (3, ["dinner", "date", "party", "fest", "birthday", "restaurant", "outing", "smart"]),
    (1, ["gym", "workout", "run", "sleep", "home", "lounge", "yoga"]),
    (2, ["college", "class", "casual", "shopping", "movie", "friends", "travel", "trip", "mall", "lab"]),
]


def _rule_intent(text: str) -> dict:
    t = text.lower()
    formality, occasion = 2, "everyday"
    for level, words in KEYWORDS:
        hit = next((w for w in words if re.search(rf"\b{w}\b", t)), None)
        if hit:
            formality, occasion = level, hit
            break
    prefer, avoid = [], []
    for c in PALETTE:
        if re.search(rf"\b(no|not|avoid|without|except)\s+(any\s+)?{c}\b", t):
            avoid.append(c)
        elif re.search(rf"\b{c}\b", t):
            prefer.append(c)
    return {"occasion": occasion, "formality": formality, "prefer_colors": prefer,
            "avoid_colors": avoid, "style_notes": "", "source": "rules"}


def parse_intent(text: str, weather: dict) -> dict:
    prompt = f"""Convert the user's request into JSON with keys:
occasion (short string), formality (integer 1-5: 1 lounge, 2 casual, 3 smart casual, 4 formal, 5 very formal/festive),
prefer_colors (list), avoid_colors (list), style_notes (short string).
Colour names must come from: {", ".join(PALETTE)}.
Weather: {weather['temp_c']} C, rain probability {weather['rain_prob']:.0%}.
Request: "{text}\""""
    data = _json(_call(prompt))
    if not isinstance(data, dict):
        return _rule_intent(text)
    try:
        f = int(data.get("formality", 2))
    except (TypeError, ValueError):
        f = 2
    return {
        "occasion": str(data.get("occasion") or "everyday")[:60],
        "formality": max(1, min(5, f)),
        "prefer_colors": [c for c in data.get("prefer_colors", []) if c in PALETTE],
        "avoid_colors": [c for c in data.get("avoid_colors", []) if c in PALETTE],
        "style_notes": str(data.get("style_notes", ""))[:200],
        "source": "llm",
    }


# ---------------------------------------------------------------- choose + explain
def _template_reason(c: dict, intent: dict, weather: dict) -> tuple[str, str]:
    items = c["items"]
    title = ", ".join(describe(g) for g in items)
    bits = [f"A {FORMALITY_WORDS[intent['formality']]} look for {intent['occasion']}"]
    if all(set(g["colors"]) <= {"black", "white", "grey", "navy", "beige", "brown", "denim"} for g in items):
        bits.append("neutral colours keep it balanced")
    elif c["parts"]["compat"] >= 0.75:
        bits.append("the colours and patterns work well together")
    if weather["temp_c"] >= 28 and all(g["warmth"] <= 2 for g in items):
        bits.append(f"light enough for {weather['temp_c']:.0f} C")
    layer = next((g for g in items if g["slot"] == "layer"), None)
    if layer and weather["temp_c"] < 22:
        bits.append(f"the {layer['category']} keeps you warm at {weather['temp_c']:.0f} C")
    elif layer:
        bits.append(f"the {layer['category']} adds polish")
    return title[0].upper() + title[1:], "; ".join(bits) + "."


def choose_and_explain(candidates: list[dict], intent: dict, weather: dict, k: int = 3,
                       use_llm: bool = True) -> list[dict]:
    """Pick k outfits from ranked candidates and attach title + reason. Item ids are validated."""
    ids = {f"O{i + 1}": c for i, c in enumerate(candidates)}
    chosen = []
    if use_llm and enabled():
        lines = [f"{oid}: " + "; ".join(f"{describe(g)} (formality {g['formality']})" for g in c["items"])
                 + f" | model score {c['score']:.2f}" for oid, c in ids.items()]
        prompt = f"""Occasion: {intent['occasion']} (formality {intent['formality']}/5). Notes: {intent['style_notes']}
Weather: {weather['temp_c']} C, {weather['description']}, rain {weather['rain_prob']:.0%}.
Candidate outfits (from the user's own wardrobe):
{chr(10).join(lines)}

Choose the best {k} DIFFERENT outfits. Return JSON:
{{"outfits": [{{"id": "O1", "title": "short name", "reason": "1-2 friendly sentences"}}]}}"""
        data = _json(_call(prompt))
        for o in (data or {}).get("outfits", []):
            oid = str(o.get("id", "")).strip()
            if oid in ids and all(oid != x["_oid"] for x in chosen):  # validator: no invented outfits
                chosen.append(dict(ids[oid], title=str(o.get("title", ""))[:80],
                                   reason=str(o.get("reason", ""))[:400], _oid=oid))
            if len(chosen) == k:
                break
    for oid, c in ids.items():  # fill up with the highest-scoring remaining outfits
        if len(chosen) >= k:
            break
        if all(oid != x["_oid"] for x in chosen):
            title, reason = _template_reason(c, intent, weather)
            chosen.append(dict(c, title=title, reason=reason, _oid=oid))
    return chosen