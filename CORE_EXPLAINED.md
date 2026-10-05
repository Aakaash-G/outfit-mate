# The `core` Folder Explained

**Outfit Mate · Team T20**

The `core` folder holds all the logic of the app. `app.py` (the screen) only displays results and calls
functions from here. Each file has one job:

| File | Job | Used by |
|---|---|---|
| `__init__.py` | Empty. Tells Python that `core` is a package, so `from core import db` works. | – |
| `config.py` | Settings, API keys, folder paths, score weights | every other file |
| `db.py` | Saves and reads data in the SQLite database | stylist, app, genstudio |
| `vision.py` | Recognises clothes in photos with CLIP | app, recommender, llm |
| `weather.py` | Gets temperature and rain | stylist |
| `llm.py` | Talks to Gemini: understands requests, explains outfits | stylist |
| `recommender.py` | Builds and scores outfits | stylist |
| `stylist.py` | Connects all the above in the right order | app |
| `genstudio.py` | Connects the trained VAE to the app | app |

**Order of a request:**

```
app.py → stylist.recommend()
           ├── weather.get_weather()
           ├── llm.parse_intent()
           ├── recommender.rank_outfits()   (uses CLIP embeddings saved by vision.py)
           ├── llm.choose_and_explain()
           └── db.save_outfit()
```

---

## 1. `config.py`: Settings

Every other file imports its settings from here, so a setting only needs changing in one place.

### Loading the `.env` file
```python
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
```
`load_dotenv()` reads `.env` and makes each line (like `LLM_API_KEY=...`) available as an environment variable.
The `try/except` lets the app start even if `python-dotenv` isn't installed.

### Folder paths
```python
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "outfit_mate.db"
for _d in (DATA_DIR, IMG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
```
- `Path(__file__)` is this file's location. `.parent.parent` goes up from `core/config.py` to the `outfit_mate` folder.
- All other paths are built from `BASE_DIR`, so the project works wherever it's placed.
- `mkdir(..., exist_ok=True)` creates the folders if they're missing, and does nothing if they exist.

### Reading settings
```python
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash").strip()
```
`os.getenv("NAME", default)` reads a setting, using the default if it's missing. With no key, the provider is `"none"`
and the app uses its keyword-rule fallback.

### CLIP model and score weights
```python
CLIP_MODEL = "ViT-B-32"; CLIP_PRETRAINED = "laion2b_s34b_b79k"
W_COMPAT, W_QUERY, W_STYLE, W_RECENCY = 0.45, 0.25, 0.20, 0.10
```
- **ViT-B-32**: a Vision Transformer CLIP model. The weights were trained on LAION-2B (2 billion image–text pairs).
- The four **weights** are used in the outfit score formula in `recommender.py`.

---

## 2. `db.py`: Database

Stores everything in one SQLite file, `data/outfit_mate.db`. SQLite is built into Python, so there's no server.
Other files never write SQL; they call these functions.

### The tables (`SCHEMA`)
| Table | Stores |
|---|---|
| **garments** | id, name, category, slot, colours (JSON text), pattern, formality 1–5, warmth 1–3, confidence, image path, **embedding** (512 CLIP numbers as bytes) |
| **outfits** | garment IDs (JSON list), occasion, score, title, reason |
| **feedback** | outfit id, action (like / skip / worn), date |
| **profile** | one row: `style_vector`, the user's taste |

`CREATE TABLE IF NOT EXISTS` means running it again is safe.

### Connecting
```python
def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c
```
- `row_factory = sqlite3.Row` lets code read columns by name: `row["category"]`.
- `check_same_thread=False` is needed because Streamlit may use different threads.
- Used as `with _conn() as c:`, which saves (commits) changes automatically at the end of the block.

`init()` runs the schema to create the tables.

### Photo paths: `resolve_image()`
Photos are stored as relative paths like `data/images/ab12.png`. This turns them into full paths. If the file isn't
found (e.g. the folder was moved), it looks for the same file name in `data/images`.

### Converting rows: `_row_to_garment()`
```python
g["colors"] = json.loads(g["colors"] or "[]")
g["embedding"] = np.frombuffer(g["embedding"], dtype=np.float32)
```
The database stores colours as text and the embedding as bytes; this converts them back into a Python list and a
NumPy array.

### Adding a garment: `add_garment(tags, image_path)`
Inserts one row. The `?` placeholders are filled safely by SQLite (prevents SQL injection). The embedding is stored with
`.tobytes()`. Returns the new ID (`cur.lastrowid`).

### Listing and cleaning
- `list_garments()`: all garments, newest first, **skipping any whose photo file is missing** so the app can't crash.
- `remove_missing()`: permanently deletes those broken rows. Called when the app starts.

### Editing and deleting
```python
def update_garment(gid, **fields):
    cols = ", ".join(f"{k}=?" for k in fields)
    c.execute(f"UPDATE garments SET {cols} WHERE id=?", (*fields.values(), gid))
```
Accepts any columns as keyword arguments, e.g. `update_garment(5, category="formal pants", formality=4)`.
Used by the **Save** button in the Edit panel. `delete_garment(gid)` removes one row.

### Outfits and feedback
- `save_outfit(...)`: stores a suggestion; garment IDs become a JSON list like `"[3, 12, 20]"`.
- `add_feedback(outfit_id, action, day)`: stores like / skip / worn with the date.

### History
- `last_worn()`: joins **feedback** and **outfits** to find the latest date each garment was worn. Used to avoid repeats.
- `wear_counts()`: how many times each garment was worn (for "never worn").
- `feedback_stats()`: `GROUP BY action` to count likes, skips and wears (for the acceptance rate).

### Style vector
`get_style_vector()` and `set_style_vector(v)` read and save the taste vector. `INSERT OR REPLACE` keeps only one row.

---

## 3. `vision.py`: Recognising clothes

Turns a photo into tags: **category, slot, colours, pattern, formality** and a **CLIP embedding**.

### Lists and constants
```python
CATEGORIES = {
    "shirt": ("top", 3, 1),
    "formal pants": ("bottom", 4, 2),
    "track pants": ("bottom", 1, 1),
    ...
}
```
- Each category maps to **(slot, default formality, warmth)**.
- Adding a category here automatically adds it to CLIP's choices and the Edit dropdown.
- `PATTERN_PROMPTS`, `FORMALITY_PROMPTS`: the text CLIP compares photos against.
- `PALETTE`: colour names with RGB values. `NEUTRALS`: colours that go with anything (black, white, grey, navy…).

### The CLIP wrapper: `class Clip`
```python
self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
```
- **CLIP** maps images and text into the **same 512-number space**, so a shirt photo is close to the text "a photo of a shirt".
- `image(img)`: preprocesses (resize to 224×224), encodes, and normalises the vector to length 1.
- `text(texts)`: encodes text prompts; results are cached so the same prompt is never encoded twice.
- `torch.no_grad()`: no gradient tracking, since we only use the model and don't train it. Saves memory.
- Because vectors have length 1, **dot product = cosine similarity**.

### Zero-shot classification: `zero_shot()`
```python
t = clip.text(prompts)
return _softmax(100.0 * t @ img_emb)
```
- `t @ img_emb` gives the similarity between the photo and every prompt at once.
- ×100 is CLIP's temperature (sharpens differences); **softmax** turns scores into probabilities.
- **Zero-shot**: no training on our categories, only text prompts.

### Background removal: `remove_background()`
```python
_REMBG_SESSION = new_session("u2netp")
return remove(small, session=_REMBG_SESSION).convert("RGBA")
```
- Uses **rembg** with the small **u2netp** model (U²-Net, about 4 MB). The default model was too large for the laptop's memory.
- The model loads once and is reused. Big photos are shrunk to 1024 px first.
- Output is RGBA: the alpha (A) channel is 0 for background pixels.
- If it fails, the original photo is returned so tagging still works.

### Colours: `dominant_colours()`
- **k-means clustering** with k = 3 on the garment's pixels (alpha > 128): start with 3 random colours, then 12 times
  assign each pixel to the nearest centre and move each centre to the average.
- Clusters under 15 % are ignored. Each centre is named by `_nearest_colour()` (closest PALETTE entry).
- `on_white()` pastes the cut-out onto a white background.

### Saving: `save_image()`
Shrinks to 768 px, saves with a random unique name (`uuid4`), returns a relative path like `data/images/ab12.png`.

### The main function: `tag_image(img, clip)`
```python
cut = remove_background(img)
clean = on_white(cut)
emb = clip.image(clean)
p_cat = zero_shot(clip, emb, [f"a photo of a {c}" for c in cats])
category = cats[int(p_cat.argmax())]
...
formality = round(0.5 * prior_formality + 0.5 * clip_formality)
```
1. Remove the background and put the garment on white.
2. Get the CLIP embedding (saved in the database, reused for scoring).
3. **Category** = most likely prompt; slot and warmth come from `CATEGORIES`.
4. **Pattern** the same way.
5. **Formality** = average of the category's default and CLIP's estimate (the expected value of the 1–5 probabilities).
6. **Colours** from k-means; a name like "navy striped shirt".
7. `confidence` = top category probability. Under 0.35 the app warns the user.

`describe(g)` turns a garment into a short readable label.

---

## 4. `weather.py`: Weather

```python
r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                 params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}, timeout=8)
entries = [e for e in r.json()["list"] if e["dt_txt"].startswith(day.isoformat())]
e = min(entries, key=lambda e: abs(int(e["dt_txt"][11:13]) - 13))
```
- Calls the free **5-day forecast**. `units=metric` gives °C; `timeout=8` stops it hanging.
- The forecast has one entry every 3 hours. It keeps the chosen date and picks the reading **closest to 1 pm**.
- Returns `temp_c`, `rain_prob` (`pop` = probability of precipitation, 0–1) and a description.
- No key, no internet, or a date past 5 days → returns `DEFAULT` (28 °C, 20 % rain). It never crashes the app.

---

## 5. `llm.py`: The language model (Gemini)

Two jobs: **understand the request** and **choose + explain outfits**. This file is the project's
**prompt engineering**.

### The system prompt
```python
SYSTEM_PROMPT = """You are Outfit Mate, a friendly personal stylist.
Rules:
- Recommend ONLY outfits from the candidate list you are given, using their ids.
- Respect the weather, the occasion's formality, the user's colour preferences and cultural dress norms.
- Never comment negatively on the user's body.
- Reply with JSON only."""
```
Sets the LLM's role and rules for every request. The first rule prevents **hallucination** (inventing clothes).

### Calling Gemini: `enabled()` and `_call()`
```python
r = client.models.generate_content(
    model=LLM_MODEL, contents=user,
    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.4,
                                       response_mime_type="application/json"))
```
- `temperature=0.4`: low randomness, so answers are consistent.
- `response_mime_type="application/json"`: forces JSON output.
- Also supports OpenAI-compatible APIs (OpenAI, Groq).
- On any error it saves the message in `LAST_ERROR` (shown in the sidebar) and returns `None` → fallback is used.

### Reading the answer: `_json()`
Finds the `{ ... }` part of the reply with a regular expression and parses it. Returns `None` if it isn't valid JSON.

### Fallback rules: `KEYWORDS` and `_rule_intent()`
```python
KEYWORDS = [
    (5, ["wedding", "reception", "festival", ...]),
    (4, ["interview", "office", "presentation", ...]),
    (1, ["gym", "workout", ...]),
    ...
]
```
- Without an LLM, keywords set the formality: "interview" → 4, "wedding" → 5, "gym" → 1.
- `\b` in the regex = word boundary, so "run" doesn't match inside "brunch".
- Colours after "no / not / avoid / without" go to `avoid_colors`; other colour words go to `prefer_colors`.

### Understanding the request: `parse_intent(text, weather)`
- Asks the LLM for JSON with: occasion, formality (1–5), prefer_colors, avoid_colors, style_notes.
- **Validates** the reply: formality clamped to 1–5, unknown colours removed, text length limited.
- If the reply is unusable → `_rule_intent()`.

Example: `"placement interview, no black"` → `{"occasion": "interview", "formality": 4, "avoid_colors": ["black"]}`

### Choosing and explaining: `choose_and_explain(candidates, intent, weather, k=3)`
```python
ids = {f"O{i + 1}": c for i, c in enumerate(candidates)}
...
if oid in ids and all(oid != x["_oid"] for x in chosen):   # validator
    chosen.append(...)
```
1. Each candidate outfit gets an ID: O1, O2, O3…
2. The prompt lists the candidates, the occasion and the weather, and asks for the best 3 **different** outfits as JSON
   (id, title, reason).
3. **Validator**: only IDs that were offered are accepted, and duplicates are skipped. This **grounds** the LLM in the
   real wardrobe.
4. If the LLM is off or returns fewer than 3, the highest-scoring remaining outfits are added with a reason from
   `_template_reason()` (e.g. "neutral colours keep it balanced; light enough for 29 C").

---

## 6. `recommender.py`: Building and scoring outfits

The core algorithm. No LLM here, so it's fast and testable.

```
score = 0.45 × compatibility + 0.25 × occasion match + 0.20 × personal style − 0.10 × recently worn
```

### `query_text(intent)`
Makes a sentence like `"a photo of formal clothing for interview"`. Each garment's CLIP similarity to this sentence
is its **occasion match**.

### `compatibility(items)`: do these clothes go together?
```python
s = 0.8
if patterned >= 2:      s -= 0.35   # two patterns clash
if len(bold) >= 3:      s -= 0.25   # too many strong colours
elif not bold:          s += 0.05   # all neutral is safe
s -= 0.1 * max(0, max(f) - min(f) - 1)   # mixing gym and formal
```
- Also −0.1 if top and bottom are the same bold colour.
- Plus a small bonus or penalty (±0.1) from how similar the items' CLIP embeddings are to each other.
- Result kept between 0 and 1.

### `filter_garments(garments, intent, weather)`
Removes:
- colours the user wants to avoid
- items more than 1 formality level away (shoes get one extra level)
- warm items (hoodies, sweaters) at 24 °C or above
- sandals for formal occasions when rain is likely

### `_recency_penalty(items, worn, day)`
Worn today = 1.0, 1 day ago = 0.75, 2 days = 0.5, 3 days = 0.25, older = 0. Averaged over the outfit.

### `rank_outfits(...)`: the main function
1. Encode the occasion sentence with CLIP (`q`). `qsim(g)` = garment's similarity to it (+0.1 for a preferred colour).
2. **Filter**, then keep the **6 best items per slot**. If no top+bottom pair is left, relax the formality tolerance and retry.
3. **Combine**: every top × bottom, plus every one-piece. Add shoes if available. Add a layer only when it's below 22 °C
   or the occasion is formal.
4. **Score** each combination with the formula. The four parts are kept for the "Why this score?" panel.
5. **Sort** by score, then keep a **varied top 10**: at most 2 outfits built on the same top.

`avoid_ids` (used by the week planner) drops recently planned items if alternatives exist.

---

## 7. `stylist.py`: Connecting everything

The app only calls these three functions.

### `recommend(prompt, city, day, clip)`
```python
wx = weather.get_weather(city, day)
intent = llm.parse_intent(prompt, wx)
cands = recommender.rank_outfits(intent, wx, garments, clip, db.get_style_vector(), worn, day)
outfits = llm.choose_and_explain(cands, intent, wx, k=3)
db.save_outfit(...)
```
- Raises a clear error if the wardrobe is empty.
- `time.perf_counter()` times each step; the app shows the total (useful as your speed result).

### `feedback(outfit, action)`: learning the user's taste
```python
new = 0.85 * style + 0.15 * mean      # like / worn
new = style - 0.05 * mean             # skip
```
- `mean` = average CLIP embedding of the outfit's items.
- **Like / wear**: the style vector moves 15 % towards the outfit (an **exponential moving average**).
- **Skip**: it moves slightly away.
- The vector is normalised back to length 1. Future outfits similar to it score higher.

### `plan_week(prompt, city, start, clip)`
Parses the request once, then picks one outfit per day for 7 days. Planned items are treated as "worn", so tops and
bottoms don't repeat on consecutive days (shoes may). Uses `use_llm=False` to stay fast.

---

## 8. `genstudio.py`: Connecting the VAE

Loads the trained VAE (`checkpoints/vae.pt`) and prepares the user's photos for it.

### Loading: `trained()`, `metrics()`, `vae()`
```python
@lru_cache(maxsize=1)
def vae():
    ck = torch.load(VAE_PATH, map_location="cpu", weights_only=False)
```
- `trained()` checks the model file exists; `metrics()` reads the saved test results.
- `@lru_cache` loads the model once and reuses it. `map_location="cpu"` works even if trained on a GPU.

### Converting photos: `to_fmnist(path)`
The VAE only knows **28×28 grey images with a white garment on black**. So the photo is: turned grey → **inverted** →
cropped to the garment → padded to a square → resized to 24×24 → centred on a 28×28 canvas → scaled to 0–1.

### Features
| Function | What it does |
|---|---|
| `to_pil()` | Turns VAE output into an image grid for the screen. |
| `generate(n, temperature)` | New designs from random latent codes. |
| `interpolate(a, b)` | Morphs garment A into garment B. |
| `clean_up(path, noise)` | Adds noise, lets the VAE rebuild it → returns original, noisy, rebuilt. |
| `similar(target, garments)` | Ranks garments by cosine similarity of their VAE codes (μ). |
| `photo_check(path)` | Reconstruction error as a **z-score** vs. real clothes; above 4 = unusual photo. |

---

## Quick summary

| If you're asked about… | Look at |
|---|---|
| How clothes are recognised | `vision.py` → `tag_image()`, `zero_shot()` |
| How outfits are chosen | `recommender.py` → `rank_outfits()`, `compatibility()` |
| How the LLM is used safely | `llm.py` → `SYSTEM_PROMPT`, `choose_and_explain()` validator |
| How the app learns taste | `stylist.py` → `feedback()` |
| Where data is stored | `db.py` → `SCHEMA` |
| How the VAE is used in the app | `genstudio.py` |
