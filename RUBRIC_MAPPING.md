# Outfit Mate: Rubric Mapping

**Team T20** · Akshay (CB.SC.U4CSE23168) · Aakash (CB.SC.U4CSE23162)

Where each phase-2 rubric criterion is covered in the project: the features, and the exact file and code that
implement them. (User Experience / Application Design is left out here.)

| Criterion | Marks | Main files |
|---|---|---|
| Data and Knowledge Preparation | 2 | `core/vision.py`, `genmodels/data.py`, `core/genstudio.py`, `sample_wardrobe/` |
| Prompt Engineering / Agent Design | 2 | `core/llm.py`, `core/stylist.py` |
| Output Quality / Accuracy | 2 | `core/recommender.py`, `genmodels/vae.py`, `scripts/evaluate.py`, `genmodels/evaluate.py` |
| Innovation and Real-world Impact | 2 | `core/stylist.py`, `core/genstudio.py`, `core/recommender.py` |

> Line numbers are from the final project files. If you edit a file, they may shift by a few lines;
> search for the function name instead.

---

## 1. Data and Knowledge Preparation (2)

### Features
- Two datasets: **Fashion-MNIST** (70,000 images) for the VAE, and a **labelled sample wardrobe** of 30 garments for testing.
- Automatic cleaning of garment photos: **background removal**, resizing, saving.
- Automatic labelling of garments with **CLIP** (category, pattern, formality) and **k-means** (colours).
- **Domain knowledge** encoded as data: clothing taxonomy, formality scale, colour palette, occasion keywords.
- Structured storage in an **SQLite database**, with data cleaning (missing photos, duplicates, tag correction).

### Where in the code

**Datasets**

| What | File | Location |
|---|---|---|
| Fashion-MNIST download, 60k train / 10k test split, batches of 128 | `genmodels/data.py` | `loaders()` lines 11–18 |
| 30 sample garment photos | `sample_wardrobe/*.png` | – |
| Correct category for each sample photo | `sample_wardrobe/labels.csv` | 30 rows |
| Bulk loading of photos into the database | `scripts/load_samples.py` | `main()` lines 22–52 |

**Data preparation (photos)**

| What | File | Location |
|---|---|---|
| Background removal with rembg (U²-Net, `u2netp`) | `core/vision.py` | `remove_background()` lines 93–99 |
| Main colours with k-means clustering | `core/vision.py` | `dominant_colours()` lines 112–134 |
| Resize and save the cleaned photo | `core/vision.py` | `save_image()` lines 137–142 |
| CLIP embedding (512 numbers per garment) | `core/vision.py` | `class Clip` lines 46–74 |
| Full tagging pipeline for one photo | `core/vision.py` | `tag_image()` lines 159–183 |
| Converting photos to Fashion-MNIST format for the VAE | `core/genstudio.py` | `to_fmnist()` lines 41–54 |

**Knowledge encoded in the system**

| Knowledge | File | Location |
|---|---|---|
| Clothing taxonomy: 21 categories with slot, formality, warmth | `core/vision.py` | `CATEGORIES` lines 10–21 |
| Pattern descriptions for CLIP | `core/vision.py` | `PATTERN_PROMPTS` lines 23–27 |
| Formality scale 1–5 | `core/vision.py` | `FORMALITY_PROMPTS`, `FORMALITY_WORDS` lines 28–32 |
| 17 named colours (RGB) and 7 neutral colours | `core/vision.py` | `PALETTE`, `NEUTRALS` lines 35–42 |
| Occasion keywords (incl. Indian festivals) → formality | `core/llm.py` | `KEYWORDS` lines 59–65 |
| Fashion compatibility rules | `core/recommender.py` | `compatibility()` lines 25–51 |
| Weather rules (heat, rain) | `core/recommender.py` | `filter_garments()` lines 55–70 |

**Storage and cleaning**

| What | File | Location |
|---|---|---|
| Database tables: garments, outfits, feedback, profile | `core/db.py` | `SCHEMA` lines 10–45 |
| Listing garments from the database | `core/db.py` | `list_garments()` lines 78–81 |
| Correcting wrong tags (Edit → Save) | `core/db.py` | `update_garment()` lines 84–89 |
| Reload without duplicates (`--fresh`) | `scripts/load_samples.py` | `main()` |

---

## 2. Prompt Engineering / Agent Design (2)

### Features
- A **system prompt** that sets the stylist's role and rules.
- **Structured JSON output** from the LLM, enforced by the API settings.
- **Two-stage LLM use**: (1) understand the request, (2) choose and explain outfits.
- **Grounding / validator**: the LLM can only choose outfit IDs it was given, so it can't invent clothes (no hallucination).
- **Output validation**: formality clamped to 1–5, unknown colours removed.
- **Fallbacks**: keyword rules and template reasons when there is no API key or the call fails.
- **Agent pipeline**: an orchestrator calls tools in order (weather → LLM → recommender → LLM → database).
- Low **temperature (0.4)** for consistent answers.

### Where in the code

| What | File | Location |
|---|---|---|
| System prompt (role, rules, "JSON only", "only use given IDs") | `core/llm.py` | `SYSTEM_PROMPT` lines 14–19 |
| Gemini call: temperature 0.4, `response_format="json_object"` | `core/llm.py` | `_call()` lines 26–45 |
| Safe JSON extraction from the reply | `core/llm.py` | `_json()` lines 48–55 |
| **Prompt 1**: request → occasion, formality, colours (JSON) | `core/llm.py` | `parse_intent()` lines 86–107 |
| Validation of the LLM's answer (clamp formality, filter colours) | `core/llm.py` | `parse_intent()` lines 96–107 |
| Fallback: keyword rules when the LLM is unavailable | `core/llm.py` | `KEYWORDS`, `_rule_intent()` lines 59–83 |
| **Prompt 2**: candidate outfits O1, O2… → choose best 3 + reasons | `core/llm.py` | `choose_and_explain()` lines 129–158 |
| **Validator**: only offered IDs accepted, no duplicates | `core/llm.py` | `choose_and_explain()` line 147 |
| Fallback reasons without an LLM | `core/llm.py` | `_template_reason()` lines 111–126 |
| **Agent orchestrator**: weather → intent → ranking → choice → save | `core/stylist.py` | `recommend()` lines 10–44 |
| Weather "tool" with a safe default | `core/weather.py` | `get_weather()` lines 11–30 |
| LLM provider, model and key settings | `core/config.py` | lines 18–22 |
| Last LLM error shown in the sidebar | `app.py` | sidebar block |

---

## 3. Output Quality / Accuracy (2)

### Features
- **Outfit quality** from a weighted score: compatibility, occasion match, personal style, recently-worn penalty.
- **Hard filters** so unsuitable clothes are never suggested (wrong formality, avoided colours, weather).
- **Diversity**: the top results aren't all built on the same top.
- **Explainability**: every suggestion shows its reason and its score breakdown ("Why this score?").
- **Measured tagging accuracy** on 30 labelled photos: category accuracy, slot accuracy, per-class accuracy, common mistakes, time per photo.
- **Measured VAE quality**: test loss (−ELBO), reconstruction, KL, MSE and FID, with best/worst reference values.
- **Upload quality check**: unusual photos are flagged using the VAE's reconstruction error.
- **Automatic tests** for the outfit logic.

### Where in the code

**Recommendation quality**

| What | File | Location |
|---|---|---|
| Score weights 0.45 / 0.25 / 0.20 / 0.10 | `core/config.py` | line 33 |
| Score formula and ranking | `core/recommender.py` | `rank_outfits()` lines 86–150 (formula at line 134) |
| Compatibility rules | `core/recommender.py` | `compatibility()` lines 25–51 |
| Occasion match (CLIP similarity) | `core/recommender.py` | `query_text()` lines 20–21, `qsim()` in `rank_outfits()` |
| Filtering unsuitable clothes | `core/recommender.py` | `filter_garments()` lines 55–70 |
| Recently-worn penalty | `core/recommender.py` | `_recency_penalty()` lines 73–82 |
| Diversity (max 2 outfits per top) | `core/recommender.py` | end of `rank_outfits()` lines 140–150 |
| Score breakdown saved for "Why this score?" | `core/recommender.py` | `"parts"` in `rank_outfits()` |
| Timing of each step (speed results) | `core/stylist.py` | `recommend()` (`time.perf_counter()`) |

**Tagging accuracy**

| What | File | Location |
|---|---|---|
| Accuracy test on the 30 labelled photos | `scripts/evaluate.py` | `labelled_images()` lines 27–43, `main()` lines 46–81 |
| Confidence warning for unsure tags (< 0.35) | `core/vision.py` / `app.py` | `tag_image()` → `confidence` |

Run: `python scripts/evaluate.py`

**VAE quality**

| What | File | Location |
|---|---|---|
| Loss: reconstruction + KL divergence | `genmodels/vae.py` | `VAE.loss()` lines 55–64 |
| Testing on 10,000 unseen images | `genmodels/train.py` | `train()` lines 73–108 |
| Sample images, reconstructions, loss curve | `genmodels/train.py` | `save_figures()` lines 38–70 → `checkpoints/samples/` |
| FID score | `genmodels/evaluate.py` | `frechet()` lines 65–71, `main()` lines 86–125 |
| Feature classifier for FID | `genmodels/evaluate.py` | `get_classifier()` lines 38–62 |
| Unusual-photo check (z-score) | `core/genstudio.py` | `photo_check()` lines 96–104 |

Run: `python -m genmodels.evaluate`

**Automatic tests**

| Test | File | Line |
|---|---|---|
| Interview → formal clothes only | `tests/test_recommender.py` | 57 |
| No hoodie in the heat | `tests/test_recommender.py` | 65 |
| Avoided colour is removed | `tests/test_recommender.py` | 70 |
| Two patterns score lower | `tests/test_recommender.py` | 77 |
| Recently worn ranks lower | `tests/test_recommender.py` | 83 |
| Keyword parser works | `tests/test_recommender.py` | 91 |
| Fallback returns 3 different outfits | `tests/test_recommender.py` | 98 |

Run: `python -m pytest -q`

> Put your measured numbers here before the review:
> tagging accuracy ____ %, slot accuracy ____ %, VAE test −ELBO ____, FID ____, acceptance rate ____ %.

---

## 4. Innovation and Real-world Impact (2)

### Features
- **Uses the clothes you already own**, unlike shopping apps that push new purchases. This encourages
  **sustainable fashion** and reduces waste.
- **Three AI techniques combined** in one app: CLIP (vision + language), an LLM (reasoning + explanation) and a
  **VAE trained from scratch** (generation).
- **Learns your taste over time** from likes, skips and wears, without retraining any model.
- **Context-aware**: weather, occasion and what you wore recently.
- **Indian context**: kurtas and sarees in the taxonomy; Diwali, Pongal and Onam in the occasion keywords.
- **Generative design features**: new clothing designs, mixing two garments, photo clean-up.
- **"Forgotten clothes"** insight that shows items you never wear.
- **Weekly planner** that avoids repeating outfits.
- **Keeps working when online services fail**: if the LLM or weather API is unavailable, rule-based fallbacks take over.
- **Low cost**: runs on a normal laptop with free APIs; no GPU needed.

### Where in the code

| What | File | Location |
|---|---|---|
| Learning taste from feedback (style vector, moving average) | `core/stylist.py` | `feedback()` lines 47–62 |
| Style vector stored per user | `core/db.py` | `get_style_vector()`, `set_style_vector()` lines 143–152 |
| Personal style in the score | `core/recommender.py` | `rank_outfits()` (`style_vec`) |
| Weather-aware suggestions | `core/weather.py`, `core/recommender.py` | `get_weather()`; `filter_garments()` |
| Avoiding recently worn clothes | `core/db.py`, `core/recommender.py` | `last_worn()` lines 111–122; `_recency_penalty()` |
| Weekly planner without repeats | `core/stylist.py` | `plan_week()` lines 65–82 |
| Indian garments | `core/vision.py` | `CATEGORIES` ("kurta", "saree") |
| Indian festivals | `core/llm.py` | `KEYWORDS` ("diwali", "pongal", "onam", "pooja") |
| VAE trained from scratch | `genmodels/vae.py` | `class VAE` lines 38–97 |
| New designs | `genmodels/vae.py` / `core/genstudio.py` | `VAE.sample()` lines 68–70; `generate()` |
| Mixing two garments (latent interpolation) | `genmodels/vae.py` / `core/genstudio.py` | `VAE.interpolate()` lines 78–82; `interpolate()` |
| Photo clean-up | `core/genstudio.py` | `clean_up()` lines 74–80 |
| Find similar clothes | `core/genstudio.py` | `similar()` lines 83–93 |
| Acceptance rate and never-worn clothes | `core/db.py` | `feedback_stats()` lines 136–139, `wear_counts()` lines 125–133 |
| Fallbacks when APIs fail | `core/llm.py`, `core/weather.py` | `_rule_intent()`, `_template_reason()`; `DEFAULT` |
