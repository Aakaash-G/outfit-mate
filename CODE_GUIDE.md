# Outfit Mate: Code Guide

**Team T20** · Akshay (CB.SC.U4CSE23168) · Aakash (CB.SC.U4CSE23162)

A short guide to every code file in the project: what each file is for, and what its main blocks of code do.
The key terms used throughout are explained first.

---

## Key terms

| Keyword | What it means |
|---|---|
| **Generative AI** | AI that creates new content (text, images) instead of only classifying it. In this project: the VAE and the LLM. |
| **CLIP** | A pretrained model by OpenAI that understands images and text in the same "space". We use it to recognise clothes in photos. |
| **Embedding** | A list of numbers that represents an image or text. Similar things get similar numbers. CLIP gives 512 numbers per image. |
| **Cosine similarity** | How alike two embeddings are, from −1 to 1. Calculated as a dot product of vectors of length 1. |
| **Zero-shot classification** | Classifying without training: CLIP compares a photo with text like "a photo of a shirt" and picks the closest. |
| **Softmax** | Turns a list of scores into probabilities that add up to 1. |
| **Background removal (rembg / U²-Net)** | Cuts the garment out of the photo so only the clothing is analysed. U²-Net is the neural network that does it. |
| **k-means clustering** | Groups similar pixel colours into k groups. Used to find a garment's main colours. |
| **LLM (Large Language Model)** | A model that understands and writes text. We use **Gemini** to read the user's request and explain outfits. |
| **Prompt / System prompt** | The instructions sent to the LLM. The system prompt sets its role and rules for every request. |
| **JSON** | A text format for structured data, e.g. `{"occasion": "interview", "formality": 4}`. The LLM is told to reply only in JSON. |
| **Hallucination** | When an LLM makes something up. Here it would mean suggesting clothes the user doesn't own. |
| **Grounding / validator** | Our fix for hallucination: the LLM may only pick outfit IDs we give it, and any other ID is rejected. |
| **Fallback** | A simpler backup used when something fails, e.g. keyword rules when there's no LLM key. |
| **Formality (1–5)** | Our scale: 1 gym/lounge, 2 casual, 3 smart casual, 4 formal, 5 very formal/festive. |
| **Slot** | Where a garment is worn: top, bottom, onepiece, shoes or layer. |
| **Compatibility score** | How well the items in an outfit go together, from fashion rules (patterns, colours, formality). |
| **Style vector** | One embedding that represents the user's taste. It moves towards liked outfits. |
| **Exponential moving average** | Updating a value as `0.85 × old + 0.15 × new`, so recent feedback counts most. Used for the style vector. |
| **VAE (Variational Autoencoder)** | Our generative model. It compresses an image into a few numbers and rebuilds it, and can create new images. |
| **Encoder / Decoder** | The encoder turns an image into numbers; the decoder turns numbers back into an image. |
| **Latent space / latent code (z)** | The small set of numbers (16 in our VAE) that describes an image. |
| **μ (mean) and σ² (variance)** | The VAE's encoder outputs a range of possible codes (a Gaussian) instead of one fixed code. |
| **Reparameterisation trick** | `z = μ + σ × ε`, with ε random. It lets us train through the random sampling step. |
| **Reconstruction loss (BCE)** | How different the rebuilt image is from the original. BCE = binary cross-entropy. |
| **KL divergence** | Measures how far the VAE's codes are from a standard normal distribution N(0, 1). Keeps the latent space smooth. |
| **ELBO** | Evidence Lower Bound. VAE loss = −ELBO = reconstruction + KL. |
| **Latent interpolation** | Decoding points between two codes, so one garment morphs into another. |
| **Anomaly score / z-score** | Reconstruction error compared with normal clothes. A high z-score means an unusual photo. |
| **Fashion-MNIST** | A dataset of 70,000 grey 28×28 clothing images in 10 classes. Used to train the VAE. |
| **Convolution (Conv2d / ConvTranspose2d)** | Neural network layers for images. Conv2d shrinks the image while finding features; ConvTranspose2d grows it back. |
| **ReLU / Sigmoid** | Activation functions. ReLU sets negatives to 0; Sigmoid squashes values into 0–1. |
| **Epoch / Batch** | An epoch is one pass over all training images; a batch is the group of images (128) processed at once. |
| **Adam optimiser** | The algorithm that updates the network's weights during training. |
| **Backpropagation** | Working out how each weight affected the loss, so it can be adjusted. |
| **FID (Fréchet Inception Distance)** | Measures how similar generated images are to real ones. Lower is better. |
| **PyTorch** | The deep learning library used to build and train the VAE. |
| **Streamlit** | A Python library that turns a script into a web app. |
| **Session state / cache** | Streamlit re-runs the script on every click. Session state keeps values between clicks; the cache keeps big models loaded. |
| **SQLite** | A small database stored in one file, built into Python. |
| **API / API key** | A way to use an online service (Gemini, OpenWeatherMap). The key identifies you; it's kept in `.env`. |
| **.env file** | A file holding private settings like API keys. It's never uploaded to GitHub. |

---

## How the files fit together

```
User types an occasion
  → app.py → stylist.recommend()
      → weather.get_weather()          temperature and rain
      → llm.parse_intent()             occasion + formality
      → recommender.rank_outfits()     filter → combine → score
      → llm.choose_and_explain()       best 3 + reasons
      → db.save_outfit()
  → 3 outfit cards → Like / Skip / Wear → stylist.feedback() updates the style vector

Separately: genmodels/train.py trains the VAE → checkpoints/vae.pt → genstudio.py → Design Studio tab
```

---

## core/config.py: Settings

**Purpose:** one place for all settings. Every other file imports from here.

| Block | What it does |
|---|---|
| `load_dotenv()` | Reads API keys from the `.env` file. |
| `BASE_DIR`, `DATA_DIR`, `IMG_DIR`, `DB_PATH` | Folder and file locations, built from the project folder. Missing folders are created. |
| `os.getenv(...)` lines | Read each setting, with a default if it's missing (LLM provider, model, weather key, city). |
| `CLIP_MODEL`, `CLIP_PRETRAINED` | Which CLIP model and weights to load (ViT-B-32, trained on LAION-2B). |
| `W_COMPAT, W_QUERY, W_STYLE, W_RECENCY` | The four weights in the outfit score: 0.45, 0.25, 0.20, 0.10. |

---

## core/db.py: Database

**Purpose:** stores everything in one SQLite file, `data/outfit_mate.db`.

| Block | What it does |
|---|---|
| `SCHEMA` | Creates four tables: **garments**, **outfits**, **feedback**, **profile**. |
| `_conn()`, `init()` | Open the database and create the tables. |
| `resolve_image()` | Turns a stored photo path into a real file path, even if the project folder moved. |
| `_row_to_garment()` | Converts a database row back into Python (colours list, embedding array). |
| `add_garment()` | Saves a new garment with its tags and CLIP embedding. |
| `list_garments()`, `remove_missing()` | List all garments; remove records whose photo file is gone. |
| `update_garment()`, `delete_garment()` | Edit or delete a garment (used by the Edit panel). |
| `save_outfit()`, `add_feedback()` | Save a suggestion; save a like / skip / worn action. |
| `last_worn()`, `wear_counts()`, `feedback_stats()` | Read history: when each item was last worn, how often, and feedback totals. |
| `get_style_vector()`, `set_style_vector()` | Read and save the user's taste vector. |

---

## core/vision.py: Recognising clothes

**Purpose:** turns a photo into tags: category, slot, colours, pattern, formality, plus a CLIP embedding.

| Block | What it does |
|---|---|
| `CATEGORIES` | Master list. Each category has (slot, default formality, warmth), e.g. `"formal pants": ("bottom", 4, 2)`. |
| `PATTERN_PROMPTS`, `FORMALITY_PROMPTS` | The text CLIP compares photos against. |
| `PALETTE`, `NEUTRALS` | Named colours with RGB values; colours that go with anything. |
| `class Clip` | Loads CLIP once. `image()` and `text()` turn photos and text into normalised embeddings. |
| `zero_shot()` | Compares a photo with many prompts and returns probabilities (softmax). |
| `remove_background()` | Cuts out the garment with rembg's small `u2netp` model; falls back to the original if it fails. |
| `dominant_colours()` | k-means on the garment's pixels to find its 1–2 main colours. |
| `save_image()` | Saves the cleaned photo with a unique name in `data/images`. |
| `tag_image()` | **The main function:** background removal → CLIP embedding → category, pattern, formality → colours → name. |
| `describe()` | A readable label like "navy striped shirt". |

---

## core/weather.py: Weather

**Purpose:** gets the temperature and chance of rain for a city and date.

| Block | What it does |
|---|---|
| `DEFAULT` | 28 °C and 20 % rain, used when there's no key or no internet. |
| `get_weather()` | Calls the OpenWeatherMap 5-day forecast and picks the reading closest to 1 pm on that date. |

---

## core/llm.py: Language model (Gemini)

**Purpose:** understands the user's request and picks and explains the final outfits. This is the **prompt engineering** part.

| Block | What it does |
|---|---|
| `SYSTEM_PROMPT` | The LLM's role and rules: only pick from the given outfits, respect weather and formality, be kind, reply in JSON. |
| `enabled()`, `_call()` | Check a key exists; call Gemini (or any OpenAI-compatible API) with low temperature and JSON output. |
| `_json()` | Safely pulls JSON out of the LLM's reply. |
| `KEYWORDS`, `_rule_intent()` | **Fallback:** keyword rules (interview → 4, gym → 1) and colour words when there's no LLM. |
| `parse_intent()` | Sends the request to the LLM and validates the answer (formality 1–5, known colours only). |
| `_template_reason()` | Builds a simple reason sentence when the LLM isn't available. |
| `choose_and_explain()` | LLM picks the best 3 of the candidates by ID; the **validator** rejects any ID that wasn't offered. |

---

## core/recommender.py: Building and scoring outfits

**Purpose:** the core algorithm. Finds the best outfits from the wardrobe.

```
score = 0.45 × compatibility + 0.25 × occasion match + 0.20 × personal style − 0.10 × recently worn
```

| Block | What it does |
|---|---|
| `query_text()` | Makes a sentence like "a photo of formal clothing for interview" to compare garments with. |
| `compatibility()` | Fashion rules: two patterns −0.35, three or more bold colours −0.25, all neutral +0.05, mixed formality penalised, plus a small CLIP "belongs together" bonus. |
| `filter_garments()` | Removes avoided colours, wrong formality, warm items in the heat, and sandals in rain for formal occasions. |
| `_recency_penalty()` | Penalty for items worn in the last 3 days. |
| `rank_outfits()` | **The main function:** filter → keep the 6 best per slot → make top × bottom (+ shoes, + layer) combinations → score each → sort → keep a varied top 10. |

---

## core/stylist.py: Connecting everything

**Purpose:** calls the other modules in the right order. The app only uses these three functions.

| Block | What it does |
|---|---|
| `recommend()` | Weather → LLM parses request → recommender scores → LLM chooses and explains → save. Times each step. |
| `feedback()` | Saves the action and updates the style vector: like/wear moves it 15 % towards the outfit, skip moves it slightly away. |
| `plan_week()` | 7 outfits, one per day, avoiding repeated tops and bottoms on consecutive days. |

---

## core/genstudio.py: VAE in the app

**Purpose:** loads the trained VAE and prepares the user's photos for it.

| Block | What it does |
|---|---|
| `trained()`, `metrics()` | Check `checkpoints/vae.pt` exists; read the saved test results. |
| `vae()` | Loads the model once and reuses it. |
| `to_fmnist()` | Converts a photo to Fashion-MNIST style: grey, inverted, cropped, centred, 28×28. |
| `to_pil()` | Turns VAE output into a picture grid for the screen. |
| `generate()`, `interpolate()`, `clean_up()` | New designs; morph A → B; add noise and rebuild. |
| `similar()` | Finds the most similar garments by comparing VAE codes. |
| `photo_check()` | Reconstruction error as a z-score; above 4 means an unusual photo. |

---

## genmodels/vae.py: The Variational Autoencoder

**Purpose:** the generative model we trained. **Most important file for the viva.**

| Block | What it does |
|---|---|
| `conv_encoder()` | Image → numbers: two Conv2d layers (28 → 14 → 7 pixels), then dense layers down to 32 values. |
| `conv_decoder()` | Numbers → image: dense layers, then two ConvTranspose2d layers (7 → 14 → 28 pixels). |
| `VAE.__init__` | Latent size 16; builds the encoder and decoder. |
| `encode()` | Splits the encoder output into **μ** and **log σ²**. |
| `decode()` | Sigmoid turns the output into pixel values 0–1. |
| `loss()` | **Reparameterisation** `z = μ + σε`, then **reconstruction (BCE) + KL divergence**. |
| `sample()` | Decodes random z from N(0, 1) → new designs. |
| `reconstruct()` | Encode then decode (used for photo clean-up). |
| `interpolate()` | Decodes points between two codes → morph. |
| `embed()` | Returns μ as a garment "fingerprint" for find-similar. |
| `anomaly_score()`, `calibrate()` | Reconstruction error, and its normal range on real test clothes. |

---

## genmodels/data.py: Dataset

| Block | What it does |
|---|---|
| `loaders()` | Downloads Fashion-MNIST and serves it in shuffled batches of 128 (60,000 train, 10,000 test). |
| `pick_device()` | Uses a GPU if available, otherwise the CPU. |

## genmodels/common.py: Shared base class

| Block | What it does |
|---|---|
| `class GenModel` | `train_step()` does one learning step: loss → `backward()` (backpropagation) → Adam update. |
| `eval_mode` | Temporarily switches the model to evaluation mode. |

## genmodels/registry.py: Training settings

Stores the VAE's default settings: 20 epochs, learning rate 0.001, batch size 128.

## genmodels/train.py: Training

| Block | What it does |
|---|---|
| `train()` | Runs the epochs, prints the loss, tests on unseen images, saves `vae.pt` and `metrics.json`. |
| `save_figures()` | Saves new designs, reconstructions, morphs and the loss curve to `checkpoints/samples/`. |

## genmodels/evaluate.py: Measuring the VAE

| Block | What it does |
|---|---|
| `Classifier`, `get_classifier()` | A small CNN trained on Fashion-MNIST, used to get image features for FID. |
| `frechet()` | The FID formula comparing generated and real images. |
| `main()` | Prints the results table: loss, KL, MSE, FID, plus real-vs-real and noise reference values. |

## genmodels/smoke_test.py

A 10-second check that the VAE builds and runs every function. Should print `7/7 checks passed`.

---

## app.py: The web interface (Streamlit)

**Purpose:** the screen. It shows things and calls `core/` functions; it has no logic of its own.

| Block | What it does |
|---|---|
| Start-up | Page setup, `db.init()`, cleans missing photos, `@st.cache_resource` loads CLIP once, session state. |
| Sidebar | City box and status lights for the LLM and weather. |
| Wardrobe tab | Upload photos → tag → save; grid of clothes with an **Edit** panel to fix tags. |
| `outfit_card()` | Shows one outfit: photos, reason, "Why this score?", Like / Skip / Wear buttons. |
| Stylist tab | Occasion box and date → `stylist.recommend()` → 3 outfit cards. |
| Week plan tab | 7 days of outfits. |
| Design Studio tab | VAE features: generate, mix, clean up, find similar, photo check. |
| Insights tab | Garment count, outfits worn, **acceptance rate**, never-worn items, chart by type. |

---

## scripts/

| File | What it does |
|---|---|
| `load_samples.py` | Loads all photos from a folder into the wardrobe without the upload button. `--fresh` clears first. |
| `evaluate.py` | Tags labelled photos (from `sample_wardrobe/labels.csv` or `data/eval/`) and prints accuracy, mistakes and time per photo. |

## tests/test_recommender.py

Seven automatic tests of the outfit logic using fake clothes, e.g. "an interview gets formal clothes",
"no hoodie in the heat", "two patterns score lower". Run with `python -m pytest -q`.

---

## Other files

| File | What it is |
|---|---|
| `requirements.txt` | List of libraries to install. |
| `.env` | Your API keys (private). |
| `.gitignore` | Files Git should not upload (`.env`, `.venv`, `data/`). |
| `sample_wardrobe/` | 30 garment photos and `labels.csv` with their correct categories. |
| `checkpoints/` | Created by training: `vae.pt`, `metrics.json`, sample images. |
| `data/` | Created when running: the database, saved photos, Fashion-MNIST. |
