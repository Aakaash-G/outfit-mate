<<<<<<< HEAD
# outfit-mate
=======
# 👗 Outfit Mate — AI-Powered Personal Stylist

> **Case Study 2 · Semester 7 · Generative AI**
> *Team T20*

Outfit Mate is an intelligent wardrobe management and outfit recommendation system that combines **computer vision (CLIP)**, **large language models (Gemini / OpenAI)**, and a **Variational Autoencoder (VAE)** to act as your personal AI stylist. Upload photos of your clothes, and the app digitises, tags, and recommends complete outfits based on the weather, occasion, your colour preferences, and personal style.

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Architecture](#-architecture)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Setup & Installation](#-setup--installation)
6. [Running the App](#-running-the-app)
7. [Design Studio (VAE)](#-design-studio-vae)
8. [Evaluation](#-evaluation)
9. [Configuration](#-configuration)
10. [Testing](#-testing)
11. [Screenshots](#-screenshots)

---

## ✨ Features

### 1. Wardrobe Digitisation (Computer Vision)
- **Upload** clothing photos (JPG / PNG / WebP)
- **Background removal** using [rembg](https://github.com/danielgatis/rembg)
- **Zero-shot classification** via [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B-32)
  - Garment **category** (t-shirt, shirt, jeans, trousers, shorts, dress, sneakers, etc.)
  - **Pattern** detection (solid, striped, checked, floral, printed)
  - **Formality** level (1–5 scale: lounge → very formal)
- **Dominant colour** extraction using mini k-means on foreground pixels
- Editable tags — fix any mistakes the AI makes

### 2. Smart Outfit Recommendations (LLM + Retrieval)
- **Weather-aware**: fetches real-time forecasts from OpenWeatherMap
- **Intent parsing**: LLM extracts occasion, formality, colour preferences from natural language ("Placement interview tomorrow, no black please")
- **Outfit scoring** with four weighted factors:
  - Compatibility (rule-based + CLIP embedding cohesion) — 45%
  - Query match (CLIP similarity to occasion text) — 25%
  - Personal style (learned from feedback) — 20%
  - Recency penalty (avoids repeats) — 10%
- **LLM explanation**: Gemini or OpenAI picks the best outfits and writes friendly reasons
- **Rule-based fallback**: works offline without any API keys

### 3. Week Planner
- Plans 7 days of outfits without repeating tops/bottoms within 2 days
- Adapts to each day's forecast
- Uses the same scoring + diversity system

### 4. Design Studio (Variational Autoencoder)
- VAE trained on **Fashion-MNIST** (70,000 images)
- **Generate new designs** from random latent codes z ~ N(0, I)
- **Design mixer**: interpolate between two garments in latent space
- **Photo clean-up**: add noise → let the VAE rebuild from latent code
- **Find similar**: cosine similarity on VAE latent embeddings (mu)
- **Upload quality check**: reconstruction error z-score flags unusual photos

### 5. Insights Dashboard
- Garment counts by type (top / bottom / shoes / layer / onepiece)
- Outfit acceptance rate (likes + worn vs. skips)
- "Forgotten clothes" — items you haven't worn recently

### 6. Learning Style Profile
- **Like / Skip / Wear** feedback on every recommended outfit
- Maintains a CLIP-space style vector that nudges future recommendations
- EMA update: `new = 0.85 × old + 0.15 × outfit_embedding`

---

## 🏗 Architecture

```
User Request ("interview tomorrow, no black")
        │
        ▼
┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Weather API │───▶│ LLM Intent  │───▶│  Retriever   │
│ (OpenWeather)│    │   Parser    │    │  & Scorer    │
└──────────────┘    │ (Gemini /   │    │ (CLIP + rules│
                    │  rule-based)│    │  + style vec)│
                    └─────────────┘    └──────┬───────┘
                                              │ top-10 candidates
                                              ▼
                                    ┌─────────────────┐
                                    │  LLM Chooser &  │
                                    │  Explainer      │
                                    │ (pick 3 + reason)│
                                    └────────┬────────┘
                                             │
                                             ▼
                                     Outfit Cards in UI
```

### Data Flow

1. **Digitise**: Photo → rembg → white background → CLIP embedding → zero-shot tags → SQLite
2. **Recommend**: Weather + intent → filter → assemble (top + bottom + shoes ± layer) → score → LLM explain
3. **Learn**: User feedback (like/skip/worn) → update style vector → better future recommendations
4. **Design Studio**: Garment photo → Fashion-MNIST format → VAE encode/decode → latent features

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) ≥ 1.40 |
| Vision Model | [OpenCLIP](https://github.com/mlfoundations/open_clip) ViT-B-32 (LAION-2B) |
| LLM | Google Gemini 2.5 Flash / OpenAI-compatible API |
| Generative Model | Custom VAE (PyTorch) on Fashion-MNIST |
| Background Removal | [rembg](https://github.com/danielgatis/rembg) (U2-Net) |
| Weather | [OpenWeatherMap](https://openweathermap.org/) Forecast API |
| Database | SQLite (zero-config, single file) |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
outfit_mate/
├── app.py                      # Streamlit app (main entry point)
├── core/                       # Application logic
│   ├── config.py               # Central configuration (.env loading)
│   ├── db.py                   # SQLite storage (garments, outfits, feedback, style)
│   ├── vision.py               # CLIP wrapper, zero-shot tagging, colour extraction
│   ├── llm.py                  # LLM intent parsing & outfit explanation
│   ├── recommender.py          # Outfit assembly, compatibility scoring, ranking
│   ├── stylist.py              # Orchestrator: weather → intent → score → explain
│   ├── weather.py              # OpenWeatherMap client with safe defaults
│   └── genstudio.py            # Bridge to the trained VAE
├── genmodels/                  # Generative model code
│   ├── vae.py                  # VAE architecture (conv encoder/decoder)
│   ├── train.py                # Training script (Fashion-MNIST)
│   ├── evaluate.py             # FID evaluation
│   ├── common.py               # Shared model interface
│   ├── data.py                 # Fashion-MNIST data loading
│   ├── registry.py             # Model registry
│   └── smoke_test.py           # Quick sanity check
├── scripts/
│   ├── bulk_add_wardrobe.py    # Batch-import images from img/ to DB
│   ├── evaluate.py             # Garment tagging accuracy evaluation
│   └── setup_eval_data.py      # Download eval images from Unsplash
├── tests/
│   └── test_recommender.py     # Unit tests (fake CLIP, no download)
├── img/                        # Uploaded wardrobe images (by category)
│   ├── casual-jeans/           # 5 images
│   ├── casual-shirts/          # 5 images
│   ├── formal-pants/           # 5 images
│   ├── formal-shirts/          # 5 images
│   ├── gym/                    # 4 images
│   └── shorts/                 # 5 images
├── data/
│   ├── outfit_mate.db          # SQLite database
│   ├── images/                 # Processed garment images (background removed)
│   ├── eval/                   # Evaluation images (15 categories, 39 images)
│   └── fashion_mnist/          # Fashion-MNIST dataset (auto-downloaded)
├── checkpoints/
│   ├── vae.pt                  # Trained VAE weights
│   ├── classifier.pt           # FID feature extractor
│   ├── metrics.json            # Evaluation metrics
│   └── samples/                # Generated sample images
├── static/images/              # Placeholder images for reference
├── download_images.py          # Download wardrobe images from Unsplash
├── generate_sample_images.py   # Generate placeholder clothing images
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
└── .gitignore
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or later
- pip (Python package manager)

### 1. Clone / extract the project

```bash
cd outfit_mate
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The first run downloads the CLIP model (~600 MB) and Fashion-MNIST (~30 MB) automatically.

### 4. Configure API keys

Edit the `.env` file in the project root:

```env
LLM_PROVIDER=gemini
LLM_API_KEY=your-gemini-api-key-here
LLM_MODEL=gemini-2.5-flash
OPENWEATHER_API_KEY=your-openweather-key
DEFAULT_CITY=Coimbatore
```

| Key | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | No | `gemini`, `openai`, or `none` (default: `none`) |
| `LLM_API_KEY` | No | API key for the chosen LLM provider |
| `LLM_MODEL` | No | Model name (default: `gemini-2.5-flash`) |
| `OPENWEATHER_API_KEY` | No | Free tier key from openweathermap.org |
| `DEFAULT_CITY` | No | City for weather forecasts (default: `Coimbatore`) |

> The app works without any API keys — it falls back to rule-based intent parsing, template-based outfit explanations, and default weather values (28 °C, 20% rain).

### 5. Download wardrobe images

```bash
python download_images.py
```

### 6. Bulk-add images to the wardrobe

```bash
python scripts/bulk_add_wardrobe.py
```

This processes each image through the CLIP pipeline (background removal → embedding → zero-shot tagging) and stores them in the SQLite database.

---

## ▶ Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with five tabs:

| Tab | Purpose |
|---|---|
| 👕 **Wardrobe** | Upload, view, edit, delete garments |
| 💬 **Stylist** | Ask for outfit recommendations by occasion |
| 📅 **Week Plan** | Auto-plan 7 days of non-repeating outfits |
| 🎨 **Design Studio** | VAE-powered generative features |
| 📊 **Insights** | Wardrobe analytics and forgotten clothes |

---

## 🎨 Design Studio (VAE)

### Training the VAE

The VAE is trained on Fashion-MNIST (70,000 greyscale 28×28 clothing images):

```bash
# Full training (20 epochs, ~10-20 min on CPU, ~3 min on GPU)
python -m genmodels.train

# Quick smoke test (1 epoch, 30 batches, ~10 seconds)
python -m genmodels.train --quick

# Custom options
python -m genmodels.train --epochs 5 --z-dim 16 --device auto
```

### Outputs

| File | Description |
|---|---|
| `checkpoints/vae.pt` | Trained weights + calibration data |
| `checkpoints/samples/vae_samples.png` | 64 randomly generated designs |
| `checkpoints/samples/vae_recon.png` | Test images vs. VAE reconstructions |
| `checkpoints/samples/vae_interp.png` | Latent space interpolations |
| `checkpoints/samples/vae_loss.png` | Training loss curve |
| `checkpoints/metrics.json` | Test metrics (ELBO, MSE, KL, FID) |

### VAE Metrics

| Metric | Value |
|---|---|
| Architecture | Conv encoder (1→32→64→256→32) / decoder (16→256→64→32→1) |
| Parameters | 1.69 M |
| Latent dimension | 16 |
| Test -ELBO | 237.77 nats/image |
| Reconstruction MSE | 0.0143 |
| KL divergence | 14.52 nats |
| FID (new samples) | 71.39 |
| FID (reconstructions) | 30.44 |
| Training time | ~7 minutes (CPU) |

### Evaluating the VAE (FID score)

```bash
python -m genmodels.evaluate
```

This computes FID (Fréchet Inception Distance) using a Fashion-MNIST classifier as the feature extractor.

---

## 📊 Evaluation

### Garment Tagging Accuracy

To evaluate CLIP's zero-shot classification on real clothing photos:

#### 1. Download evaluation images

```bash
python scripts/setup_eval_data.py
```

Downloads 39 real clothing photos from Unsplash across 15 categories into `data/eval/`.

#### 2. Run the evaluation

```bash
python scripts/evaluate.py
```

Reports:
- **Category accuracy**: How often CLIP correctly identifies the garment type
- **Slot accuracy**: How often the garment lands in the right slot (top/bottom/shoes/layer/onepiece)
- **Per-class breakdown**: Accuracy for each category
- **Confusion matrix**: Most common misclassifications
- **Speed**: Milliseconds per image (including background removal)

### Unit Tests

```bash
python -m pytest -q
```

Tests cover:
- Formal requests get formal items
- No hoodies in hot weather (≥24°C)
- Colour avoidance works
- Two-pattern outfits are penalised
- Recently-worn items are ranked lower
- Rule-based intent parsing (formality levels, colour preferences)
- Outfit selection without LLM returns valid, non-duplicate results

---

## ⚙ Configuration

### Recommendation Weights

Defined in `core/config.py`:

```python
W_COMPAT  = 0.45   # Outfit compatibility (patterns, colours, formality spread)
W_QUERY   = 0.25   # Match to the user's occasion/request
W_STYLE   = 0.20   # Personal style (learned from feedback)
W_RECENCY = 0.10   # Penalty for recently-worn items
```

### CLIP Model

```python
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
```

### Garment Categories

20 categories across 5 slots:

| Slot | Categories |
|---|---|
| **Top** | t-shirt, shirt, polo shirt, blouse, kurta, crop top |
| **Bottom** | jeans, trousers, shorts, skirt, track pants |
| **Onepiece** | dress, saree |
| **Layer** | sweater, hoodie, jacket, blazer |
| **Shoes** | sneakers, formal shoes, sandals, boots |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest -q

# Run with verbose output
python -m pytest -v

# Run VAE smoke test (no trained model needed)
python -m genmodels.smoke_test
```

---

## 📸 Screenshots

After launching the app with `streamlit run app.py`, you'll see:

1. **Wardrobe Tab**: Grid of digitised garments with editable tags
2. **Stylist Tab**: Natural language input → weather-aware outfit cards with reasons
3. **Week Planner**: 7-day outfit calendar adapted to daily weather
4. **Design Studio**: VAE-generated designs, design mixer, similarity search
5. **Insights**: Wardrobe analytics, wear counts, forgotten clothes

---

## 📝 Key Design Decisions

1. **CLIP for zero-shot tagging** — No labelled training data needed; works on any garment type
2. **Rule-based fallback** — The app is fully functional without any API keys
3. **SQLite** — Zero-config database, no server, perfect for a personal wardrobe
4. **Fashion-MNIST VAE** — Lightweight generative model that runs on any laptop CPU
5. **Modular architecture** — Each module (vision, llm, recommender, stylist, weather) is independent and testable
6. **Style learning via EMA** — Simple but effective: like/skip feedback moves the style vector in CLIP space

---

## 📄 License

This project is part of academic coursework (Semester 7, Generative AI Case Study).

---

*Built with ❤️ by Team T20*
>>>>>>> e328f5a (Initial commit)
