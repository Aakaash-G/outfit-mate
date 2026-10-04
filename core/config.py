"""Central configuration. Values come from the .env file in the project folder."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "outfit_mate.db"
for _d in (DATA_DIR, IMG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- LLM ------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

# ---- Weather --------------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Coimbatore").strip()

# ---- CLIP -----------------------------------------------------------------
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

# ---- Recommendation weights -----------
W_COMPAT, W_QUERY, W_STYLE, W_RECENCY = 0.45, 0.25, 0.20, 0.10