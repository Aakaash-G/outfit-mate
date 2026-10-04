"""Garment understanding: background removal, CLIP embeddings, zero-shot tagging, colour extraction."""
import uuid

import numpy as np
from PIL import Image

from .config import CLIP_MODEL, CLIP_PRETRAINED, IMG_DIR

# category -> (slot, default formality, warmth)
CATEGORIES = {
    "t-shirt": ("top", 2, 1), "shirt": ("top", 3, 1), "polo shirt": ("top", 2, 1),
    "blouse": ("top", 3, 1), "kurta": ("top", 4, 1), "crop top": ("top", 2, 1),
    "sweater": ("layer", 2, 3), "hoodie": ("layer", 1, 3), "jacket": ("layer", 2, 3),
    "blazer": ("layer", 4, 2),
    "jeans": ("bottom", 2, 2), "trousers": ("bottom", 3, 2), "shorts": ("bottom", 1, 1),
    "skirt": ("bottom", 3, 1), "track pants": ("bottom", 1, 1),
    "dress": ("onepiece", 3, 1), "saree": ("onepiece", 5, 1),
    "sneakers": ("shoes", 2, 1), "formal shoes": ("shoes", 4, 1), "sandals": ("shoes", 1, 1),
    "boots": ("shoes", 2, 2),
}
PATTERNS = ["solid", "striped", "checked", "floral", "printed"]
PATTERN_PROMPTS = {
    "solid": "a plain solid-colour garment", "striped": "a striped garment",
    "checked": "a checked plaid garment", "floral": "a garment with a floral pattern",
    "printed": "a garment with a graphic print or logo",
}
FORMALITY_PROMPTS = [
    "loungewear or gym clothing", "casual everyday clothing", "smart casual clothing",
    "formal office clothing", "very formal party, wedding or ethnic festive clothing",
]
FORMALITY_WORDS = {1: "relaxed", 2: "casual", 3: "smart-casual", 4: "formal", 5: "very formal / festive"}

# named colours for nearest-colour lookup
PALETTE = {
    "black": (25, 25, 25), "white": (240, 240, 240), "grey": (128, 128, 128), "navy": (30, 40, 80),
    "blue": (40, 90, 180), "light blue": (150, 190, 230), "denim": (80, 105, 140), "red": (190, 30, 40),
    "maroon": (110, 25, 35), "pink": (230, 150, 170), "orange": (230, 120, 40), "yellow": (235, 200, 50),
    "beige": (215, 195, 160), "brown": (110, 70, 40), "olive": (110, 110, 50), "green": (40, 130, 70),
    "purple": (110, 60, 140),
}
NEUTRALS = {"black", "white", "grey", "navy", "beige", "brown", "denim"}


# ---------------------------------------------------------------- CLIP wrapper
class Clip:
    """Thin wrapper around open_clip returning L2-normalised numpy vectors."""

    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED)
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model.eval()
        self._text_cache = {}

    def image(self, img: Image.Image) -> np.ndarray:
        with self.torch.no_grad():
            x = self.preprocess(img.convert("RGB")).unsqueeze(0)
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy().astype(np.float32)

    def text(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            with self.torch.no_grad():
                f = self.model.encode_text(self.tokenizer(missing))
                f = f / f.norm(dim=-1, keepdim=True)
            for t, v in zip(missing, f.cpu().numpy().astype(np.float32)):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts])


def get_clip() -> Clip:
    return Clip()


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def zero_shot(clip: Clip, img_emb: np.ndarray, prompts: list[str]) -> np.ndarray:
    """Probabilities of the image matching each prompt."""
    t = clip.text(prompts)
    return _softmax(100.0 * t @ img_emb)


# ---------------------------------------------------------------- image helpers
def remove_background(img: Image.Image) -> Image.Image:
    """Returns RGBA with transparent background. Falls back to the original if rembg is missing."""
    try:
        from rembg import remove
        return remove(img.convert("RGB")).convert("RGBA")
    except Exception:
        return img.convert("RGBA")


def on_white(img_rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
    bg.paste(img_rgba, mask=img_rgba.split()[-1])
    return bg


def _nearest_colour(rgb) -> str:
    return min(PALETTE, key=lambda n: sum((a - b) ** 2 for a, b in zip(PALETTE[n], rgb)))


def dominant_colours(img_rgba: Image.Image, k: int = 3) -> list[str]:
    arr = np.asarray(img_rgba.convert("RGBA").resize((96, 96)))
    rgb = arr[..., :3].reshape(-1, 3).astype(float)
    alpha = arr[..., 3].reshape(-1)
    px = rgb[alpha > 128]
    if (alpha < 128).mean() < 0.02 or len(px) < 50:  # no usable mask: drop near-white background
        px = rgb[~(rgb > 225).all(axis=1)]
    if len(px) < k:
        px = rgb
    rng = np.random.default_rng(0)
    centres = px[rng.choice(len(px), k, replace=False)]
    for _ in range(12):  # tiny k-means
        lab = ((px[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array([px[lab == i].mean(0) if (lab == i).any() else centres[i] for i in range(k)])
    shares = np.bincount(lab, minlength=k) / len(px)
    names = []
    for i in shares.argsort()[::-1]:
        if shares[i] < 0.15:
            continue
        n = _nearest_colour(centres[i])
        if n not in names:
            names.append(n)
    return names[:2] or [_nearest_colour(centres[shares.argmax()])]


def save_image(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((768, 768))
    path = IMG_DIR / f"{uuid.uuid4().hex}.png"
    img.save(path)
    return str(path)


def make_collage(paths: list[str], height: int = 256) -> Image.Image:
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(im.resize((max(1, int(im.width * height / im.height)), height)))
    out = Image.new("RGB", (sum(i.width for i in ims) + 10 * (len(ims) + 1), height + 20), "white")
    x = 10
    for im in ims:
        out.paste(im, (x, 10))
        x += im.width + 10
    return out


# ---------------------------------------------------------------- main entry point
def tag_image(img: Image.Image, clip: Clip) -> dict:
    """Full wardrobe-digitisation step for one photo."""
    cut = remove_background(img)
    clean = on_white(cut)
    emb = clip.image(clean)

    cats = list(CATEGORIES)
    p_cat = zero_shot(clip, emb, [f"a photo of a {c}" for c in cats])
    category = cats[int(p_cat.argmax())]
    slot, prior_formality, warmth = CATEGORIES[category]

    p_pat = zero_shot(clip, emb, list(PATTERN_PROMPTS.values()))
    pattern = PATTERNS[int(p_pat.argmax())]

    p_form = zero_shot(clip, emb, [f"a photo of {f}" for f in FORMALITY_PROMPTS])
    clip_formality = float((p_form * np.arange(1, 6)).sum())
    formality = int(round(0.5 * prior_formality + 0.5 * clip_formality))

    colours = dominant_colours(cut)
    name = " ".join([colours[0]] + ([pattern] if pattern != "solid" else []) + [category])
    return {
        "name": name, "category": category, "slot": slot, "colors": colours, "pattern": pattern,
        "formality": max(1, min(5, formality)), "warmth": warmth,
        "confidence": float(p_cat.max()), "embedding": emb, "clean_image": clean,
    }


def describe(g: dict) -> str:
    pat = "" if g["pattern"] == "solid" else f"{g['pattern']} "
    return f"{' & '.join(g['colors'])} {pat}{g['category']}"