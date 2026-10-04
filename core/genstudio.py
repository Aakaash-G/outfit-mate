"""Bridge between the app and the trained VAE (genmodels/vae.py).

Fails soft: if the VAE hasn't been trained yet, the Design Studio simply shows how to train it.
Garment photos are converted to the Fashion-MNIST format (grey, inverted, 28x28, centred)
so the VAE can work on the user's own clothes.
"""
import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

from .config import BASE_DIR

CKPT_DIR = BASE_DIR / "checkpoints"
VAE_PATH = CKPT_DIR / "vae.pt"


def trained() -> bool:
    return VAE_PATH.exists()


def metrics() -> dict:
    p = CKPT_DIR / "metrics.json"
    return json.loads(p.read_text()).get("vae", {}) if p.exists() else {}


@lru_cache(maxsize=1)
def vae():
    import torch
    from genmodels.vae import VAE
    ck = torch.load(VAE_PATH, map_location="cpu", weights_only=False)
    model = VAE(z_dim=ck.get("z_dim", 16))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model.calibration = ck.get("calibration", {})
    return model


# ---------------------------------------------------------------- image conversion
def to_fmnist(path: str):
    """Garment photo on white background -> tensor (1, 1, 28, 28) in the Fashion-MNIST style."""
    import torch
    g = ImageOps.invert(Image.open(path).convert("L"))       # white background -> black
    arr = np.asarray(g)
    ys, xs = np.where(arr > 20)
    if len(xs):
        g = g.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = max(g.size)
    sq = Image.new("L", (side, side), 0)
    sq.paste(g, ((side - g.width) // 2, (side - g.height) // 2))
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(sq.resize((24, 24), Image.LANCZOS), (2, 2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)[None, None]


def to_pil(batch, nrow=8, scale=4) -> Image.Image:
    """Tensor (N, 1, 28, 28) in [0, 1] -> upscaled grid image."""
    from torchvision.utils import make_grid
    grid = make_grid(batch.detach().cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=1.0)
    img = Image.fromarray((grid[0].numpy() * 255).astype(np.uint8))
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


# ---------------------------------------------------------------- features
def generate(n=16, temperature=1.0):
    return vae().sample(n, temperature=temperature)


def interpolate(path_a, path_b, steps=10):
    return vae().interpolate(to_fmnist(path_a), to_fmnist(path_b), steps)


def clean_up(path, noise=0.3):
    """Add noise to a garment, then let the VAE rebuild it: original | noisy | VAE output."""
    import torch
    x = to_fmnist(path)
    g = torch.Generator().manual_seed(0)
    noisy = (x + noise * torch.randn(x.shape, generator=g)).clamp(0, 1)
    return torch.cat([x, noisy, vae().reconstruct(noisy)])


def similar(target, garments, k=4):
    """Garments whose VAE latent codes (mu) are closest by cosine similarity."""
    import torch
    others = [g for g in garments if g["id"] != target["id"]]
    if not others:
        return []
    x = torch.cat([to_fmnist(g["image_path"]) for g in [target] + others])
    z = vae().embed(x).numpy()
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sims = z[1:] @ z[0]
    return [(others[i], float(sims[i])) for i in np.argsort(-sims)[:k]]


def photo_check(path) -> dict | None:
    """How unusual is this upload for the VAE? z-score of reconstruction error vs real test clothes."""
    if not trained():
        return None
    m = vae()
    cal = m.calibration or {}
    score = float(m.anomaly_score(to_fmnist(path))[0])
    z = (score - cal["score_mean"]) / cal["score_std"] if cal.get("score_std") else 0.0
    return {"recon_error": score, "z_score": z, "unusual": z > 4}