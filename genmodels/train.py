"""Train the Outfit Mate VAE on Fashion-MNIST.

Examples
    python -m genmodels.train                      # default: 20 epochs
    python -m genmodels.train --epochs 5           # faster
    python -m genmodels.train --quick              # 1 short epoch, just to check it works

Outputs (in checkpoints/):
    vae.pt                     weights + test metrics
    samples/vae_samples.png    64 brand-new designs from random latent codes
    samples/vae_recon.png      top 4 rows = real test images, bottom 4 rows = VAE reconstructions
    samples/vae_interp.png     each row morphs one garment into another
    samples/vae_loss.png       training curve (-ELBO, reconstruction, KL per epoch)
    metrics.json               test metrics (used by the app and the report)
"""
import argparse
import json
import time

import torch
from torchvision.utils import save_image

from .data import CKPT_DIR, loaders, pick_device
from .registry import REGISTRY


def _mean(dicts):
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0]}


def evaluate(model, loader, device, limit=None):
    model.eval()
    out = [model.evaluate(x.to(device), y.to(device)) for i, (x, y) in enumerate(loader) if not limit or i < limit]
    model.train()
    return _mean(out)


def save_figures(model, test_loader, device, history):
    out = CKPT_DIR / "samples"
    out.mkdir(parents=True, exist_ok=True)
    x, y = next(iter(test_loader))
    x = x.to(device)
    save_image(model.sample(64), out / "vae_samples.png", nrow=8, pad_value=1)
    save_image(torch.cat([x[:32], model.reconstruct(x[:32])]), out / "vae_recon.png", nrow=8, pad_value=1)
    # morph between pairs of different garment types
    rows = []
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
        ia, ib = (y == a).nonzero(), (y == b).nonzero()
        if len(ia) and len(ib):
            ia, ib = ia[0].item(), ib[0].item()
            rows.append(model.interpolate(x[ia:ia + 1], x[ib:ib + 1], 10))
    if rows:
        save_image(torch.cat(rows), out / "vae_interp.png", nrow=10, pad_value=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = range(1, len(history) + 1)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for k, lab in [("neg_elbo", "-ELBO (loss)"), ("recon_bce", "reconstruction"), ("kl", "KL")]:
            ax.plot(ep, [h[k] for h in history], marker="o", label=lab)
        ax.set_xlabel("epoch")
        ax.set_ylabel("nats per image")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "vae_loss.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def train(epochs=None, batch=None, lr=None, device="auto", limit=None, workers=0, z_dim=16):
    spec = REGISTRY["vae"]
    dev = pick_device(device)
    epochs = epochs or spec["epochs"]
    torch.manual_seed(0)
    model = spec["cls"](z_dim=z_dim).to(dev)
    model.setup_optim(lr or spec["lr"])
    train_loader, test_loader = loaders(batch or spec["batch"], workers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== VAE: {n_params / 1e6:.2f} M parameters, latent size {z_dim}, {epochs} epochs, device {dev} ===")

    start, history = time.time(), []
    for ep in range(epochs):
        t, logs = time.time(), []
        for i, (x, y) in enumerate(train_loader):
            if limit and i >= limit:
                break
            logs.append(model.train_step(x.to(dev), y.to(dev)))
        history.append(_mean(logs))
        print(f"epoch {ep + 1:>2}/{epochs}  {time.time() - t:5.0f}s  "
              + "  ".join(f"{k}={v:.3f}" for k, v in history[-1].items()), flush=True)

    test = evaluate(model, test_loader, dev, limit)
    calib = model.calibrate(test_loader, dev)
    print("test: " + "  ".join(f"{k}={v:.4f}" for k, v in test.items()))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"key": "vae", "state_dict": model.state_dict(), "test": test, "calibration": calib,
                "epochs": epochs, "params": n_params, "z_dim": z_dim, "history": history}, CKPT_DIR / "vae.pt")
    save_figures(model, test_loader, dev, history)
    (CKPT_DIR / "metrics.json").write_text(json.dumps({"vae": {
        "display": model.display, "params_M": round(n_params / 1e6, 2), "epochs": epochs, "z_dim": z_dim,
        "train_minutes": round((time.time() - start) / 60, 1), "device": str(dev), "quick_run": bool(limit),
        **{k: round(v, 4) for k, v in test.items()}}}, indent=2))
    print(f"saved checkpoints/vae.pt and figures in checkpoints/samples/  ({(time.time() - start) / 60:.1f} min)")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--z-dim", type=int, default=16, help="latent size (try 2 to plot the latent space)")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--quick", action="store_true", help="1 epoch of 30 batches (smoke run)")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    train(epochs=1 if a.quick else a.epochs, batch=a.batch, lr=a.lr, device=a.device,
          limit=30 if a.quick else None, workers=a.workers, z_dim=a.z_dim)


if __name__ == "__main__":
    main()