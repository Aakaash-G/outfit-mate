"""Evaluate the trained VAE: test metrics + a Fashion-MNIST FID score for its generated designs.

FID (Frechet Inception Distance) normally uses an ImageNet network; for 28x28 clothes we use the
same formula on features of a small CNN classifier trained on Fashion-MNIST. Lower = generated
designs look statistically more like real clothes. As reference points the script also reports
the FID of (a) VAE reconstructions and (b) pure noise.

Run after training:   python -m genmodels.evaluate            (1000 samples)
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CKPT_DIR, loaders, pick_device
from .registry import REGISTRY


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


def get_classifier(device, epochs=3):
    path = CKPT_DIR / "classifier.pt"
    clf = Classifier().to(device)
    if path.exists():
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()
    print("Training the feature classifier for FID (one-off, ~1-3 min)...")
    tr, te = loaders(128)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in tr:
            loss = F.cross_entropy(clf(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in te:
            correct += (clf(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += y.numel()
    print(f"  classifier test accuracy: {correct / total:.1%}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), path)
    return clf


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(s1)
    s1_half = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    eig = np.linalg.eigvalsh(s1_half @ s2 @ s1_half)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.clip(eig, 0, None)).sum())


@torch.no_grad()
def feats(clf, x, chunk=500):
    return torch.cat([clf.features(x[i:i + chunk]) for i in range(0, len(x), chunk)]).cpu().numpy()


def load_vae(device):
    ck = torch.load(CKPT_DIR / "vae.pt", map_location=device, weights_only=False)
    model = REGISTRY["vae"]["cls"](z_dim=ck.get("z_dim", 16)).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of samples for FID")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = pick_device(a.device)
    if not (CKPT_DIR / "vae.pt").exists():
        raise SystemExit("Train first: python -m genmodels.train")
    model, ck = load_vae(dev)
    clf = get_classifier(dev)

    _, te = loaders(500)
    batches = [x for _, (x, _) in zip(range(2 * ((a.n + 499) // 500)), te)]
    real = torch.cat(batches)[:a.n].to(dev)
    other = torch.cat(batches)[a.n:2 * a.n].to(dev)  # a second real set: the "best possible" FID
    fr = feats(clf, real)

    samples = torch.cat([model.sample(min(500, a.n - i)) for i in range(0, a.n, 500)])
    results = {
        "FID real vs real (floor)": frechet(feats(clf, other), fr) if len(other) > 1 else float("nan"),
        "FID VAE reconstructions": frechet(feats(clf, model.reconstruct(real)), fr),
        "FID VAE new samples": frechet(feats(clf, samples), fr),
        "FID random noise (ceiling)": frechet(feats(clf, torch.rand_like(real)), fr),
    }

    t = ck["test"]
    print("\n| Metric | Value |\n|---|---|")
    print(f"| Test -ELBO (nats per image) | {t['neg_elbo']:.2f} |")
    print(f"| Reconstruction term (nats) | {t['recon_bce']:.2f} |")
    print(f"| KL term (nats) | {t['kl']:.2f} |")
    print(f"| Reconstruction MSE | {t['recon_mse']:.4f} |")
    for k, v in results.items():
        print(f"| {k} | {v:.2f} |")
    print(f"| Parameters | {ck['params'] / 1e6:.2f} M |")

    mp = CKPT_DIR / "metrics.json"
    m = json.loads(mp.read_text()) if mp.exists() else {}
    m.setdefault("vae", {})["fid"] = round(results["FID VAE new samples"], 2)
    m["vae"]["fid_reconstructions"] = round(results["FID VAE reconstructions"], 2)
    mp.write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()