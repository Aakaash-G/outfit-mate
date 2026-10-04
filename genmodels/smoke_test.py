"""10-second check that the VAE builds, trains a step and runs every app feature (random data, CPU).

Run this first:   python -m genmodels.smoke_test
"""
import torch

from .registry import REGISTRY


def main():
    torch.manual_seed(0)
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    spec = REGISTRY["vae"]
    m = spec["cls"]()
    m.setup_optim(spec["lr"])
    checks = {
        "train step": lambda: m.train_step(x, y),
        "evaluate": lambda: m.evaluate(x, y),
        "sample": lambda: m.sample(4).shape == (4, 1, 28, 28),
        "reconstruct": lambda: m.reconstruct(x).shape == x.shape,
        "interpolate": lambda: m.interpolate(x[:1], x[1:2], 6).shape == (6, 1, 28, 28),
        "embed": lambda: m.embed(x).shape == (8, m.z_dim),
        "anomaly score": lambda: m.anomaly_score(x).shape == (8,),
    }
    ok = 0
    for name, fn in checks.items():
        try:
            r = fn()
            assert r is not False, "wrong output shape"
            print(f"PASS {name}")
            ok += 1
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(checks)} checks passed ({sum(p.numel() for p in m.parameters()) / 1e6:.2f} M parameters)")


if __name__ == "__main__":
    main()