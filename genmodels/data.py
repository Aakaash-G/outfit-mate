"""Fashion-MNIST loading (downloaded automatically the first time, about 30 MB)."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent / "data" / "fashion_mnist"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def loaders(batch=128, workers=0):
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()  # -> (1, 28, 28) in [0, 1]
    train = datasets.FashionMNIST(ROOT, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(ROOT, train=False, download=True, transform=tf)
    pin = torch.cuda.is_available()
    return (DataLoader(train, batch, shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin),
            DataLoader(test, batch, shuffle=False, num_workers=workers, pin_memory=pin))


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")