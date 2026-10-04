"""Shared helpers for all generative models.

Every model follows the same small interface so one training script works for all 13:
    model.setup_optim(lr)        create optimiser(s)
    model.train_step(x, y)       one update on a batch, returns a dict of metrics
    model.evaluate(x, y)         metrics on a batch without updating
    model.sample(n, y=None)      new images, shape (n, 1, 28, 28), values in [0, 1]
x is a batch of Fashion-MNIST images in [0, 1], shape (N, 1, 28, 28); y are class labels 0-9.
"""
import math

import torch
import torch.nn as nn

D = 28 * 28
LN2 = math.log(2)
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def binarize(x):
    """Discrete autoregressive models work on black/white pixels."""
    return (x > 0.5).float()


class GenModel(nn.Module):
    key = ""
    display = ""
    family = ""
    conditional = False  # can it generate a chosen class (e.g. "Dress")?
    can_sample = True

    def setup_optim(self, lr):
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def loss(self, x, y):
        raise NotImplementedError

    def train_step(self, x, y):
        loss, metrics = self.loss(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        self.opt.step()
        return metrics

    @torch.no_grad()
    def evaluate(self, x, y):
        return self.loss(x, y)[1]

    def sample(self, n, y=None):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    def labels(self, n, y=None):
        """Class labels for sampling: None -> random classes, int -> all that class."""
        if y is None:
            return torch.randint(0, 10, (n,), device=self.device)
        if isinstance(y, int):
            return torch.full((n,), y, dtype=torch.long, device=self.device)
        return y.to(self.device)


def nll_metrics(nll_per_image):
    m = nll_per_image.mean()
    return m, {"nll_nats": m.item(), "bits_per_dim": m.item() / (D * LN2)}


class eval_mode:
    """Context manager: temporarily switch a model to eval mode (dropout / batch-norm)."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        self.model.eval()

    def __exit__(self, *exc):
        self.model.train(self.was_training)