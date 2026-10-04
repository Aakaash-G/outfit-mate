"""Variational Autoencoder (Kingma & Welling, 2014) for Outfit Mate.

The encoder squeezes a 28x28 garment image into a small latent code z (16 numbers).
Instead of one fixed code it outputs a Gaussian q(z|x) = N(mu, sigma^2), and the decoder
rebuilds the image from a sample of it. Training maximises the ELBO:

    loss = reconstruction error  +  KL( N(mu, sigma^2) || N(0, I) )

The KL term keeps the latent space smooth and centred, which is what makes the VAE useful
for several app features at once:
    sample()         new garment designs, decoded from random z ~ N(0, I)
    interpolate()    design mixer: morph garment A into garment B
    embed()          latent mean mu as a garment "fingerprint" for find-similar
    reconstruct()    clean-up: rebuild a noisy or blurry photo from its code
    anomaly_score()  reconstruction error: flags uploads that don't look like clothing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import GenModel, eval_mode


def conv_encoder(out_dim):
    return nn.Sequential(
        nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),            # 28 -> 14
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),           # 14 -> 7
        nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Linear(256, out_dim))


def conv_decoder(z_dim):
    return nn.Sequential(
        nn.Linear(z_dim, 256), nn.ReLU(), nn.Linear(256, 64 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 7 -> 14
        nn.ConvTranspose2d(32, 1, 4, 2, 1))              # 14 -> 28 (logits)


class VAE(GenModel):
    key, display, family = "vae", "Variational Autoencoder (VAE)", "Autoencoder"

    def __init__(self, z_dim=16, beta=1.0):
        super().__init__()
        self.z_dim, self.beta = z_dim, beta
        self.enc = conv_encoder(2 * z_dim)
        self.dec = conv_decoder(z_dim)

    # ---------------------------------------------------------------- core
    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, 1)
        return mu, logvar.clamp(-10, 10)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def loss(self, x, y=None):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)       # reparameterisation trick
        logits = self.dec(z)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
        loss = (recon + self.beta * kl).mean()
        mse = ((torch.sigmoid(logits) - x) ** 2).flatten(1).mean(1)
        return loss, {"neg_elbo": (recon + kl).mean().item(), "recon_bce": recon.mean().item(),
                      "kl": kl.mean().item(), "recon_mse": mse.mean().item()}

    # ---------------------------------------------------------------- app features
    @torch.no_grad()
    def sample(self, n, y=None, temperature=1.0):
        with eval_mode(self):
            return self.decode(torch.randn(n, self.z_dim, device=self.device) * temperature)

    @torch.no_grad()
    def reconstruct(self, x):
        with eval_mode(self):
            return self.decode(self.encode(x)[0])

    @torch.no_grad()
    def interpolate(self, xa, xb, steps=8):
        with eval_mode(self):
            za, zb = self.encode(xa)[0], self.encode(xb)[0]
            t = torch.linspace(0, 1, steps, device=self.device)[:, None]
            return self.decode((1 - t) * za + t * zb)

    @torch.no_grad()
    def embed(self, x):
        with eval_mode(self):
            return self.encode(x)[0]

    @torch.no_grad()
    def anomaly_score(self, x):
        return ((self.reconstruct(x) - x) ** 2).flatten(1).mean(1)

    @torch.no_grad()
    def calibrate(self, loader, device, batches=20):
        """Typical anomaly score on real test clothes, used to judge user uploads."""
        s = torch.cat([self.anomaly_score(x.to(device)) for _, (x, _) in zip(range(batches), loader)])
        return {"score_mean": s.mean().item(), "score_std": s.std().item()}