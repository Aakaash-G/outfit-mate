"""The generative model used by Outfit Mate: a Variational Autoencoder trained on Fashion-MNIST."""
from .vae import VAE

# 20 epochs take roughly 10-20 minutes on a laptop CPU, a few minutes on a GPU.
REGISTRY = {
    "vae": dict(cls=VAE, epochs=20, lr=1e-3, batch=128,
                role="New designs, design mixer, photo clean-up, find-similar, upload check"),
}