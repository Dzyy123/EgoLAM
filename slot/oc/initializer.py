"""OCLAL fixed slot initialization (per-slot learnable mean and std).

Reference: Object-Centric Latent Action Learning (arXiv:2502.09680), Appendix C
"Fixed Initialization for Slot Stability":

    train:     s_k^(init) = mu_k + sigma_k (.) eps,   eps ~ N(0, I)
    inference: s_k^(init) = mu_k

The per-slot (mu_k, sigma_k) parameterization -- instead of VideoSAUR's default
`RandomInit`, which shares a single (mu, sigma) across all slots -- makes the same
slot index correspond to the same semantic object across episodes. This is what
enables training one action probe per slot index in Stage 2.
"""

import torch
from torch import nn


class FixedInit(nn.Module):
    """Fixed per-slot initialization with optional training-time noise.

    Args:
        n_slots: number of slots K.
        dim: slot dimension d_s.
        initial_std: init scale for mu and sigma. Defaults to dim ** -0.5 (same
            default as VideoSAUR initializers).
    """

    def __init__(self, n_slots: int, dim: int, initial_std: float | None = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        if initial_std is None:
            initial_std = dim**-0.5
        # mu_k: [1, K, D]
        self.mean = nn.Parameter(torch.randn(1, n_slots, dim) * initial_std)
        # sigma_k stored in log space for positivity: [1, K, D]
        self.log_std = nn.Parameter(torch.log(torch.ones(1, n_slots, dim) * initial_std))

    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns initial slots of shape [B, K, D]."""
        if self.training:
            noise = torch.randn(batch_size, self.n_slots, self.dim, device=self.mean.device)
            return self.mean + noise * self.log_std.exp()
        # Deterministic inference: s_k = mu_k
        return self.mean.expand(batch_size, -1, -1)
