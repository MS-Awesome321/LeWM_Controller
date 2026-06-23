import torch
import torch.nn.functional as F
from torch import nn



class _ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentDeltaPredictor(nn.Module):
    """Predicts the next JEPA CLS embedding given the current embedding + a delta embedding.

    Both inputs must already be in embedding space — raw delta_xyz should be
    encoded with FourierDeltaEmbedder before calling this module.

    emb:       (B, D) – JEPA projected CLS token
    delta_emb: (B, D) – Fourier-embedded motor displacement
    returns:   (B, D) – predicted next embedding (residual on input)
    """

    def __init__(
        self,
        emb_dim: int = 192,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(emb_dim)
        self.delta_norm = nn.LayerNorm(emb_dim)

        self.blocks = nn.ModuleList([
            _ResidualBlock(emb_dim, hidden_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.out  = nn.Linear(emb_dim, emb_dim)

        # zero-init: starts as identity, residual grows during training
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, emb: torch.Tensor, delta_emb: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(emb) + self.delta_norm(delta_emb)
        for block in self.blocks:
            x = block(x)
        return emb + self.out(self.norm(x))
