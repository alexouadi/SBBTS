import math

import torch
import torch.nn.functional as F
from torch import nn

from encoder_only import EncoderOnly


def get_timestep_embedding(timesteps, embedding_dim=128):
    """Get timestep embedding.

    Get timestep embedding. This routine is part of the SBBTS workflow and related utilities.

    Args:
        timesteps: Tensor of time indices/timestamps to encode.
        embedding_dim: Size of the sinusoidal time embedding.

    Returns:
        Computed output(s) produced by the function.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0, 1])
    return emb


class MLP(torch.nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim):
        """Initialize the module/class state.

        Configure internal attributes used by the SBBTS model and utilities.

        Args:
            input_dim: Dimensionality of the raw input space.
            d_model: Internal embedding dimension used by the networks.
            hidden_dim: Hidden dimension for MLP projection blocks.

        Returns:
            None.
        """
        super().__init__()
        self.d_model = d_model

        self.t_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.y_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

        self.cond_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, input_dim)
        )

    def forward(self, t, y, h):
        """Forward.

        Forward. This routine is part of the SBBTS workflow and related utilities.

        Args:
            t: Continuous time variable.
            y: Current state values.
            h: Context embedding from the temporal encoder.

        Returns:
            Computed output(s) produced by the function.
        """
        t_embed = self.t_encoder(get_timestep_embedding(t, self.d_model))  # (B, L, d_model)
        y_embed = self.y_encoder(y)  # (B, L, d_model)
        y_emb = torch.cat([t_embed, y_embed, h], dim=-1)
        return self.cond_fusion(y_emb)  # (B, L, d)


class ScoreNN(torch.nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim, nhead, n_layers, L, device):
        """Initialize the module/class state.

        Configure internal attributes used by the SBBTS model and utilities.

        Args:
            input_dim: Dimensionality of the raw input space.
            d_model: Internal embedding dimension used by the networks.
            hidden_dim: Hidden dimension for MLP projection blocks.
            nhead: Number of attention heads in the Transformer encoder.
            n_layers: Number of Transformer encoder layers.
            L: Maximum sequence length used by the encoder mask.
            device: Torch device used for allocations and cleanup.

        Returns:
            None.
        """
        super().__init__()

        self.tf_encoder = EncoderOnly(input_dim, d_model, nhead, n_layers, L, device)
        self.get_drift = MLP(input_dim, d_model, hidden_dim)

    def forward(self, t, y, y_past):
        """Forward.

        Forward. This routine is part of the SBBTS workflow and related utilities.

        Args:
            t: Continuous time variable.
            y: Current state values.
            y_past: Past trajectory used as temporal context.

        Returns:
            Computed output(s) produced by the function.
        """
        h = self.tf_encoder(y_past)
        return self.get_drift(t, y, h)


# for beta small
class InverseMLP(torch.nn.Module):
    def __init__(self, input_dim, d_model, t_model):
        """Initialize the module/class state.

        Configure internal attributes used by the SBBTS model and utilities.

        Args:
            input_dim: Dimensionality of the raw input space.
            d_model: Internal embedding dimension used by the networks.
            t_model: Drift network that consumes time/state embeddings.

        Returns:
            None.
        """
        super().__init__()
        self.d_model = d_model

        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_model),
            nn.LayerNorm(t_model),
            nn.GELU(),
            nn.Linear(t_model, t_model)
        )

        self.y_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model + t_model, min(d_model, t_model)),
            nn.SiLU(),
            nn.Linear(min(d_model, t_model), input_dim)
        )

    def forward(self, t, y):
        """Forward.

        Forward. This routine is part of the SBBTS workflow and related utilities.

        Args:
            t: Continuous time variable.
            y: Current state values.

        Returns:
            Computed output(s) produced by the function.
        """
        t_embed = self.t_encoder(t)
        y_embed = self.y_encoder(y)
        y_emb = torch.cat([t_embed, y_embed], dim=-1)
        return self.decoder(y_emb)
