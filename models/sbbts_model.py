import math

import torch
import torch.nn.functional as F
from torch import nn

from encoder_only import EncoderOnly

def get_timestep_embedding(timesteps, embedding_dim=128):
<<<<<<< HEAD
    """
=======
    """Build sinusoidal time embeddings used by the drift network.

>>>>>>> f57dbd1e40ae50f9a610a18df70ef2ba0fb4ae11
    Args:
        timesteps: Tensor of times to embed.
        embedding_dim: Dimension of the sinusoidal embedding.

    Returns:
        Time-embedding tensor.
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
<<<<<<< HEAD
        """
=======
        """Initialize the MLP blocks used to predict the drift term.

>>>>>>> f57dbd1e40ae50f9a610a18df70ef2ba0fb4ae11
        Args:
            input_dim: Input feature dimension.
            d_model: Transformer/embedding dimension.
            hidden_dim: Hidden size of the MLP blocks.
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
<<<<<<< HEAD
        """
        Args:
            t: Continuous time variable.
            y: Current state values.
            h: Context embedding from the temporal encoder.
=======
        """Run a forward pass for the module.

        Args:
            t: Continuous time tensor.
            y: Current state tensor.
            h: Context embedding from the encoder.
>>>>>>> f57dbd1e40ae50f9a610a18df70ef2ba0fb4ae11

        Returns:
            Module output tensor.
        """
        t_embed = self.t_encoder(get_timestep_embedding(t, self.d_model))  # (B, L, d_model)
        y_embed = self.y_encoder(y)  # (B, L, d_model)
        y_emb = torch.cat([t_embed, y_embed, h], dim=-1)
        return self.cond_fusion(y_emb)  # (B, L, d)

class ScoreNN(torch.nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim, nhead, n_layers, L, device):
<<<<<<< HEAD
        """
        Args:
            input_dim: Dimensionality of the raw input space.
            d_model: Internal embedding dimension used by the networks.
            hidden_dim: Hidden dimension for MLP projection blocks.
            nhead: Number of attention heads in the Transformer encoder.
            n_layers: Number of Transformer encoder layers.
            L: Maximum sequence length used by the encoder mask.
            device: Torch device used for allocations and cleanup.
=======
        """Initialize the full score network (encoder + drift head).

        Args:
            input_dim: Input feature dimension.
            d_model: Transformer/embedding dimension.
            hidden_dim: Hidden size of the MLP blocks.
            nhead: Number of attention heads.
            n_layers: Number of transformer encoder layers.
            L: Maximum sequence length.
            device: Torch device used by the model.
>>>>>>> f57dbd1e40ae50f9a610a18df70ef2ba0fb4ae11
        """
        super().__init__()

        self.tf_encoder = EncoderOnly(input_dim, d_model, nhead, n_layers, L, device)
        self.get_drift = MLP(input_dim, d_model, hidden_dim)

    def forward(self, t, y, y_past):
<<<<<<< HEAD
        """
        Args:
            t: Continuous time variable.
            y: Current state values.
            y_past: Past trajectory used as temporal context.
=======
        """Run a forward pass for the module.

        Args:
            t: Continuous time tensor.
            y: Current state tensor.
            y_past: Past trajectory/context sequence.

        Returns:
            Module output tensor.
>>>>>>> f57dbd1e40ae50f9a610a18df70ef2ba0fb4ae11
        """
        h = self.tf_encoder(y_past)
        return self.get_drift(t, y, h)

# for beta small
class InverseMLP(torch.nn.Module):
    def __init__(self, input_dim, d_model, t_model):
<<<<<<< HEAD
        """
        Args:
            input_dim: Dimensionality of the raw input space.
            d_model: Internal embedding dimension used by the networks.
            t_model: Drift network that consumes time/state embeddings.
=======
        """Initialize the inverse transport MLP used for small-beta correction.

        Args:
            input_dim: Input feature dimension.
            d_model: Transformer/embedding dimension.
            t_model: Pretrained drift model used for inverse transport.
>>>>>>> f57dbd1e40ae50f9a610a18df70ef2ba0fb4ae11
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
<<<<<<< HEAD
        """
        Args:
            t: Continuous time variable.
            y: Current state values.
=======
        """Run a forward pass for the module.

        Args:
            t: Continuous time tensor.
            y: Current state tensor.

        Returns:
            Module output tensor.
>>>>>>> f57dbd1e40ae50f9a610a18df70ef2ba0fb4ae11
        """
        t_embed = self.t_encoder(t)
        y_embed = self.y_encoder(y)
        y_emb = torch.cat([t_embed, y_embed], dim=-1)
        return self.decoder(y_emb)
