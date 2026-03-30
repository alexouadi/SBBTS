import math

import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):

        """Initialize the module/class state.

        Configure internal attributes used by the SBBTS model and utilities.

        Args:
            d_model: Internal embedding dimension used by the networks.
            max_len: Maximum sequence length supported by positional encodings.

        Returns:
            None.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

    Args:
            x: Input embedding tensor of shape (batch, length, d_model).

        Returns:
            Computed output(s) produced by the function.
        """
        return self.pe[:x.size(1)]

class EncoderOnly(torch.nn.Module):
    def __init__(self, input_dim, d_model, nhead, n_layers, N, device):
        """Initialize the module/class state.

        Configure internal attributes used by the SBBTS model and utilities.

        Args:
            input_dim: Dimensionality of the raw input space.
            d_model: Internal embedding dimension used by the networks.
            nhead: Number of attention heads in the Transformer encoder.
            n_layers: Number of Transformer encoder layers.
            N: Number of time points (or sequence length minus one, depending on context).
            device: Torch device used for allocations and cleanup.

        Returns:
            None.
        """
        super().__init__()

        self.mask = nn.Transformer(batch_first=True).generate_square_subsequent_mask(N).to(device)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.past_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)  # , norm=norm)

    def forward(self, y_past, training=False):
        """Forward.

    Args:
            y_past: Past trajectory used as temporal context.
            training: Whether to use the causal mask for training-time encoding.

        Returns:
            Computed output(s) produced by the function.
        """
        y_proj = self.input_proj(y_past)  # (B, L, d_model)
        y_emb = y_proj + self.pe(y_proj)  # (B, L, d_model)

        if training:
            return self.past_encoder(y_emb, mask=self.mask)  # (B, L, d_model)
        return self.past_encoder(y_emb)[:, -1:]  # (B, 1, d_model)
