import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):

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
        return self.pe[:x.size(1)]


class EncoderOnly(torch.nn.Module):
    def __init__(self, input_dim, d_model, nhead, n_layers, N, device):
        super().__init__()

        self.mask = nn.Transformer(batch_first=True).generate_square_subsequent_mask(N).to(device)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.past_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)  # , norm=norm)

    def forward(self, y_past, training=False):
        y_proj = self.input_proj(y_past)  # (B, L, d_model)
        y_emb = y_proj + self.pe(y_proj)  # (B, L, d_model)

        if training:
            return self.past_encoder(y_emb, mask=self.mask)  # (B, L, d_model)
        return self.past_encoder(y_emb)[:, -1:]  # (B, 1, d_model)
