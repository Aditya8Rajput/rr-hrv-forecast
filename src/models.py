import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        horizon: int = 30,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        win_mean = x.mean(dim=1, keepdim=True)  # (B, 1, 1)
        out, _ = self.rnn(x - win_mean)
        last_hidden = self.norm(out[:, -1, :])
        return self.head(last_hidden) + win_mean.squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(
        self,
        horizon: int = 30,
        hidden_size: int = 112,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        win_mean = x.mean(dim=1, keepdim=True)
        out, _ = self.rnn(x - win_mean)
        last_hidden = self.norm(out[:, -1, :])
        return self.head(last_hidden) + win_mean.squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        horizon: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        # Concatenate mean-pooled context + final beat representation
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        win_mean = x.mean(dim=1, keepdim=True)
        h = self.pos_enc(self.input_proj(x - win_mean))
        h = self.norm(self.encoder(h))
        pooled = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=-1)
        return self.head(pooled) + win_mean.squeeze(-1)


def build_model(name: str, horizon: int = 30) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMForecaster(horizon=horizon)
    if name == "gru":
        return GRUForecaster(horizon=horizon)
    if name == "transformer":
        return TransformerForecaster(horizon=horizon)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)