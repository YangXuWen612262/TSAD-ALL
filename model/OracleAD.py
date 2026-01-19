import torch
import torch.nn as nn
from typing import Tuple, Dict

class TemporalAttnPool(nn.Module):
    """
    Attention pooling over time dimension.
    Input: h [B, T, H]
    Output: c [B, H]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.proj(h))              # [B, T, H]
        a = self.score(u).squeeze(-1)            # [B, T]
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # [B, T, 1]
        c = (h * w).sum(dim=1)                   # [B, H]
        return c

class OracleAD(nn.Module):
    """
    OracleAD (structured temporal causality) implementation.
    """
    def __init__(
        self,
        num_vars: int,
        window_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        assert window_len >= 2, "window_len L must be >= 2"
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.N = num_vars
        self.L = window_len
        self.H = hidden_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.pool = TemporalAttnPool(hidden_dim)

        # Causal Graph Learning (MHSA)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_proj = nn.Linear(hidden_dim, 1)

    def encode(self, x_hist: torch.Tensor) -> torch.Tensor:
        B, T, N = x_hist.shape
        x_in = x_hist.permute(0, 2, 1).contiguous().view(B * N, T, 1)
        h_seq, _ = self.encoder(x_in)
        c = self.pool(h_seq)
        c = c.view(B, N, self.H)
        return c

    def cross_variable_attn(self, c: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mhsa(c, c, c, need_weights=False)
        c1 = self.norm1(c + attn_out)
        c2 = self.norm2(c1 + self.ffn(c1))
        return c2

    def decode(self, c_star: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H = c_star.shape
        
        # Init hidden state
        h0 = torch.zeros(self.num_layers, B * N, H, device=c_star.device, dtype=c_star.dtype)
        c0 = torch.zeros(self.num_layers, B * N, H, device=c_star.device, dtype=c_star.dtype)
        h0[-1] = c_star.reshape(B * N, H)

        # Decode
        T_dec = self.L
        z = torch.zeros(B * N, T_dec, 1, device=c_star.device, dtype=c_star.dtype)
        dec_out, _ = self.decoder(z, (h0, c0))
        y = self.out_proj(dec_out).squeeze(-1)

        y_recon = y[:, : self.L - 1]
        y_pred = y[:, self.L - 1]

        xhat_hist = y_recon.view(B, N, self.L - 1).permute(0, 2, 1).contiguous()
        xhat_last = y_pred.view(B, N)
        return xhat_hist, xhat_last

    @torch.no_grad()
    def distance_matrix(self, c_star: torch.Tensor) -> torch.Tensor:
        """Compute pairwise L2 distance matrix [B, N, N]"""
        return torch.cdist(c_star, c_star, p=2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hist = x[:, : self.L - 1, :]
        c = self.encode(x_hist)
        c_star = self.cross_variable_attn(c)
        xhat_hist, xhat_last = self.decode(c_star)
        return {"xhat_hist": xhat_hist, "xhat_last": xhat_last, "c_star": c_star}