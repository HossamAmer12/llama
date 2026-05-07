from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 12
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None


# ──────────────────────────────────────────────
# Layer 1a — RMS normalization
# ──────────────────────────────────────────────
# Hint: unlike LayerNorm, no mean subtraction.
# Formula: x / sqrt(mean(x^2) + eps)  *  gamma
# Learnable parameter: self.weight (shape: dim,)
# ──────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # TODO

    def _norm(self, x: torch.Tensor):
        # TODO
        pass

    def forward(self, x: torch.Tensor):
        # TODO
        pass


# ──────────────────────────────────────────────
# Layer 1b — RoPE frequency precomputation
# ──────────────────────────────────────────────
# Returns a complex tensor of shape (seq_len, head_dim/2)
# representing rotation angles for each position.
# Formula: theta_i = 10000^(-2i/head_dim), then
#          freqs[m] = m * theta  (outer product)
#          result = polar(1, freqs)  →  complex
# ──────────────────────────────────────────────
def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
) -> torch.Tensor:
    # TODO
    pass


# ──────────────────────────────────────────────
# Layer 1c — Apply RoPE to query / key tensors
# ──────────────────────────────────────────────
# Input x shape:  (B, Seq_Len, H, Head_Dim)
# Output shape:   same as input
# Steps:
#   1. view_as_complex after reshaping last dim to (..., 2)
#   2. unsqueeze freqs_complex for batch + head dims
#   3. multiply (rotation in complex plane)
#   4. view_as_real, reshape back
# ──────────────────────────────────────────────
def apply_rotary_embeddings(
    x: torch.Tensor, freqs_complex: torch.Tensor, device: str
) -> torch.Tensor:
    # TODO
    pass


# ──────────────────────────────────────────────
# Layer 2 helper — repeat K/V heads for GQA
# ──────────────────────────────────────────────
# Expands (B, Seq, n_kv_heads, head_dim)
#      to (B, Seq, n_kv_heads * n_rep, head_dim)
# If n_rep == 1, return x unchanged.
# ──────────────────────────────────────────────
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # TODO
    pass


# ──────────────────────────────────────────────
# Layer 2+3 — Self-attention with KV cache
# ──────────────────────────────────────────────
# __init__ checklist:
#   - n_kv_heads, n_heads_q, n_rep, head_dim
#   - wq: dim  → n_heads   * head_dim  (no bias)
#   - wk: dim  → n_kv_heads * head_dim (no bias)
#   - wv: dim  → n_kv_heads * head_dim (no bias)
#   - wo: n_heads * head_dim → dim     (no bias)
#   - cache_k, cache_v: zeros (max_batch, max_seq, n_kv_heads, head_dim)
#
# forward checklist:
#   1. Project xq, xk, xv and reshape to head dims
#   2. Apply RoPE to xq and xk
#   3. Write xk, xv into cache at start_pos
#   4. Read full keys/values from cache [0 : start_pos+seq_len]
#   5. repeat_kv so K/V heads match Q heads
#   6. Transpose to (B, H, Seq, Head_Dim) for matmul
#   7. scores = (xq @ keys.T) / sqrt(head_dim)
#   8. softmax in float32, cast back
#   9. output = scores @ values, reshape, project with wo
# ──────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # TODO

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
    ) -> torch.Tensor:
        # TODO
        pass


# ──────────────────────────────────────────────
# Layer 4a — SwiGLU feed-forward block
# ──────────────────────────────────────────────
# hidden_dim derivation (do this from memory):
#   start: 4 * dim
#   apply: int(2 * hidden / 3)
#   apply: ffn_dim_multiplier if set
#   round up to nearest multiple of `multiple_of`
#
# Three linear layers: w1, w2, w3 (all no bias)
# forward: output = w2( silu(w1(x)) * w3(x) )
# ──────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        pass


# ──────────────────────────────────────────────
# Layer 4b — Transformer encoder block
# ──────────────────────────────────────────────
# Pre-norm architecture:
#   h   = x + attention( norm(x) )
#   out = h + ffn( norm(h) )
# Two RMSNorm instances: attention_norm, ffn_norm
# ──────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # TODO

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        # TODO
        pass


# ──────────────────────────────────────────────
# Layer 5 — Full transformer (inference only)
# ──────────────────────────────────────────────
# __init__ checklist:
#   - tok_embeddings: Embedding(vocab_size, dim)
#   - layers: ModuleList of n_layers EncoderBlocks
#   - norm: RMSNorm
#   - output: Linear(dim, vocab_size, bias=False)
#   - freqs_complex: precomputed for max_seq_len * 2
#
# forward(tokens, start_pos):
#   - assert seq_len == 1  (decode-only, KV cache assumed)
#   - embed tokens
#   - slice freqs_complex[start_pos : start_pos + seq_len]
#   - run through all layers
#   - norm → output projection → return .float()
# ──────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1
        # TODO
    
        

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        # TODO
        pass


# ──────────────────────────────────────────────
# Quick smoke test — run this to check your work
# ──────────────────────────────────────────────
# if __name__ == "__main__":
#     args = ModelArgs(
#         dim=128,
#         n_layers=2,
#         n_heads=4,
#         n_kv_heads=2,
#         vocab_size=1000,
#         max_batch_size=2,
#         max_seq_len=64,
#         device="cpu",
#     )

#     model = Transformer(args)

#     # Simulate prefill of 10 tokens one at a time
#     for pos in range(10):
#         tok = torch.randint(0, args.vocab_size, (1, 1))
#         logits = model(tok, start_pos=pos)
#         assert logits.shape == (1, 1, args.vocab_size), f"Bad shape at pos {pos}: {logits.shape}"

#     print("All shape checks passed.")