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
        # avoid division by zero by adding a small epsilon inside the square root
        self.eps = eps
        self.dim = dim
        # the gamma parameter in the formula, initialized to ones
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Compute the squared mean with epsilon for numerical stability
        mean_square = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        # Compute the root mean square
        rms = torch.rsqrt(mean_square)
        return x * rms
        
    def forward(self, x: torch.Tensor):
        output = self.weight * self._norm(x.float())
        return output.to(x.dtype)


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
    # Head dimension must be even for RoPE since we pair up sin/cos components into complex numbers
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
    
    # Build the theta vector: (head_dim/2,)
    i = torch.arange(head_dim // 2, device=device)
    theta_vec = theta ** (-2 * i / head_dim)  # shape: (head_dim/2,)
    
    # Build the position vector: (seq_len,)
    pos = torch.arange(seq_len, device=device)  # shape: (seq_len,)
    
    # Compute the frequency tensor: (seq_len, head_dim/2)
    freqs = pos.unsqueeze(-1) * theta_vec.unsqueeze(0)  # shape: (seq_len, head_dim/2)
    
    # Convert to complex numbers: (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)  # shape: (seq_len, head_dim/2)

    return freqs_complex


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

    # Reshape x to separate the last dimension into complex pairs
    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))  # shape: (B, Seq_Len, H, Head_Dim/2)
    
    # Unsqueeze freqs_complex to align with x_complex for broadcasting
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # shape: (1, Seq_Len, 1, Head_Dim/2)
    
    # Perform the complex multiplication (rotation)
    x_rotated = x_complex * freqs_complex  # shape: (B, Seq_Len, H, Head_Dim/2)
    
    # Convert back to real numbers by interleaving the real and imaginary parts
    x_out = torch.view_as_real(x_rotated).flatten(-2)  # shape: (B, Seq_Len, H, Head_Dim)
    
    # Flatten the last two dimensions back to the original shape
    x_out = x_out.view(*x.shape)  # shape: (B, Seq_Len, H, Head_Dim)
    
    return x_out.type_as(x).to(device)  # ensure output has the same dtype as input


# ──────────────────────────────────────────────
# Layer 2 helper — repeat K/V heads for GQA
# ──────────────────────────────────────────────
# Expands (B, Seq, n_kv_heads, head_dim)
#      to (B, Seq, n_kv_heads * n_rep, head_dim)
# If n_rep == 1, return x unchanged.
# ──────────────────────────────────────────────
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA (grouped query attention).
    
    Input:  (B, Seq, n_kv_heads, head_dim)
    Output: (B, Seq, n_kv_heads * n_rep, head_dim)
    
    Each KV head is repeated contiguously: [k0, k0, k0, k0, k1, k1, k1, k1]
    """
    if n_rep == 1:
        return x
    B, Seq, n_kv_heads, head_dim = x.shape
    x = x.unsqueeze(3).repeat(1, 1, 1, n_rep, 1)  # shape: (B, Seq, n_kv_heads, n_rep, head_dim)
    x = x.view(B, Seq, n_kv_heads * n_rep, head_dim)  # shape: (B, Seq, n_kv_heads * n_rep, head_dim)
    return x

# ──────────────────────────────────────────────
# Layer 2+3 — Self-attention with KV cache
# ──────────────────────────────────────────────
# __init__ checklist:
#   - n_kv_heads, n_heads, n_rep, head_dim
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
        
        self.n_heads = args.n_heads
        self.n_heads_kv = args.n_kv_heads or args.n_heads
        self.n_rep = self.n_heads // self.n_heads_kv
        self.head_dim = args.dim // args.n_heads
        assert self.n_heads % self.n_heads_kv == 0, "n_heads must be divisible by n_kv_heads"
        
        # Define the linear layers for query, key, value, and output projections
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        
        # Initialize the key and value caches
        self.register_buffer("cache_k", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim), persistent=False)
        self.register_buffer("cache_v", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim), persistent=False)
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
    ) -> torch.Tensor:
        
        # View operation is because the linear layers output (B, Seq_Len, n_heads * head_dim) and we want to reshape to separate heads
        # Multiply input by projection matrices to get query, key, value tensors
        B, Seq_Len, _ = x.shape
        xq = self.wq(x).view(B, Seq_Len, self.n_heads, self.head_dim)  # shape: (B, Seq_Len, n_heads, head_dim)
        xk = self.wk(x).view(B, Seq_Len, self.n_heads_kv, self.head_dim)  # shape: (B, Seq_Len, n_heads_kv, head_dim)
        xv = self.wv(x).view(B, Seq_Len, self.n_heads_kv, self.head_dim)  # shape: (B, Seq_Len, n_heads_kv, head_dim)   
        
        # Apply RoPE to queries and keys
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        
        # Write the "new" keys and values into the cache at the appropriate positions
        self.cache_k[:B, start_pos : start_pos + Seq_Len] = xk
        self.cache_v[:B, start_pos : start_pos + Seq_Len] = xv
        
        # Read the full keys and values from the cache for the current sequence (up to start_pos + Seq_Len)
        keys = self.cache_k[:B, 0: start_pos + Seq_Len]  # shape: (B, start_pos + Seq_Len, n_heads_kv, head_dim)
        values = self.cache_v[:B, 0: start_pos + Seq_Len]  # shape: (B, start_pos + Seq_Len, n_heads_kv, head_dim)
        
        # Repeat the keys and values to match the number of query heads if necessary (GQA)
        keys = repeat_kv(keys, self.n_rep)  # shape: (B, start_pos + Seq_Len, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # shape: (B, start_pos + Seq_Len, n_heads, head_dim)
        
        # Transpose for attention computation: (B, n_heads, Seq_Len, head_dim)
        xq = xq.transpose(1, 2)  # shape: (B, n_heads, Seq_Len, head_dim)
        keys = keys.transpose(1, 2)  # shape: (B, n_heads, start_pos + Seq_Len, head_dim)
        values = values.transpose(1, 2)  # shape: (B, n_heads, start_pos + Seq_Len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)  # shape: (B, n_heads, Seq_Len, start_pos + Seq_Len) 
        
        # Compute attention probabilities with softmax in float32 for numerical stability
        scores = scores.float()  # ensure scores are in float32 for softmax stability
        attn_probs = F.softmax(scores, dim=-1)  # shape: (B, n_heads, Seq_Len, start_pos + Seq_Len)
        attn_probs = attn_probs.type_as(scores)  # cast back to original dtype if needed
        
        output = torch.matmul(attn_probs, values)  # shape: (B, n_heads, Seq_Len, head_dim) 
        
        # Reshape and project the output back to the original dimension
        output = output.transpose(1, 2).contiguous().view(B, Seq_Len, self.n_heads * self.head_dim)  # shape: (B, Seq_Len, n_heads * head_dim)
        output = self.wo(output)  # shape: (B, Seq_Len, dim)    
        
        return output    
        


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
        # Compute the hidden dimension based on the provided arguments
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = ((hidden_dim + args.multiple_of - 1) // args.multiple_of) * args.multiple_of  # round up to nearest multiple of multiple_of        
        
        # Define the three linear layers for the feed-forward network (gate, up, down)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False) 
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Compute the feed-forward output using the SwiGLU activation function
        gate = F.silu(self.w1(x))  # shape: (B, Seq_Len, hidden_dim)
        x_3 = self.w3(x)  # shape: (B, Seq_Len, hidden_dim)
        fused = gate * x_3  # shape: (B, Seq_Len, hidden_dim)
        output = self.w2(fused)  # shape: (B, Seq_Len, dim)
        return output

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
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.attention = SelfAttention(args)
        self.ffn = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        
        
    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.ffn(self.ffn_norm(h))
        return out

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
        
        self.args = args
        self.args_vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # Define the number of layers
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])
        
        # Define the rms normalization layer
        self.norm = RMSNorm(args.dim, args.norm_eps)
        
        # Define the output linear layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Freqs_complex precomputation
        self.freq_complex = precompute_theta_pos_frequencies(
            head_dim=args.dim // args.n_heads,
            seq_len=args.max_seq_len * 2,
            device=args.device,
        )
        
        
        

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        
        # text of batch size, sequence length, and assert that sequence length is 1 (decode-only)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "This model is decode-only and expects seq_len=1"
        
        # Embed tokens
        x = self.tok_embeddings(tokens)  # shape: (B, Seq_Len, Dim)
        
        # Retrieve the relevant slice of precomputed frequencies (do not modify self.freq_complex)
        freqs = self.freq_complex[start_pos : start_pos + seq_len]  # shape: (Seq_Len, Head_Dim/2)
        
        # apply each encoder block in sequence
        for layer in self.layers:
            x = layer(x, start_pos, freqs)  # shape: (B, Seq_Len, Dim)
        
        x = self.norm(x)  # shape: (B, Seq_Len, Dim)
        logits = self.output(x)  # shape: (B, Seq_Len, Vocab_Size)
        
        return logits.float()  # ensure output is in float32 for softmax stability
                


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
#     print("Model initialized successfully. Running smoke test...")
#     print(model)
    
#     print("Testing forward pass with dummy tokens...")
#     # Simulate prefill of 10 tokens one at a time
#     for pos in range(10):
#         tok = torch.randint(0, args.vocab_size, (1, 1))
#         logits = model(tok, start_pos=pos)
#         assert logits.shape == (1, 1, args.vocab_size), f"Bad shape at pos {pos}: {logits.shape}"

#     print("All shape checks passed.")