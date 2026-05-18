# Will start building the llama2 model

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


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

    @classmethod
    def from_json(cls, json_path: str) -> "ModelArgs":
        """Load ModelArgs from a JSON file."""
        with open(json_path, 'r') as f:
            params = json.load(f)
        return cls(**params)


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
        self.scale = dim ** -0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
      return x * self.weight

    def forward(self, x: torch.Tensor):
      return self._norm(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


# ──────────────────────────────────────────────
# Layer — Sinoidal absolute positional embeddings 
# ──────────────────────────────────────────────
# PE(pos, 2*i)   = sin(pos / (10000 ** (2*i / d_model)))
# PE(pos, 2*i+1) = cos(pos / (10000 ** (2*i / d_model)))
# ──────────────────────────────────────────────
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Create the positional embeddings matrix
        # Shape: [Max_seq_len, dim]
        pe = torch.zeros(max_seq_len, dim)
        
        # Shape: [max_seq_len, 1]
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        
        # Shape: [dim/2]
        # 1 / (10000 ** (2*i / d_model)) = exp(-(2*i / d_model) * log(10000))
        # exp(log(a)) = a
        # a^x = (exp(log(a)))^x = exp(log(a) * x)
        div_term = torch.exp(-1*torch.arange(0, dim, 2)/ dim * (math.log(10000.0)))
        # div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        # Apply the sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:seq_len]


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
    # paper says that the head dimension must be divisible by 2
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"

    # Build the the theta
    # shape (head_dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim)).to(device)

    # Construct the m positions
    m = torch.arange(seq_len, device=device)

    # Outer product of m and freqs
    # shape (seq_len, head_dim/2)
    freqs = torch.outer(m, freqs).float()

    # Compute the result -> (seq_len, head_dim/2)
    # Result is a complex number of torch.polar(abs, angle)
    # Abs = 1, angle is freqs or theta
    result = torch.polar(torch.ones_like(freqs), freqs)
    return result



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

    # Input shape for x
    B, S, H, D = x.shape
    assert D % 2 == 0, "head_dim must be divisible by 2"

    # Output shape [B, Seq_len, H, Head_Dim/2]
    # view as complex after reshaping last dim
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # the theta
    freqs_complex = freqs_complex.to(device)

    # Freqs complex:
    # (seq_len, dim/2) => (1, seq_len, 1, dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # element wise multiplication
    # from right to left (what's different, broadcast) -> Broadcase over B, H
    # (B, S, H, Dim/2) * (1, S, 1, Dim/2) => (B, S, H, Dim/2)
    # multiply (rotation in complex plane)
    x_complex = x_complex * freqs_complex

    # view_as_real, reshape back (B, S, H, Dim/2) -> [B, S, H, Dim/2, 2]
    # last dimension of size 2 represents the real and imaginary components of complex numbers.
    x = torch.view_as_real(x_complex)

    # reshape to return to the original dimension
    x = x.reshape(B, S, H, D)
    return x


# ──────────────────────────────────────────────
# Layer 2 helper — repeat K/V heads for GQA
# ──────────────────────────────────────────────
# Expands (B, Seq, n_kv_heads, head_dim)
#      to (B, Seq, n_kv_heads * n_rep, head_dim)
# If n_rep == 1, return x unchanged.
# ──────────────────────────────────────────────
# Before expand:  [k0, k1, k2]  shape: (B, S, n_kv, 1, head_dim)
# After expand:   [k0k0k0, k1k1k1, k2k2k2]  shape: (B, S, n_kv, n_rep, head_dim)
# After reshape:  [k0, k0, k0, k1, k1, k1, k2, k2, k2]  shape: (B, S, n_kv*n_rep, head_dim)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    else:
        B, Seq, n_kv_heads, head_dim = x.shape
        x = x.unsqueeze(-2).expand(B, Seq, n_kv_heads, n_rep, head_dim)
        x = x.reshape(B, Seq, n_rep * n_kv_heads, head_dim)
        return x


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
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.n_heads_kv = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads
        # print(self.n_heads_kv, args.n_kv_heads, " n heads KV **** Hosssam")
        self.n_rep = self.n_heads // self.n_heads_kv   # ✅ uses resolved value
        assert self.n_heads % self.n_heads_kv == 0, "n_heads must be divisible by n_kv_heads"

        # Linear layers (output of Wk, Wv should be KV based )
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)


        # Initialize the key and value caches (B, S, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim), persistent=False)
        self.register_buffer("cache_v", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
    ) -> torch.Tensor:
        # B, S, D is the shape of the input
        B, S, _ = x.shape

        # Linear projections for the self-attention
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_heads_kv, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_heads_kv, self.head_dim)

        # Apply RoPE to xq and xk
        xq = apply_rotary_embeddings(xq, freqs_complex, device=xq.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=xk.device)

        # Save the cached position at the start position
        self.cache_k[:B, start_pos:start_pos+S, :, :] = xk
        self.cache_v[:B, start_pos:start_pos+S, :, :] = xv

        # Retrieve everything from 0 to start_pos + S
        # Up to B to avoid the max batch size
        cache_k = self.cache_k[:B, :start_pos+S, :, :]
        cache_v = self.cache_v[:B, :start_pos+S, :, :]

        # Compute softmax:
        # repeat KV: You use everything in the cache
        # input shape is: (B, S_kv, n_kv, head_dim)
        # output shape: (B, S_kv, n_kv*n_rep, head_dim)
        # output shape for each of k, v: (B, S_kv, n_heads, head_dim)
        # S_kv is the sequence length of everything
        keys   = repeat_kv(cache_k, self.n_rep)
        values = repeat_kv(cache_v, self.n_rep)

        # xq shape: (B, S, self.n_heads, self.head_dim)
        # keys shape: (B, S_kv, self.n_heads, self.head_dim)
        # Matmul (After two transposes reduce on head dim): (B, n_heads, S, head_dim) * (B, n_heads, head_dim, S) ->
        # Output: (B, n_heads, S, S_kv)

        # Same thing
        # keys = keys.permute(0, 2, 1, 3)
        # values = values.permute(0, 2, 1, 3)
        scores = torch.matmul(xq.transpose(1,2), keys.transpose(1, 2).transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)

        # Multiply with the v
        # Scores: (B, n_heads, S, S_kv); values (B, S_kv, n_heads, head_dim)
        # (1) (B, n_heads, S, S_kv) * (B, S_kv, n_heads, head_dim) ->
        # (2) (B, n_heads, S, S_kv) * (B, n_heads, S_kv, head_dim) -> B, n_heads, S, head_dim

        # Objective: (B, n_heads, S, S_kv) @ (B, n_heads, S_kv, head_dim) → (B, n_heads, S, head_dim)
        output = torch.matmul(scores, values.transpose(1,2))
        output = output.transpose(1, 2).contiguous().view(B, S,
                                                          self.n_heads*self.head_dim)

        # Apply the output projection
        output = self.wo(output)

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

        # ffn dim_multiplier
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)

        # Round up to the nearest integer
        if hidden_dim % args.multiple_of != 0:
            hidden_dim = math.ceil(hidden_dim / args.multiple_of) * args.multiple_of

        # Init the projection layers
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()


    # Three linear layers: w1, w2, w3 (all no bias)
    # forward: output = w2( silu(w1(x)) * w3(x) )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      output = self.w2(self.silu(self.w1(x)) * self.w3(x))
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
        # Define the dim, attention norm, ffn norm, attention norm, ffn
        self.dim = args.dim
        self.attention_norm = RMSNorm(self.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, args.norm_eps)
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)


    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
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

        # Save the args
        self.args = args
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.dim = args.dim

        # inititlize the embedding
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # initilize the positional embedding
        # self.pos_embedding = SinusoidalPositionalEmbedding(args.dim, args.max_seq_len * 2)
        
        # Layers inside the transformer
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Complex rotation anagle frequency
        self.freqs_complex = precompute_theta_pos_frequencies(self.dim // self.args.n_heads,
                                                              args.max_seq_len * 2,
                                                              args.device)
        self.freqs_complex = self.freqs_complex.to(args.device)



    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        # Get the shape of the tokenized input tokens
        B, seq_len = tokens.shape
        assert seq_len == 1

        # Embed the tokens --> B, S, D
        x = self.tok_embeddings(tokens)
        
        # if self.args.positional_embedding_type == "sinusoidal":
        #     # Add the positional embeddings to the token embeddings
        #     x = x + self.pos_embedding(tokens).unsqueeze(0) # unsqueeze for batch dimension

        # Slice the complex frequencies
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            x = layer(x, start_pos, freqs_complex) # B, S, D
        x = self.norm(x) # B, S, D
        x = self.output(x) # B, S, V
        return x.float()

    def forward_train(self, tokens: torch.Tensor) -> torch.Tensor:
        B, seq_len = tokens.shape
        assert seq_len > 1

        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[:seq_len]
        # Upper right triange
        mask = torch.triu(torch.full((seq_len, seq_len),
                                     float("-inf"), device=self.args.device,
                                      dtype=h.dtype), diagonal=1)

        for layer in self.layers:

          # Layer norm
          r    = layer.attention_norm(h)
          attn = layer.attention
          hd   = attn.head_dim

          # attention linear projections
          xq   = attn.wq(r).view(B, seq_len, attn.n_heads,  hd)
          xk   = attn.wk(r).view(B, seq_len, attn.n_heads_kv,  hd)
          xv   = attn.wv(r).view(B, seq_len, attn.n_heads_kv,  hd)

          # q, k rotary embeddings
          xq = apply_rotary_embeddings(xq, freqs_complex, device=xq.device)
          xk = apply_rotary_embeddings(xk, freqs_complex, device=xk.device)

          # Repeat KV cache
          xk = repeat_kv(xk, attn.n_rep)
          xv = repeat_kv(xv, attn.n_rep)

          # Attention scores with its transposes
          scores = torch.matmul(xq.transpose(1,2), xk.transpose(1, 2).transpose(-2, -1)) / math.sqrt(hd)
          scores = scores + mask[:seq_len, :seq_len]
          scores = F.softmax(scores, dim=-1)
          output = torch.matmul(scores, xv.transpose(1,2))
          output = output.view(B, seq_len, attn.n_heads * hd)


        output = self.norm(output)
        output = self.output(output)
        return output.float()
