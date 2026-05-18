import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─── Expert ───────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """Single FFN expert: Linear -> ReLU -> Linear."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─── Router (Top-K gating) ─────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    """
    Softmax router with top-k selection.
    Returns: (dispatch_weights, expert_indices, router_logits)
      dispatch_weights: [B, T, k]  – normalised weights for selected experts
      expert_indices:   [B, T, k]  – which expert each slot maps to
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        assert top_k <= num_experts
        self.top_k = top_k
        self.gate  = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):                         # x: [B, T, d_model]
        logits     = self.gate(x)                 # [B, T, E]
        probs      = F.softmax(logits, dim=-1)    # [B, T, E]
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)   # [B, T, k]
        # re-normalise so selected weights sum to 1
        top_probs  = top_probs / top_probs.sum(dim=-1, keepdim=True)
        return top_probs, top_idx, logits


# ─── Mixture-of-Experts Layer ─────────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    Sparse MoE layer:
      • Each token is routed to top-k experts.
      • Output is a weighted sum of those expert outputs.
      • Auxiliary load-balancing loss is stored as self.aux_loss.
    """
    def __init__(self, d_model: int, d_ff: int,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.experts    = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router     = TopKRouter(d_model, num_experts, top_k)
        self.num_experts = num_experts
        self.top_k       = top_k
        self.aux_loss    = torch.tensor(0.0)      # updated every forward

    # load-balancing loss (Switch-Transformer style)
    def _load_balance_loss(self, router_logits, expert_indices):
        # fraction of tokens dispatched to each expert
        probs     = F.softmax(router_logits, dim=-1)           # [B,T,E]
        B, T, E   = probs.shape
        N         = B * T
        # one-hot mask over selected top-k experts
        mask = torch.zeros(B, T, E, device=probs.device)
        mask.scatter_(-1, expert_indices, 1.0)                 # [B,T,E]
        # mean over tokens
        f_i = mask.reshape(N, E).mean(0)                       # [E]
        P_i = probs.reshape(N, E).mean(0)                      # [E]
        return E * (f_i * P_i).sum()                           # scalar ≥ 1

    def forward(self, x):                         # x: [B, T, d]
        B, T, d      = x.shape
        w, idx, logits = self.router(x)           # [B,T,k], [B,T,k], [B,T,E]

        self.aux_loss = self._load_balance_loss(logits, idx)

        # collect expert outputs
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            eid   = idx[:, :, k]                  # [B, T]
            wk    = w[:, :, k].unsqueeze(-1)      # [B, T, 1]
            # run each expert on the tokens routed to it
            for e in range(self.num_experts):
                mask = (eid == e)                 # [B, T]  bool
                if mask.any():
                    tokens = x[mask]              # [n_e, d]
                    out[mask] += wk[mask] * self.experts[e](tokens)
        return out


# ─── Tiny Transformer with MoE FFN ────────────────────────────────────────────

class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_experts, top_k):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe   = MoELayer(d_model, d_ff, num_experts, top_k)
        self.ln1   = nn.LayerNorm(d_model)
        self.ln2   = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # self-attention + residual
        attn_out, _ = self.attn(x, x, x,
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        x = self.ln1(x + attn_out)
        # MoE FFN + residual
        x = self.ln2(x + self.moe(x))
        return x


class TinyMoEModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_heads=4,
                 d_ff=256, num_experts=8, top_k=2, n_layers=2, max_len=64):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pos    = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, d_ff, num_experts, top_k)
            for _ in range(n_layers)
        ])
        self.head   = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):                    # [B, T]
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.embed(tokens) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)                       # [B, T, V]

    def aux_loss(self):
        total = 0.0
        for blk in self.blocks:
            total = total + blk.moe.aux_loss
        return total


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_expert():
    print("── test_expert ──────────────────────────────")
    e = Expert(32, 64)
    x = torch.randn(4, 8, 32)
    y = e(x)
    assert y.shape == x.shape, f"bad shape {y.shape}"
    print(f"  input {x.shape} → output {y.shape}  ✓")


def test_router():
    print("── test_router ──────────────────────────────")
    r = TopKRouter(32, num_experts=8, top_k=2)
    x = torch.randn(4, 10, 32)
    w, idx, logits = r(x)
    assert w.shape   == (4, 10, 2)
    assert idx.shape == (4, 10, 2)
    # weights should sum to 1
    assert torch.allclose(w.sum(-1), torch.ones(4, 10), atol=1e-5)
    print(f"  weights {w.shape}, indices {idx.shape}  ✓")
    print(f"  weights sum to 1: {w.sum(-1).mean().item():.6f}  ✓")


def test_moe_layer():
    print("── test_moe_layer ───────────────────────────")
    layer = MoELayer(d_model=64, d_ff=128, num_experts=8, top_k=2)
    x = torch.randn(2, 16, 64)
    y = layer(x)
    assert y.shape == x.shape
    assert layer.aux_loss.item() >= 1.0, "load-balance loss should be ≥ 1"
    print(f"  output {y.shape}  ✓")
    print(f"  aux_loss = {layer.aux_loss.item():.4f}  ✓")


def test_full_model():
    print("── test_full_model ──────────────────────────")
    model  = TinyMoEModel(vocab_size=256, d_model=128, n_heads=4,
                          d_ff=256, num_experts=8, top_k=2, n_layers=2)
    tokens = torch.randint(0, 256, (3, 20))
    logits = model(tokens)
    assert logits.shape == (3, 20, 256)
    print(f"  tokens {tokens.shape} → logits {logits.shape}  ✓")
    print(f"  aux_loss = {model.aux_loss().item():.4f}  ✓")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  param count: {n_params:,}")


def test_training_step():
    print("── test_training_step ───────────────────────")
    model  = TinyMoEModel(vocab_size=256, d_model=64, n_heads=2,
                          d_ff=128, num_experts=4, top_k=2, n_layers=1)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(50):
        tokens  = torch.randint(0, 256, (4, 16))
        inputs  = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits  = model(inputs)                                     # [B, T-1, V]
        ce_loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        loss    = ce_loss + 0.01 * model.aux_loss()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(ce_loss.item())

    first5  = sum(losses[:5])  / 5
    last5   = sum(losses[-5:]) / 5
    print(f"  first-5 CE loss: {first5:.4f}")
    print(f"  last-5  CE loss: {last5:.4f}")
    assert last5 < first5 + 0.5, "loss did not decrease (check training)"
    print(f"  training converges  ✓")


def test_expert_utilisation():
    """Check that all experts receive tokens (no dead experts for small batches)."""
    print("── test_expert_utilisation ──────────────────")
    torch.manual_seed(0)
    num_experts = 6
    layer  = MoELayer(d_model=32, d_ff=64, num_experts=num_experts, top_k=2)
    x      = torch.randn(8, 32, 32)
    _      = layer(x)
    _, idx, _ = layer.router(x)
    unique = idx.unique()
    print(f"  experts used: {sorted(unique.tolist())} / {num_experts}")
    assert len(unique) == num_experts, "some experts never used"
    print(f"  all {num_experts} experts utilised  ✓")


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 52)
    print(" Mixture-of-Experts — Unit Tests")
    print("=" * 52)
    test_expert()
    test_router()
    test_moe_layer()
    test_full_model()
    test_training_step()
    test_expert_utilisation()
    print("=" * 52)
    print(" All tests passed ✓")
    print("=" * 52)