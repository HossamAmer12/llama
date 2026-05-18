"""
Dropout — correct implementation with inverted scaling

Two modes:
  Training : randomly zero out each unit with probability p,
             then SCALE UP survivors by 1/(1-p) so the
             expected value of each unit stays the same.
             → called "inverted dropout" (used by PyTorch, most frameworks)

  Eval     : no masking, no scaling — pass through unchanged.
             The scale-up during training already compensated, so
             inference requires zero code changes.

Why scale during training (not eval)?
  Without scaling, dropping p fraction of units reduces the expected
  activation magnitude by (1-p). At eval time with all units active,
  the layer would suddenly see ~1/(1-p)x larger activations than during
  training → distribution shift → degraded performance.
  Scaling UP during training keeps the expected value constant in both modes.
"""

import torch
import torch.nn as nn


# ── Manual inverted dropout ────────────────────────────────────────────────────
def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    Args:
        x        : input tensor, any shape
        p        : probability of zeroing a unit  (0 = no dropout, 1 = zero everything)
        training : True during training, False during eval/inference

    Returns tensor with same shape and same expected value as x.
    """
    if not training or p == 0.0:
        return x                              # eval: pass through unchanged

    # Bernoulli mask: 1 with prob (1-p), 0 with prob p
    # torch.rand_like(x) generates uniform random numbers in [0, 1) with same shape as x
    
    rand_generator = torch.rand_like(x)
    print(f"Random values for dropout mask:\n{rand_generator}")
    mask  = (rand_generator > p).float()

    # Inverted scaling: divide by (1-p) so E[output] = E[input]
    # If p=0.4, then we keep 60% of units, so we scale survivors by 1/0.6 = 1.6667
    # Without scale:
    # training → expected value 1.0
    # eval → always 2.0
    # mismatch — next layer was trained on inputs averaging 1.0 but at inference gets 2.0

    # With scale:
    # training → expected value 2.0
    # eval → always 2.0
    # consistent — next layer sees the same magnitude in both cases
    scale = 1.0 / (1.0 - p)

    return x * mask * scale


# ── PyTorch nn.Dropout does exactly this ──────────────────────────────────────
class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "p must be in [0, 1)"
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dropout(x, self.p, self.training)
        # self.training is set automatically by model.train() / model.eval()


# ── Sanity checks ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.ones(10_000)   # all 1s → easy to check expected value
    p = 0.4

    # ── Training mode ─────────────────────────────────────────────────────────
    out_train = dropout(x, p=p, training=True)
    frac_zero = (out_train == 0).float().mean().item()
    mean_val  = out_train.mean().item()

    print("=== Training mode ===")
    print(f"  p = {p}  →  ~{p*100:.0f}% of units zeroed")
    print(f"  Fraction actually zeroed : {frac_zero:.3f}  (expected {p})")
    print(f"  Mean of output           : {mean_val:.4f}  (expected 1.0000)")
    print(f"  Scale factor applied     : {1/(1-p):.4f}")

    # ── Eval mode ─────────────────────────────────────────────────────────────
    out_eval = dropout(x, p=p, training=False)
    print("\n=== Eval mode ===")
    print(f"  Mean of output : {out_eval.mean().item():.4f}  (expected 1.0000, no scaling)")
    print(f"  Any zeros?     : {(out_eval == 0).any().item()}")

    # ── nn.Module version ─────────────────────────────────────────────────────
    print("\n=== nn.Module version ===")
    layer = Dropout(p=0.4)

    layer.train()
    out = layer(x)
    print(f"  train()  mean={out.mean():.4f}  zeros={( out==0).float().mean():.3f}")

    layer.eval()
    out = layer(x)
    print(f"  eval()   mean={out.mean():.4f}  zeros={(out==0).float().mean():.3f}")
