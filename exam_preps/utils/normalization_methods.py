'''Normalizing Functions in PyTorch
- BatchNorm: normalizes across batch dimension, has running stats, used in CNNs
- LayerNorm: normalizes across feature dimension, no running stats, used in Transformers
- RMSNorm: like LayerNorm but skips mean subtraction, cheaper, used in LLaMA
'''


import torch

# ── Batch Normalization ───────────────────────────────────────────────────────
class BatchNormManual:
    """
    x         : (N, C)
    normalizes across N (the batch) for each feature C independently
    running stats updated during training, used during eval
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma   = torch.ones(num_features)
        self.beta    = torch.zeros(num_features)
        self.eps     = eps
        self.momentum = momentum

        # running stats for eval
        self.running_mean = torch.zeros(num_features)
        self.running_var  = torch.ones(num_features)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            # across the batch dimension dim=0, for each feature independently
            mean = x.mean(dim=0)                        # (C,) — across batch
            var  = x.var(dim=0, unbiased=False)         # (C,)

            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (x - mean) / (var + self.eps).sqrt()   # (N, C)
        return self.gamma * x_hat + self.beta


# ── Layer Normalization ───────────────────────────────────────────────────────
class LayerNormManual:
    """
    x         : (N, C)
    normalizes across C (the features) for each sample N independently
    no running stats needed — eval and train are identical
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.gamma = torch.ones(normalized_shape)
        self.beta  = torch.zeros(normalized_shape)
        self.eps   = eps

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        # across the feature dimension dim=1, for each sample independently
        mean  = x.mean(dim=-1, keepdim=True)            # (N, 1)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)  # (N, 1)
        x_hat = (x - mean) / (var + self.eps).sqrt()   # (N, C)
        return self.gamma * x_hat + self.beta


# ── RMS Normalization ─────────────────────────────────────────────────────────
class RMSNormManual:
    """
    x         : (N, C)
    like LayerNorm but skips mean subtraction — cheaper, used in LLaMA/Qwen
    no running stats needed
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.gamma = torch.ones(normalized_shape)
        self.eps   = eps

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        rms   = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()  # (N, 1)
        x_hat = x / rms                                                    # (N, C)
        return self.gamma * x_hat


# ── sanity checks ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    N, C = 6, 8
    x    = torch.randn(N, C)

    # BatchNorm
    bn      = BatchNormManual(C)
    bn_ref  = torch.nn.BatchNorm1d(C, affine=True)
    bn_ref.weight.data  = bn.gamma.clone()
    bn_ref.bias.data    = bn.beta.clone()

    out     = bn.forward(x, training=True)
    out_ref = bn_ref(x)
    print(f"BatchNorm  err (train): {(out - out_ref).abs().max().item():.2e}")

    # eval mode — run a few training steps to build running stats, then compare
    bn_ref.eval()
    out_eval     = bn.forward(x, training=False)
    out_eval_ref = bn_ref(x)
    print(f"BatchNorm  err (eval) : {(out_eval - out_eval_ref).abs().max().item():.2e}")

    # LayerNorm
    ln      = LayerNormManual(C)
    ln_ref  = torch.nn.LayerNorm(C)
    out     = ln.forward(x)
    out_ref = ln_ref(x)
    print(f"LayerNorm  err        : {(out - out_ref).abs().max().item():.2e}")

    # RMSNorm
    rms     = RMSNormManual(C)
    rms_ref = torch.nn.RMSNorm(C)
    out     = rms.forward(x)
    out_ref = rms_ref(x)
    print(f"RMSNorm    err        : {(out - out_ref).abs().max().item():.2e}")