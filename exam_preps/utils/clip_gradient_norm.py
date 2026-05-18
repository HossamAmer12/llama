'''
Clip Gradient Norm Functions

There are two common ways to clip gradients in deep learning:
1. Clip by value: clip each individual gradient element to a specified range (e.g. [-1, 1]).
2. Clip by global norm: compute the global norm of all gradients and scale them down if the norm exceeds a threshold.

Here, we are implementing by global norm, which is more common and generally more effective for training stability. 
The function `clip_grad_global_norm` computes the global norm of the gradients and scales them down 
if they exceed the specified `max_norm`. 
It also returns the global norm before clipping, which can be useful for logging and monitoring during training.

'''


import torch

def clip_grad_global_norm(parameters, max_norm: float, eps: float = 1e-6) -> float:
    """
    parameters : iterable of tensors with .grad populated (e.g. model.parameters())
    max_norm   : threshold
    returns    : global norm before clipping (useful for logging)
    """
    # collect all gradients that exist
    grads = [p.grad for p in parameters if p.grad is not None]

    # global norm: sqrt of sum of squared L2 norms across all param grads
    global_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    # only clip if norm exceeds threshold
    if global_norm > max_norm:
        scale = max_norm / (global_norm + eps)
        for g in grads:
            g.mul_(scale)           # in-place to modify .grad directly

    return global_norm.item()


# ── sanity check against torch.nn.utils.clip_grad_norm_ ──────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    # two parameter tensors with known gradients
    p1 = torch.nn.Parameter(torch.randn(4, 4))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    p1.grad = torch.randn(4, 4)
    p2.grad = torch.randn(4, 4)

    # save original grads
    g1_orig = p1.grad.clone()
    g2_orig = p2.grad.clone()

    max_norm = 1.0

    # ours
    norm_before = clip_grad_global_norm([p1, p2], max_norm)
    g1_ours = p1.grad.clone()
    g2_ours = p2.grad.clone()

    # restore and run reference
    p1.grad.copy_(g1_orig)
    p2.grad.copy_(g2_orig)
    norm_ref = torch.nn.utils.clip_grad_norm_([p1, p2], max_norm)

    print(f"global norm before clipping : {norm_before:.6f}")
    print(f"g1 err : {(g1_ours - p1.grad).abs().max().item():.2e}")
    print(f"g2 err : {(g2_ours - p2.grad).abs().max().item():.2e}")
    
    print(f"reference norm before clipping : {norm_ref:.6f}")
    print(f"norms match : {abs(norm_before - norm_ref) < 1e-6}")