import torch
import math

def get_lr(step: int, warmup_steps: int, total_steps: int,
           lr_max: float, lr_min: float = 0.0) -> float:
    """
    Linear warmup then cosine annealing.

    step          : current training step (0-indexed)
    warmup_steps  : how many steps to ramp up
    total_steps   : total training steps
    lr_max        : peak learning rate (reached at end of warmup)
    lr_min        : floor learning rate (reached at end of cosine decay)
    """
    # warmup phase
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)

    # cosine annealing phase
    # cosine progress: 0 → 1 over the course of the cosine phase
    progress = (step - warmup_steps) / (total_steps - warmup_steps)  # 0 → 1
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def apply_lr(optimizer: torch.optim.Optimizer, lr: float):
    for group in optimizer.param_groups:
        group['lr'] = lr


# ── sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    warmup_steps = 100
    total_steps  = 1000
    lr_max       = 1e-3
    lr_min       = 1e-5

    # spot check key points
    print(f"step   0 (start of warmup) : {get_lr(0,   warmup_steps, total_steps, lr_max, lr_min):.6f}  expect ~0")
    print(f"step  50 (mid warmup)      : {get_lr(50,  warmup_steps, total_steps, lr_max, lr_min):.6f}  expect {lr_max*0.5:.6f}")
    print(f"step 100 (peak)            : {get_lr(100, warmup_steps, total_steps, lr_max, lr_min):.6f}  expect {lr_max:.6f}")
    print(f"step 550 (mid cosine)      : {get_lr(550, warmup_steps, total_steps, lr_max, lr_min):.6f}  expect ~{(lr_max+lr_min)/2:.6f}")
    print(f"step 999 (end)             : {get_lr(999, warmup_steps, total_steps, lr_max, lr_min):.6f}  expect ~{lr_min:.6f}")

    # training loop usage
    model     = torch.nn.Linear(8, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)

    for step in range(total_steps):
        lr = get_lr(step, warmup_steps, total_steps, lr_max, lr_min)
        apply_lr(optimizer, lr)

        x    = torch.randn(4, 8)
        loss = model(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"step {step:4d} | lr {lr:.6f} | loss {loss.item():.4f}")