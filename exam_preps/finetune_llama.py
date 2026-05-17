"""
train.py — fine-tuning on TinyShakespeare with gradient accumulation
"""

import os, math, time, json, requests, torch, torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from model_exam_preps import ModelArgs, Transformer, apply_rotary_embeddings, repeat_kv

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_TEST      = True   # ← set False on Colab to load real checkpoint
CHECKPOINTS_DIR = "/Users/hossam.amer/Documents/workspace/Llama2_7b_weights" if LOCAL_TEST else "/content"
TOKENIZER_PATH  = f"{CHECKPOINTS_DIR}/tokenizer.model"
DATA_URL        = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH       = "./tinyshakespeare.txt"

SEQ_LEN        = 64
BATCH_SIZE     = 1
GRAD_ACCUM     = 16    # effective batch size = BATCH_SIZE * GRAD_ACCUM = 16
MAX_STEPS      = 200   # each step = GRAD_ACCUM forward passes
EVAL_EVERY     = 10
LR             = 1e-3 if LOCAL_TEST else 5e-6
GRAD_CLIP      = 1.0
DATA_FRACTION  = 0.05
DEVICE         = "cpu" if LOCAL_TEST else ("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print("Downloading TinyShakespeare...")
    with open(DATA_PATH, "w") as f:
        f.write(requests.get(DATA_URL).text)

def load_data(tokenizer: SentencePieceProcessor):
    with open(DATA_PATH, "r") as f:
        raw = f.read()
    # Use a small fraction to fit in Colab memory
    raw = raw[: int(len(raw) * DATA_FRACTION)]
    token_ids = tokenizer.encode(raw, out_type=int)
    data = torch.tensor(token_ids, dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:]   # train, val

def get_batch(data: torch.Tensor):
    """Return (x, y) each of shape (BATCH_SIZE, SEQ_LEN)."""
    ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x  = torch.stack([data[i     : i + SEQ_LEN    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_tiny_model(vocab_size: int) -> Transformer:
    """Tiny model for local Mac testing — no checkpoint needed."""
    args = ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        max_batch_size=2,
        max_seq_len=64,
        device="cpu",
    )
    return Transformer(args)

def build_model(tokenizer: SentencePieceProcessor) -> Transformer:
    with open(Path(CHECKPOINTS_DIR) / "params.json") as f:
        params = json.load(f)

    args = ModelArgs(
        max_seq_len=SEQ_LEN,
        max_batch_size=BATCH_SIZE,
        device=DEVICE,
        **params,
    )
    args.vocab_size = tokenizer.vocab_size()

    if DEVICE == "cuda":
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.bfloat16)

    model = Transformer(args).to(DEVICE)

    # Load pretrained weights
    import glob as _glob
    ckpts = sorted(_glob.glob(f"{CHECKPOINTS_DIR}/*.pth"))
    assert ckpts, f"No .pth files found in {CHECKPOINTS_DIR}"
    print(f"Loading checkpoint: {ckpts[0]}")
    state_dict = torch.load(ckpts[0], map_location="cpu", mmap=True)
    state_dict.pop("rope.freqs", None)
    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded.")

    torch.set_default_dtype(torch.float32)
    return model

# ── Optimizer ─────────────────────────────────────────────────────────────────
def build_optimizer(model: Transformer):
    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        fused=(DEVICE == "cuda"),  # fused kernel only available on CUDA
    )

# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    tokenizer = SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    train_data, val_data = load_data(tokenizer)
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    model  = build_tiny_model(tokenizer.vocab_size()) if LOCAL_TEST else build_model(tokenizer)
    optim  = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    print("Model built and optimizer ready.")
    print(f"Model: {model}")
    print("Starting training...")
    
    model.train()
    optim.zero_grad()

    train_losses, val_losses, log_steps = [], [], []

    t0 = time.time()
    for step in range(MAX_STEPS):
        # ── gradient accumulation ────────────────────────────────────────────
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(train_data)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
                logits = model.forward_train(x)                     # (B, T, V)
                loss   = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),               # (B*T, V)
                    y.view(-1),                                     # (B*T,)
                )
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # ── optimizer step ───────────────────────────────────────────────────
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        # ── logging ──────────────────────────────────────────────────────────
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                xv, yv   = get_batch(val_data)
                vlogits  = model.forward_train(xv)
                val_loss = nn.functional.cross_entropy(
                    vlogits.view(-1, vlogits.size(-1)), yv.view(-1)
                ).item()
            model.train()

            train_losses.append(accum_loss)
            val_losses.append(val_loss)
            log_steps.append(step)

            elapsed = time.time() - t0
            print(f"step {step:4d} | train_loss {accum_loss:.4f} | val_loss {val_loss:.4f} | {elapsed:.1f}s")

    print("Training done.")
    plot_losses(log_steps, train_losses, val_losses)

def plot_losses(steps, train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_losses, label="train")
    plt.plot(steps, val_losses,   label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("losses.png", dpi=150)
    plt.show()
    print("Plot saved to losses.png")

if __name__ == "__main__":
    train()
