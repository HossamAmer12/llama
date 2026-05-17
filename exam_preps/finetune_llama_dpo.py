"""
finetune_llama_dpo.py — DPO fine-tuning on math preference pairs
"""

import copy, time, json, glob as _glob, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from model_exam_preps import ModelArgs, Transformer
from preference_data import load_preference_data, get_preference_batch

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_TEST      = True   # ← set False on Colab to load real checkpoint
CHECKPOINTS_DIR = "/Users/hossam.amer/Documents/workspace/Llama2_7b_weights" if LOCAL_TEST else "/content"
TOKENIZER_PATH  = f"{CHECKPOINTS_DIR}/tokenizer.model"

BATCH_SIZE  = 1
GRAD_ACCUM  = 4
MAX_STEPS   = 200
EVAL_EVERY  = 10
LR          = 1e-3 if LOCAL_TEST else 5e-6
GRAD_CLIP   = 1.0
BETA        = 0.1   # DPO temperature — controls how far policy drifts from reference
DEVICE      = "cpu" if LOCAL_TEST else ("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_tiny_model(vocab_size: int) -> Transformer:
    args = ModelArgs(
        dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=vocab_size, max_batch_size=2, max_seq_len=64, device="cpu",
    )
    return Transformer(args)

def build_model(tokenizer: SentencePieceProcessor) -> Transformer:
    with open(Path(CHECKPOINTS_DIR) / "params.json") as f:
        params = json.load(f)
    args = ModelArgs(max_seq_len=64, max_batch_size=BATCH_SIZE, device=DEVICE, **params)
    args.vocab_size = tokenizer.vocab_size()

    torch.set_default_dtype(torch.float16 if DEVICE == "cuda" else torch.bfloat16)
    model = Transformer(args).to(DEVICE)

    ckpts = sorted(_glob.glob(f"{CHECKPOINTS_DIR}/*.pth"))
    assert ckpts, f"No .pth files found in {CHECKPOINTS_DIR}"
    state_dict = torch.load(ckpts[0], map_location="cpu", mmap=True)
    state_dict.pop("rope.freqs", None)
    model.load_state_dict(state_dict, strict=False)

    torch.set_default_dtype(torch.float32)
    return model

# ── DPO core ──────────────────────────────────────────────────────────────────
def get_response_log_probs(model: Transformer, input_ids: torch.Tensor, response_start_idx: int):
    """Sum of log-probs over response tokens only.

    input_ids : (B, T)
    returns   : (B,)
    """
    logits = model.forward_train(input_ids)          # (B, T, V)
    logits = logits[:, :-1, :].float()               # (B, T-1, V)  predict next token
    
    # Shifted labels for next-token prediction
    labels = input_ids[:, 1:]                        # (B, T-1)

    # mask to response tokens only
    logits = logits[:, response_start_idx:, :]
    labels = labels[:, response_start_idx:]

    log_probs        = F.log_softmax(logits, dim=-1)
    token_log_probs  = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, resp_len)
    return token_log_probs.sum(-1)                   # (B,)


def dpo_loss(policy, ref_model, chosen, rejected, response_start_idx):
    """DPO loss: -log σ(β * (log π(yw|x)/πref(yw|x) - log π(yl|x)/πref(yl|x)))

    Returns loss, chosen_reward, rejected_reward (last two for logging).
    """
    log_p_w = get_response_log_probs(policy,    chosen,   response_start_idx)
    log_p_l = get_response_log_probs(policy,    rejected, response_start_idx)

    with torch.no_grad():
        log_r_w = get_response_log_probs(ref_model, chosen,   response_start_idx)
        log_r_l = get_response_log_probs(ref_model, rejected, response_start_idx)

    chosen_reward   = BETA * (log_p_w - log_r_w)
    rejected_reward = BETA * (log_p_l - log_r_l)

    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward.mean().item(), rejected_reward.mean().item()

# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    tokenizer = SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    data = load_preference_data()
    print(f"Preference pairs: {len(data)}")

    policy = build_tiny_model(tokenizer.vocab_size()) if LOCAL_TEST else build_model(tokenizer)

    # Reference model: same weights, fully frozen
    ref_model = copy.deepcopy(policy)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    optim  = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.01,
                                fused=(DEVICE == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    train_losses, chosen_rewards_log, rejected_rewards_log, log_steps = [], [], [], []

    policy.train()
    optim.zero_grad()
    t0 = time.time()

    for step in range(MAX_STEPS):
        accum_loss = accum_chosen = accum_rejected = 0.0

        for _ in range(GRAD_ACCUM):
            chosen, rejected, response_start_idx = get_preference_batch(
                data, tokenizer, BATCH_SIZE, DEVICE
            )

            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
                loss, c_rew, r_rew = dpo_loss(policy, ref_model, chosen, rejected, response_start_idx)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss     += loss.item()
            accum_chosen   += c_rew / GRAD_ACCUM
            accum_rejected += r_rew / GRAD_ACCUM

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if step % EVAL_EVERY == 0:
            train_losses.append(accum_loss)
            chosen_rewards_log.append(accum_chosen)
            rejected_rewards_log.append(accum_rejected)
            log_steps.append(step)

            margin  = accum_chosen - accum_rejected
            elapsed = time.time() - t0
            print(f"step {step:4d} | loss {accum_loss:.4f} | "
                  f"chosen_rew {accum_chosen:+.3f} | rejected_rew {accum_rejected:+.3f} | "
                  f"margin {margin:+.3f} | {elapsed:.1f}s")

    print("Training done.")
    plot_losses(log_steps, train_losses, chosen_rewards_log, rejected_rewards_log)

    # ── Eval: ref model (before DPO) vs policy (after DPO) ───────────────────
    eval_prompts = [
        "Q: A car travels at 30 mph for 4 hours. How far does it travel?\nA:",
        "Q: Oranges cost $3 each. How much do 6 oranges cost?\nA:",
        "Q: A room is 5 m long and 4 m wide. What is its area?\nA:",
        "Q: You invest $1000 at 5% annual interest for 3 years. How much interest?\nA:",
        "Q: A box has 6 rows with 8 items each. How many items total?\nA:",
    ]
    answers = [" 120 miles", " $18", " 20 square meters", " $150", " 48 items"]

    ref_acc  = evaluate(ref_model, "Reference (before DPO)", tokenizer, eval_prompts, answers)
    pol_acc  = evaluate(policy,    "Policy   (after  DPO)", tokenizer, eval_prompts, answers)
    print(f"\nAccuracy — Reference: {ref_acc}/{len(answers)}  |  Policy: {pol_acc}/{len(answers)}")


# Generate for one prompt using greedy decoding, used in evaluation
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    """Greedy generation — appends one token at a time using forward_train."""
    model.eval()
    input_ids = tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits     = model.forward_train(tokens)          # (1, T, V)
            next_token = logits[0, -1, :].argmax(-1)          # scalar
            if next_token.item() == tokenizer.eos_id():
                break
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=-1)

    generated_ids = tokens[0, len(input_ids):].tolist()
    return tokenizer.decode(generated_ids)


def evaluate(model, model_name: str, tokenizer, prompts: list, answers: list) -> int:
    """Run greedy generation on each prompt, print response vs ground truth, return accuracy count."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    correct = 0
    for prompt, answer in zip(prompts, answers):
        response = generate(model, tokenizer, prompt)
        hit      = answer.strip().lower() in response.strip().lower()
        correct += int(hit)
        print(f"\nPrompt  : {prompt.strip()}")
        print(f"Expected: {answer.strip()}")
        print(f"Got     : {response.strip()}")
        print(f"Correct : {'YES' if hit else 'NO'}")

    model.train()
    return correct


def plot_losses(steps, losses, chosen_rewards, rejected_rewards):
    margins = [c - r for c, r in zip(chosen_rewards, rejected_rewards)]

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    ax1.plot(steps, losses)
    ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.set_title("DPO Loss")

    ax2.plot(steps, chosen_rewards,   label="chosen")
    ax2.plot(steps, rejected_rewards, label="rejected")
    ax2.set_xlabel("step"); ax2.set_ylabel("reward"); ax2.set_title("Rewards"); ax2.legend()

    ax3.plot(steps, margins, color="green")
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("step"); ax3.set_ylabel("margin"); ax3.set_title("Reward Margin (↑ good)")

    plt.tight_layout()
    plt.savefig("dpo_losses.png", dpi=150)
    plt.show()
    print("Plot saved to dpo_losses.png")


if __name__ == "__main__":
    train()
