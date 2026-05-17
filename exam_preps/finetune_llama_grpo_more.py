"""
finetune_llama_grpo.py — GRPO fine-tuning on math preference pairs
# The other code is working on local setup
# This file is being edited to run on Colab with the real Llama 2 7b checkpoint, which is too large to load twice for DPO (ref + policy).
# So this GRPO version only keeps one model (no frozen ref), and uses the current policy for both generation and evaluation of log-probs. This is a bit less principled than DPO, but still captures the core idea of group-normalised rewards and importance sampling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GRPO Algorithm — step by step (one training iteration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Sample a prompt from the dataset
          Pick one (prompt, correct_answer) pair at random.

Step 2 — Generate a group of G responses
          Run the current policy (actor) with temperature sampling
          to produce G diverse candidate responses.
          At each token, record the log-prob → these are the OLD log-probs
          used later as the importance-weight denominator.
          Stop each response at EOS or MAX_NEW_TOKENS.

Step 3 — Score each response with the reward function
          For each of the G responses compute:
            r = 1.0  if correct number AND correct unit
            r = 0.5  if correct number OR correct unit
            r = 0.1  if any number was generated (partial credit)
            r = 0.0  if no number found

Step 4 — Compute group-normalised advantages
          μ = mean(rewards),  σ = std(rewards)
          A_i = (r_i − μ) / (σ + ε)
          This turns raw rewards into a zero-centred signal:
          responses above the group average get positive advantage,
          below average get negative.
          If σ ≈ 0 (all responses got the same reward) → skip, no signal.

Step 5 — Recompute log-probs WITH gradients (new policy)
          For each response, run a fresh forward pass through the actor
          (now with grad enabled) to get new_log_prob = Σ log π(y_t | x, y<t)
          over response tokens only.

Step 6 — Compute importance ratio
          ratio = exp(new_log_prob − old_log_prob)
          This corrects for the distribution shift between the policy
          at generation time and the policy at update time.

Step 7 — PPO-clipped policy gradient loss
          Clipping prevents a single response from causing a huge update:
          pg_loss = −min(ratio × A_i,  clip(ratio, 1−ε, 1+ε) × A_i)

Step 8 — KL penalty against frozen reference
          kl = new_log_prob − ref_log_prob
          Keeps the policy from drifting too far from pretrained weights.
          item_loss = pg_loss + β × kl

Step 9 — Average over the group and backprop
          loss = mean(item_loss_i  for i in 1..G)
          loss.backward() → clip gradients → optimizer step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import re, copy, time, json, glob as _glob, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from model_exam_preps import ModelArgs, Transformer
from preference_data import load_preference_data

SEED   = 43
torch.manual_seed(SEED)


# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_TEST      = True
CHECKPOINTS_DIR = "/Users/hossam.amer/Documents/workspace/Llama2_7b_weights" if LOCAL_TEST else "/content"
TOKENIZER_PATH  = f"{CHECKPOINTS_DIR}/tokenizer.model"

G              = 8      # larger group → more reward diversity
TEMPERATURE    = 1.2    # sampling temperature → more exploration
MAX_NEW_TOKENS = 20     # max response length
BATCH_SIZE     = 1      # prompts per micro-step
GRAD_ACCUM     = 4
MAX_STEPS      = 200
EVAL_EVERY     = 10
LR             = 1e-3 if LOCAL_TEST else 5e-6
GRAD_CLIP      = 1.0
BETA           = 0.0 if LOCAL_TEST else 0.1   # KL penalty — 0 locally (ref is random, no point penalizing)
EPS            = 0.2    # PPO clip range
DEVICE         = "cpu" if LOCAL_TEST else ("cuda" if torch.cuda.is_available() else "cpu")

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

# ── Reward function ───────────────────────────────────────────────────────────
def extract_number(text: str):
    text = text.replace(',', '')
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    return float(match.group()) if match else None

def extract_unit(text: str) -> str:
    if '$' in text:
        return 'dollar'
    text = text.lower()
    for unit in [
        'square meters', 'square feet', 'nautical miles',
        'miles', 'meters', 'feet', 'knots',
        'liters', 'gallons', 'cups',
        'students', 'seats', 'items', 'chairs', 'spots',
        'eggs', 'plants', 'books', 'marbles', 'fruits', 'units', 'pages',
    ]:
        if unit in text:
            return unit
    return ''

def compute_reward(response: str, ground_truth: str) -> float:
    """
    Production (Colab, real Llama2):
      1.0  — number AND unit both match
      0.5  — number matches, unit wrong (or vice versa)
      0.1  — generated any number at all (partial credit, keeps gradient flowing)
      0.0  — no number generated

    LOCAL_TEST proxy (tiny random model):
      Rewards response-token diversity. Random models get trapped in repetition
      loops ("waited waited waited…") — their digit tokens have low initial weights
      and almost never get sampled. Diversity score creates real variance across the
      G samples so GRPO has a gradient signal to work with.
    """
    if LOCAL_TEST:
        tokens = response.split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)   # 0.0 (all same) → 1.0 (all unique)

    pred_num  = extract_number(response)
    true_num  = extract_number(ground_truth)
    pred_unit = extract_unit(response)
    true_unit = extract_unit(ground_truth)

    if pred_num is None:
        return 0.0

    num_match  = (true_num is not None and
                  abs(pred_num - true_num) < 1e-3 * max(abs(true_num), 1.0))
    unit_match = (pred_unit == true_unit)

    if num_match and unit_match:
        return 1.0
    elif num_match or unit_match:
        return 0.5
    return 0.1

# ── Generation & log-probs ────────────────────────────────────────────────────
def sample_response(model, prompt_tokens: torch.Tensor, tokenizer):
    """Sample one response with temperature. Stops at EOS.

    Returns:
        full_seq     : (1, T_prompt + T_response) — full token ids
        old_log_probs: (T_response,) — per-token log-probs from generation (no grad)
                       used as the denominator in the importance ratio
    """
    tokens        = prompt_tokens.clone()
    old_log_probs = []

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits     = model.forward_train(tokens)                        # (1, T, V)
            last_logit = logits[0, -1]                                      # (V,)
            next_token = torch.multinomial(
                F.softmax(last_logit / TEMPERATURE, dim=-1), num_samples=1  # (1,)
            )
            log_p = F.log_softmax(last_logit, dim=-1).gather(0, next_token) # (1,)
            old_log_probs.append(log_p)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)   # (1, T+1)
            if next_token.item() == tokenizer.eos_id():
                break

    old_log_probs = torch.cat(old_log_probs, dim=0)   # (T_response,)
    return tokens, old_log_probs

def get_response_log_probs(model, full_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Sum of log-probs over response tokens. Returns scalar tensor (with grad)."""
    logits = model.forward_train(full_ids)                      # (1, T, V)
    logits = logits[:, :-1, :].float()                          # (1, T-1, V)
    # Shifted labels for next-token prediction
    labels = full_ids[:, 1:]                                    # (1, T-1)

    logits = logits[:, response_start - 1:, :]   # -1: shifted seq, r0 is predicted at R-1
    labels = labels[:, response_start - 1:]

    log_probs       = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(-1).squeeze(0)                   # scalar

# ── GRPO loss ─────────────────────────────────────────────────────────────────
def grpo_loss(model, ref_model, tokenizer, example: dict):
    """
    For one example:
      1. Generate G responses, collect old log-probs from sampling
      2. Score each with compute_reward → advantages
      3. For each response:
           new_log_prob  = log π_actor(y|x)      [with grad]
           ref_log_prob  = log π_ref(y|x)        [no grad]
           ratio         = exp(new - old)         [importance weight]
           pg_loss       = -min(ratio*A, clip(ratio,1-eps,1+eps)*A)
           kl            = new_log_prob - ref_log_prob
           item_loss     = pg_loss + β * kl
      4. Average over group

    Returns loss (scalar), rewards (list), response texts (list).
    """
    prompt       = example["prompt"]
    ground_truth = example["chosen"]

    prompt_ids    = tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    prompt_len    = len(prompt_ids)

    # ── Step 1: generate G responses, capture old log-probs ──────────────────
    samples = [sample_response(model, prompt_tensor, tokenizer) for _ in range(G)]
    full_seqs, old_log_probs_list = zip(*samples)
    texts = [tokenizer.decode(seq[0, prompt_len:].tolist()) for seq in full_seqs]

    # ── Step 2: rewards & advantages ─────────────────────────────────────────
    rewards = torch.tensor([compute_reward(t, ground_truth) for t in texts],
                           dtype=torch.float32)

    if rewards.std() < 1e-4:   # all same reward → no useful gradient signal
        return torch.tensor(0.0), rewards.tolist(), texts

    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # ── Step 3: PPO-clip + KL per response ───────────────────────────────────
    model.train()
    total_loss = 0.0

    for i, (seq, old_lp) in enumerate(zip(full_seqs, old_log_probs_list)):
        if seq.shape[1] <= prompt_len:   # empty response
            continue

        new_log_prob = get_response_log_probs(model,     seq, prompt_len)   # with grad
        with torch.no_grad():
            ref_log_prob = get_response_log_probs(ref_model, seq, prompt_len)

        old_log_prob = old_lp.sum().detach()
        ratio        = torch.exp(new_log_prob - old_log_prob)

        A_i     = advantages[i].to(DEVICE)
        clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS)
        pg_loss = -torch.min(ratio * A_i, clipped * A_i)

        kl   = new_log_prob - ref_log_prob
        total_loss = total_loss + pg_loss + BETA * kl

    loss = total_loss / G
    return loss, rewards.tolist(), texts

# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    tokenizer = SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    data  = load_preference_data()
    print(f"Preference pairs: {len(data)}")

    model = build_tiny_model(tokenizer.vocab_size()) if LOCAL_TEST else build_model(tokenizer)
    # keep a frozen copy only for eval comparison (not used in GRPO loss)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    optim  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                                fused=(DEVICE == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    train_losses, mean_rewards_log, log_steps = [], [], []

    model.train()
    optim.zero_grad()
    t0 = time.time()

    for step in range(MAX_STEPS):
        accum_loss = 0.0
        accum_reward = 0.0

        for _ in range(GRAD_ACCUM):
            example = data[torch.randint(0, len(data), (1,)).item()]

            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
                loss, rewards, _ = grpo_loss(model, ref_model, tokenizer, example)

            if loss.grad_fn is not None:   # skip if all rewards were identical
                scaler.scale(loss / GRAD_ACCUM).backward()
                accum_loss += (loss / GRAD_ACCUM).item()
            accum_reward += sum(rewards) / (len(rewards) * GRAD_ACCUM)

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if step % EVAL_EVERY == 0:
            train_losses.append(accum_loss)
            mean_rewards_log.append(accum_reward)
            log_steps.append(step)

            elapsed = time.time() - t0
            print(f"step {step:4d} | loss {accum_loss:.4f} | "
                  f"mean_reward {accum_reward:.3f} | {elapsed:.1f}s")

    print("Training done.")
    plot_curves(log_steps, train_losses, mean_rewards_log)

    # ── Eval ─────────────────────────────────────────────────────────────────
    eval_prompts = [
        "Q: A car travels at 30 mph for 4 hours. How far does it travel?\nA:",
        "Q: Oranges cost $3 each. How much do 6 oranges cost?\nA:",
        "Q: A room is 5 m long and 4 m wide. What is its area?\nA:",
        "Q: You invest $1000 at 5% annual interest for 3 years. How much interest?\nA:",
        "Q: A box has 6 rows with 8 items each. How many items total?\nA:",
    ]
    answers = [" 120 miles", " $18", " 20 square meters", " $150", " 48 items"]

    ref_acc = evaluate(ref_model, "Reference (before GRPO)", tokenizer, eval_prompts, answers)
    pol_acc = evaluate(model,     "Policy   (after  GRPO)", tokenizer, eval_prompts, answers)
    print(f"\nAccuracy — Reference: {ref_acc}/{len(answers)}  |  Policy: {pol_acc}/{len(answers)}")

# ── Generation for eval (greedy) ──────────────────────────────────────────────
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
    tokens    = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits     = model.forward_train(tokens)
            next_token = logits[0, -1, :].argmax(-1)
            if next_token.item() == tokenizer.eos_id():
                break
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=-1)
    model.train()
    return tokenizer.decode(tokens[0, len(input_ids):].tolist())

def evaluate(model, model_name: str, tokenizer, prompts: list, answers: list) -> int:
    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    correct = 0
    for prompt, answer in zip(prompts, answers):
        response = generate(model, tokenizer, prompt)
        hit      = answer.strip().lower() in response.strip().lower()
        correct += int(hit)
        print(f"\nPrompt  : {prompt.strip()}")
        print(f"Expected: {answer.strip()}")
        print(f"Got     : {response.strip()}")
        print(f"Correct : {'YES' if hit else 'NO'}")
    return correct

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_curves(steps, losses, mean_rewards):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, losses)
    ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.set_title("GRPO Loss")

    ax2.plot(steps, mean_rewards, color="green")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("step"); ax2.set_ylabel("mean reward"); ax2.set_title("Mean Group Reward (↑ good)")

    plt.tight_layout()
    plt.savefig("grpo_curves.png", dpi=150)
    plt.show()
    print("Plot saved to grpo_curves.png")


if __name__ == "__main__":
    train()
