"""
finetune_llama_rlhf.py — RLHF pipeline (schematic, runnable locally)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RLHF — 4 phases
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1 — SFT (Supervised Fine-Tuning)
  Collect human demonstrations (prompt → ideal response).
  Fine-tune the base model on them with next-token prediction loss.
  This gives us a "sensible" policy to start RL from.

Phase 2 — Preference Data Collection (offline, shown as a dataset here)
  For each prompt, generate two model responses.
  A human annotator picks which response is better.
  Result: a dataset of (prompt, chosen, rejected) triples.

Phase 3 — Reward Model Training
  Take the SFT model, add a scalar head (linear layer → 1 number).
  Train it on preference pairs with the Bradley-Terry loss:
    loss = -log σ(r_chosen − r_rejected)
  After training, the reward model assigns a quality score to any response.

Phase 4 — PPO (Proximal Policy Optimization)
  Start from the SFT model (policy) + a frozen copy (reference).
  For each prompt:
    a. Sample G responses from the policy.
    b. Score each with the reward model.
    c. Subtract a KL penalty vs the reference: keeps policy from drifting.
       total_reward = r_model_score − β * KL(policy ‖ reference)
    d. Normalize rewards across the group → advantages.
    e. Update policy with PPO-clipped gradient:
       loss = −min(ratio * A,  clip(ratio, 1−ε, 1+ε) * A)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import copy, time, json, glob as _glob, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from model_exam_preps import ModelArgs, Transformer
from preference_data import load_preference_data, get_preference_batch

SEED = 43
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_TEST      = True
CHECKPOINTS_DIR = "/Users/hossam.amer/Documents/workspace/Llama2_7b_weights" if LOCAL_TEST else "/content"
TOKENIZER_PATH  = f"{CHECKPOINTS_DIR}/tokenizer.model"

DEVICE     = "cpu" if LOCAL_TEST else ("cuda" if torch.cuda.is_available() else "cpu")
LR_SFT     = 1e-3 if LOCAL_TEST else 5e-6
LR_RM      = 1e-4 if LOCAL_TEST else 1e-5   # lower than SFT — head overfits fast on small data
LR_PPO     = 1e-3 if LOCAL_TEST else 5e-6
SFT_STEPS  = 50
RM_STEPS   = 30   # stop before margin explodes; ~10-20 steps is enough to learn preference direction
PPO_STEPS  = 50
EVAL_EVERY = 10
GRAD_CLIP  = 1.0
G          = 4       # responses per prompt in PPO
TEMPERATURE = 1.2
MAX_NEW_TOKENS = 20
BETA       = 0.0 if LOCAL_TEST else 0.1   # KL penalty (0 locally — ref is random)
EPS        = 0.2                           # PPO clip range

# ── Phase 1 data — human demonstrations ──────────────────────────────────────
DEMONSTRATIONS = [
    {"prompt": "Q: A car travels at 30 mph for 4 hours. How far?\nA:",    "response": " 120 miles"},
    {"prompt": "Q: Oranges cost $3 each. How much do 6 cost?\nA:",         "response": " $18"},
    {"prompt": "Q: A room is 5 m long and 4 m wide. What is its area?\nA:", "response": " 20 square meters"},
    {"prompt": "Q: You invest $1000 at 5% for 3 years. How much interest?\nA:", "response": " $150"},
    {"prompt": "Q: A box has 6 rows with 8 items each. How many total?\nA:", "response": " 48 items"},
]

# ── Model builders ────────────────────────────────────────────────────────────
def build_tiny_model(vocab_size: int) -> Transformer:
    args = ModelArgs(
        dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=vocab_size, max_batch_size=4, max_seq_len=64, device="cpu",
    )
    return Transformer(args)

def build_model(tokenizer: SentencePieceProcessor) -> Transformer:
    with open(Path(CHECKPOINTS_DIR) / "params.json") as f:
        params = json.load(f)
    args = ModelArgs(max_seq_len=64, max_batch_size=4, device=DEVICE, **params)
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

# ── Reward model ──────────────────────────────────────────────────────────────
class RewardModel(nn.Module):
    """LM backbone + scalar head.

    Takes the last-token logit vector and maps it to a single reward score.
    In production: would use the last hidden state rather than logits.
    """
    def __init__(self, backbone: Transformer, vocab_size: int):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(vocab_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns scalar reward per sequence. Shape: (B,)"""
        with torch.no_grad():
            logits      = self.backbone.forward_train(input_ids)  # (B, T, V) — backbone frozen
        last_logits = logits[:, -1, :].float()                    # (B, V)
        return self.head(last_logits).squeeze(-1)                 # (B,)

# ── Shared generation helpers ─────────────────────────────────────────────────
def sample_response(model, prompt_tokens: torch.Tensor, tokenizer):
    """Sample one response. Returns (full_seq, old_log_probs_sum)."""
    tokens        = prompt_tokens.clone()
    old_log_probs = []
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits     = model.forward_train(tokens)
            last_logit = logits[0, -1]
            next_tok   = torch.multinomial(F.softmax(last_logit / TEMPERATURE, dim=-1), 1)
            log_p      = F.log_softmax(last_logit, dim=-1).gather(0, next_tok)
            old_log_probs.append(log_p)
            tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=-1)
            if next_tok.item() == tokenizer.eos_id():
                break
    return tokens, torch.cat(old_log_probs).sum()   # (1, T), scalar

def get_response_log_probs(model, full_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Sum of log-probs over response tokens only (with grad). Returns scalar."""
    logits = model.forward_train(full_ids)           # (1, T, V)
    logits = logits[:, :-1, :].float()               # (1, T-1, V)
    labels = full_ids[:, 1:]                         # (1, T-1)
    logits = logits[:, response_start - 1:, :]
    labels = labels[:, response_start - 1:]
    log_probs       = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(-1).squeeze(0)        # scalar

# ── Phase 1: SFT ──────────────────────────────────────────────────────────────
def sft_train(model: Transformer, tokenizer: SentencePieceProcessor) -> Transformer:
    """Fine-tune on human demonstrations via next-token prediction."""
    print(f"\n{'='*60}\n  Phase 1 — SFT\n{'='*60}")
    optim = torch.optim.AdamW(model.parameters(), lr=LR_SFT, weight_decay=0.01)
    model.train()
    t0 = time.time()

    for step in range(SFT_STEPS):
        ex         = DEMONSTRATIONS[step % len(DEMONSTRATIONS)]
        prompt_ids = tokenizer.encode(ex["prompt"],   out_type=int, add_bos=True,  add_eos=False)
        resp_ids   = tokenizer.encode(ex["response"], out_type=int, add_bos=False, add_eos=True)
        input_ids  = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device=DEVICE)
        resp_start = len(prompt_ids)

        logits = model.forward_train(input_ids)          # (1, T, V)
        # Compute loss on response tokens only
        logits = logits[:, resp_start - 1:-1, :].float() # (1, R, V)
        labels = input_ids[:, resp_start:]               # (1, R)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        if step % EVAL_EVERY == 0:
            print(f"  step {step:3d} | sft_loss {loss.item():.4f} | {time.time()-t0:.1f}s")

    print("  SFT done.")
    return model

# ── Phase 3: Reward model training ────────────────────────────────────────────
def train_reward_model(reward_model: RewardModel,
                       tokenizer: SentencePieceProcessor,
                       data: list) -> RewardModel:
    """Train reward model on preference pairs.

    Bradley-Terry loss: loss = -log σ(r_chosen − r_rejected)
    Chosen response should score higher than rejected response.
    """
    print(f"\n{'='*60}\n  Phase 3 — Reward Model Training\n{'='*60}")
    # Only train the scalar head — backbone is a frozen feature extractor.
    # Training the backbone too causes unbounded margin growth and divergence.
    optim = torch.optim.AdamW(reward_model.head.parameters(), lr=LR_RM, weight_decay=0.01)
    reward_model.train()
    t0 = time.time()

    for step in range(RM_STEPS):
        chosen, rejected, _ = get_preference_batch(data, tokenizer, batch_size=1, device=DEVICE)

        r_w = reward_model(chosen)    # (1,) — score for chosen response
        r_l = reward_model(rejected)  # (1,) — score for rejected response

        # Bradley-Terry: preferred response must outscore the rejected one
        loss = -F.logsigmoid(r_w - r_l).mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), GRAD_CLIP)
        optim.step()

        if step % EVAL_EVERY == 0:
            margin = (r_w - r_l).mean().item()
            print(f"  step {step:3d} | rm_loss {loss.item():.4f} | "
                  f"margin {margin:+.3f} | {time.time()-t0:.1f}s")

    print("  Reward model training done.")
    return reward_model

# ── Phase 4: PPO ──────────────────────────────────────────────────────────────
def ppo_loss(policy: Transformer, ref_model: Transformer,
             reward_model: RewardModel, tokenizer: SentencePieceProcessor,
             example: dict):
    """
    One PPO update step for a single prompt:
      a. Sample G responses from policy, record old log-probs.
      b. Score each response with the reward model.
      c. KL penalty: β * (new_log_prob − ref_log_prob) keeps policy near reference.
      d. total_reward = rm_score − β * KL  → group-normalise → advantages.
      e. PPO-clipped loss: −min(ratio * A,  clip(ratio, 1−ε, 1+ε) * A)
    """
    prompt       = example["prompt"]
    prompt_ids   = tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    prompt_len   = len(prompt_ids)

    # ── a. Generate G responses ───────────────────────────────────────────────
    samples = [sample_response(policy, prompt_tensor, tokenizer) for _ in range(G)]
    full_seqs, old_log_probs = zip(*samples)

    # ── b. Reward model scores ────────────────────────────────────────────────
    reward_model.eval()
    with torch.no_grad():
        rm_scores = torch.tensor(
            [reward_model(seq).item() for seq in full_seqs], dtype=torch.float32
        )
    reward_model.train()

    # Length penalty: discourage reward hacking via empty responses.
    # Without this, PPO learns to output EOS immediately because the RM
    # still assigns a positive score to prompt-only inputs.
    response_lens = torch.tensor(
        [seq.shape[1] - prompt_len for seq in full_seqs], dtype=torch.float32
    )
    rm_scores = rm_scores + torch.where(response_lens < 2, torch.tensor(-50.0), torch.tensor(0.0))

    # ── c. KL penalty — computed per response inside the loop below ───────────
    # ── d. total_reward = rm_score − β*KL → advantages ───────────────────────
    # (we'll accumulate KL-adjusted rewards, then normalise)
    kl_penalties = []
    new_log_probs = []
    ref_log_probs_list = []

    policy.train()
    for seq in full_seqs:
        if seq.shape[1] <= prompt_len:
            kl_penalties.append(torch.tensor(0.0))
            new_log_probs.append(torch.tensor(0.0))
            ref_log_probs_list.append(torch.tensor(0.0))
            continue
        new_lp = get_response_log_probs(policy,    seq, prompt_len)          # with grad
        with torch.no_grad():
            ref_lp = get_response_log_probs(ref_model, seq, prompt_len)      # no grad
        kl_penalties.append(new_lp - ref_lp)
        new_log_probs.append(new_lp)
        ref_log_probs_list.append(ref_lp)

    kl_tensor   = torch.stack([k.detach() for k in kl_penalties])
    total_rewards = rm_scores.to(DEVICE) - BETA * kl_tensor

    if total_rewards.std() < 1e-4:
        return torch.tensor(0.0), rm_scores.tolist()

    advantages = (total_rewards - total_rewards.mean()) / (total_rewards.std() + 1e-8)

    # ── e. PPO-clipped loss ───────────────────────────────────────────────────
    total_loss = torch.tensor(0.0, device=DEVICE)
    for i, (new_lp, old_lp) in enumerate(zip(new_log_probs, old_log_probs)):
        if not isinstance(new_lp, torch.Tensor) or new_lp.grad_fn is None:
            continue
        ratio   = torch.exp(new_lp - old_lp.detach())
        A_i     = advantages[i].to(DEVICE)
        clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS)
        pg_loss = -torch.min(ratio * A_i, clipped * A_i)
        total_loss = total_loss + pg_loss + BETA * kl_penalties[i]

    return total_loss / G, rm_scores.tolist()


def ppo_train(policy: Transformer, ref_model: Transformer,
              reward_model: RewardModel, tokenizer: SentencePieceProcessor,
              data: list):
    print(f"\n{'='*60}\n  Phase 4 — PPO\n{'='*60}")
    optim  = torch.optim.AdamW(policy.parameters(), lr=LR_PPO, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    t0 = time.time()

    policy.train()
    optim.zero_grad()

    for step in range(PPO_STEPS):
        example = data[torch.randint(0, len(data), (1,)).item()]

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            loss, rm_scores = ppo_loss(policy, ref_model, reward_model, tokenizer, example)

        if loss.grad_fn is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
        optim.zero_grad()

        if step % EVAL_EVERY == 0:
            mean_r = sum(rm_scores) / len(rm_scores)
            print(f"  step {step:3d} | ppo_loss {loss.item():.4f} | "
                  f"mean_rm_score {mean_r:.3f} | {time.time()-t0:.1f}s")

    print("  PPO done.")

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    "Q: A car travels at 30 mph for 4 hours. How far does it travel?\nA:",
    "Q: Oranges cost $3 each. How much do 6 oranges cost?\nA:",
    "Q: A room is 5 m long and 4 m wide. What is its area?\nA:",
    "Q: You invest $1000 at 5% annual interest for 3 years. How much interest?\nA:",
    "Q: A box has 6 rows with 8 items each. How many items total?\nA:",
]
EVAL_ANSWERS = [" 120 miles", " $18", " 20 square meters", " $150", " 48 items"]


def generate(model: Transformer, tokenizer: SentencePieceProcessor,
             prompt: str, max_new_tokens: int = 20) -> str:
    """Greedy decoding — used for evaluation.

    EOS is only respected after at least one real token has been generated.
    Tiny random models often have high EOS logit, which would produce empty
    strings and make SFT vs RLHF indistinguishable in evaluation.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
    tokens    = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_logits = model.forward_train(tokens)[0, -1, :].clone()
            if LOCAL_TEST:
                # Random models have high EOS/BOS logits — suppress them so
                # evaluation produces visible output to compare SFT vs RLHF.
                next_logits[tokenizer.eos_id()] = float('-inf')
                next_logits[tokenizer.bos_id()] = float('-inf')
            next_token = next_logits.argmax(-1)
            if next_token.item() == tokenizer.eos_id():
                break
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=-1)
    model.train()
    return tokenizer.decode(tokens[0, len(input_ids):].tolist())


def evaluate(model: Transformer, model_name: str,
             tokenizer: SentencePieceProcessor,
             reward_model: RewardModel) -> int:
    """Greedy-decode each eval prompt, print response vs expected, score with reward model."""
    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    correct = 0
    for prompt, answer in zip(EVAL_PROMPTS, EVAL_ANSWERS):
        response  = generate(model, tokenizer, prompt)
        hit       = answer.strip().lower() in response.strip().lower()
        correct  += int(hit)

        # Score the response with the reward model (only when non-empty)
        if response.strip():
            input_ids  = tokenizer.encode(prompt + response, out_type=int,
                                          add_bos=True, add_eos=True)
            ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                rm_score_str = f"{reward_model(ids_tensor).item():.3f}"
        else:
            rm_score_str = "N/A (empty response)"

        print(f"\n  Prompt  : {prompt.strip()}")
        print(f"  Expected: {answer.strip()}")
        print(f"  Got     : {response.strip()}")
        print(f"  RM score: {rm_score_str}   Correct: {'YES' if hit else 'NO'}")
    return correct


# ── Main: wire the 4 phases together ─────────────────────────────────────────
def main():
    tokenizer = SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    # ── Phase 1: SFT ─────────────────────────────────────────────────────────
    base_model = build_tiny_model(tokenizer.vocab_size()) if LOCAL_TEST else build_model(tokenizer)
    sft_train(base_model, tokenizer)

    # ── Phase 2: preference data (collected offline by human annotators) ──────
    print(f"\n{'='*60}\n  Phase 2 — Preference Data\n{'='*60}")
    pref_data = load_preference_data()
    print(f"  {len(pref_data)} preference pairs loaded (chosen / rejected).")

    # ── Phase 3: reward model ─────────────────────────────────────────────────
    rm_backbone  = copy.deepcopy(base_model)
    reward_model = RewardModel(rm_backbone, tokenizer.vocab_size()).to(DEVICE)
    train_reward_model(reward_model, tokenizer, pref_data)
    for p in reward_model.parameters():
        p.requires_grad = False
    reward_model.eval()

    # ── Phase 4: PPO ──────────────────────────────────────────────────────────
    policy    = copy.deepcopy(base_model)
    ref_model = copy.deepcopy(base_model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    ppo_train(policy, ref_model, reward_model, tokenizer, pref_data)

    # ── Eval: SFT baseline vs RLHF policy ────────────────────────────────────
    sft_acc = evaluate(ref_model, "SFT baseline (before PPO)", tokenizer, reward_model)
    ppo_acc = evaluate(policy,    "RLHF policy  (after  PPO)", tokenizer, reward_model)
    n = len(EVAL_ANSWERS)
    print(f"\nAccuracy — SFT: {sft_acc}/{n}  |  RLHF: {ppo_acc}/{n}")
    print("\nRLHF pipeline complete.")


if __name__ == "__main__":
    main()
