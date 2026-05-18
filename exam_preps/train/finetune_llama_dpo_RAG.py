"""
finetune_llama_dpo_RAG.py — DPO fine-tuning + Retrieval-Augmented Generation

RAG pipeline:
  1. Build a knowledge base of solved math problems
  2. At query time, find the most similar example (TF-IDF cosine similarity)
  3. Prepend the retrieved example to the prompt as a few-shot context
  4. Generate — the model now has a worked example to condition on

Why RAG helps (especially on top of DPO):
  - DPO taught the model the *style* of a good answer (concise, numeric, with unit)
  - RAG provides the *content* — a similar worked example to guide the reasoning
  - Together → style + content → correct answers, even for a small model

4-way evaluation:
  reference + no RAG   vs   reference + RAG
  DPO policy + no RAG  vs   DPO policy + RAG   ← the interesting comparison
"""

import copy, time, json, glob as _glob, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentencepiece import SentencePieceProcessor
from model_exam_preps import ModelArgs, Transformer
from preference_data import load_preference_data, get_preference_batch

SEED = 43
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_TEST      = True
CHECKPOINTS_DIR = "/Users/hossam.amer/Documents/workspace/Llama2_7b_weights" if LOCAL_TEST else "/content"
TOKENIZER_PATH  = f"{CHECKPOINTS_DIR}/tokenizer.model"

BATCH_SIZE  = 1
GRAD_ACCUM  = 4
MAX_STEPS   = 200
EVAL_EVERY  = 10
LR          = 1e-3 if LOCAL_TEST else 5e-6
GRAD_CLIP   = 1.0
BETA        = 0.1
DEVICE      = "cpu" if LOCAL_TEST else ("cuda" if torch.cuda.is_available() else "cpu")
# RAG prompts are longer — increase max_seq_len to fit retrieved context
MAX_SEQ_LEN = 128 if LOCAL_TEST else 256

# ── Knowledge base ────────────────────────────────────────────────────────────
# These are the "documents" RAG retrieves from.
# In production: embeddings stored in a vector DB (FAISS, Pinecone, etc.)
# Here: TF-IDF over question strings — fast, no GPU needed, interpretable.

KNOWLEDGE_BASE = [
    # distance / speed
    {"q": "A train travels at 60 mph for 2 hours. How far does it travel?",   "a": "120 miles"},
    {"q": "A bike rides at 15 mph for 3 hours. How far does it go?",           "a": "45 miles"},
    {"q": "A plane flies at 500 mph for 4 hours. What is the distance?",       "a": "2000 miles"},
    {"q": "A runner runs at 8 mph for 1.5 hours. How far?",                    "a": "12 miles"},
    {"q": "A bus goes at 40 mph for 2.5 hours. How far does it travel?",       "a": "100 miles"},
    # cost / price
    {"q": "Apples cost $2 each. How much do 9 apples cost?",                  "a": "$18"},
    {"q": "Pens cost $4 each. How much do 5 pens cost?",                      "a": "$20"},
    {"q": "Books cost $12 each. How much do 3 books cost?",                   "a": "$36"},
    {"q": "Coffees cost $5 each. How much do 6 coffees cost?",                "a": "$30"},
    {"q": "Tickets cost $8 each. How much do 4 tickets cost?",                "a": "$32"},
    # area
    {"q": "A garden is 6 m long and 3 m wide. What is its area?",             "a": "18 square meters"},
    {"q": "A hall is 10 m long and 4 m wide. What is its area?",              "a": "40 square meters"},
    {"q": "A field is 8 m long and 5 m wide. What is its area?",              "a": "40 square meters"},
    {"q": "A room is 7 m long and 3 m wide. What is its area?",               "a": "21 square meters"},
    {"q": "A court is 9 m by 6 m. What is its area?",                         "a": "54 square meters"},
    # interest
    {"q": "You invest $500 at 10% annual interest for 2 years. How much interest?",  "a": "$100"},
    {"q": "You invest $2000 at 5% for 1 year. How much interest?",                   "a": "$100"},
    {"q": "You invest $800 at 10% for 3 years. How much interest?",                  "a": "$240"},
    {"q": "You deposit $600 at 5% per year for 2 years. What is the interest?",      "a": "$60"},
    # grid / counting
    {"q": "A grid has 4 rows and 7 items each. How many items total?",        "a": "28 items"},
    {"q": "A shelf has 5 rows with 6 books each. How many books total?",      "a": "30 books"},
    {"q": "A crate has 3 layers of 8 bottles each. How many bottles total?",  "a": "24 bottles"},
    {"q": "A box has 7 rows with 5 items each. How many items total?",        "a": "35 items"},
]

# ── RAG retrieval ─────────────────────────────────────────────────────────────
class KnowledgeBase:
    """TF-IDF index over question strings.

    Why TF-IDF here instead of embeddings:
      - No GPU or embedding model required
      - Interpretable: retrieval score = lexical overlap weighted by rarity
      - Fast to build and query on CPU
      - Good enough when questions share keywords (speed, area, cost...)
    In production: replace with dense embeddings (sentence-transformers + FAISS).
    """
    def __init__(self, docs: list):
        self.docs       = docs
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix     = self.vectorizer.fit_transform([d["q"] for d in docs])

    def retrieve(self, query: str, k: int = 1) -> list:
        """Return the top-k most similar (q, a) pairs for the query."""
        q_vec = self.vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self.matrix).flatten()
        top_k = sims.argsort()[-k:][::-1]
        return [self.docs[i] for i in top_k]


def build_rag_prompt(query: str, retrieved: list) -> str:
    """Prepend retrieved examples as few-shot context before the actual query.

    Format:
        Example: Q: <retrieved question>
                 A: <retrieved answer>

        Q: <actual question>
        A:
    """
    context = ""
    for ex in retrieved:
        context += f"Example: Q: {ex['q']}\nA: {ex['a']}\n\n"
    return context + query


# ── Model builders ────────────────────────────────────────────────────────────
def build_tiny_model(vocab_size: int) -> Transformer:
    args = ModelArgs(
        dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=vocab_size, max_batch_size=2, max_seq_len=MAX_SEQ_LEN, device="cpu",
    )
    return Transformer(args)

def build_model(tokenizer: SentencePieceProcessor) -> Transformer:
    with open(Path(CHECKPOINTS_DIR) / "params.json") as f:
        params = json.load(f)
    args = ModelArgs(max_seq_len=MAX_SEQ_LEN, max_batch_size=BATCH_SIZE, device=DEVICE, **params)
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
def get_response_log_probs(model, input_ids: torch.Tensor, response_start_idx: int):
    logits = model.forward_train(input_ids)          # (B, T, V)
    logits = logits[:, :-1, :].float()               # (B, T-1, V)
    labels = input_ids[:, 1:]                        # (B, T-1)
    logits = logits[:, response_start_idx - 1:, :]
    labels = labels[:, response_start_idx - 1:]
    log_probs       = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(-1)                   # (B,)

def dpo_loss(policy, ref_model, chosen, rejected, response_start_idx):
    log_p_w = get_response_log_probs(policy,    chosen,   response_start_idx)
    log_p_l = get_response_log_probs(policy,    rejected, response_start_idx)
    with torch.no_grad():
        log_r_w = get_response_log_probs(ref_model, chosen,   response_start_idx)
        log_r_l = get_response_log_probs(ref_model, rejected, response_start_idx)
    chosen_reward   = BETA * (log_p_w - log_r_w)
    rejected_reward = BETA * (log_p_l - log_r_l)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward.mean().item(), rejected_reward.mean().item()

# ── DPO training loop ─────────────────────────────────────────────────────────
def train_dpo(policy, ref_model, tokenizer, data):
    print(f"\n{'='*55}\n  DPO Training\n{'='*55}")
    optim  = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.01,
                                fused=(DEVICE == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
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
            margin  = accum_chosen - accum_rejected
            elapsed = time.time() - t0
            print(f"  step {step:4d} | loss {accum_loss:.4f} | "
                  f"margin {margin:+.3f} | {elapsed:.1f}s")

    print("  DPO done.")
    return policy

# ── Generation ────────────────────────────────────────────────────────────────
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    """Greedy generation. Works with both plain and RAG-augmented prompts."""
    model.eval()
    input_ids = tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
    # Truncate to fit model's max_seq_len
    if len(input_ids) > MAX_SEQ_LEN - max_new_tokens:
        input_ids = input_ids[-(MAX_SEQ_LEN - max_new_tokens):]
    tokens = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits     = model.forward_train(tokens)
            next_token = logits[0, -1, :].argmax(-1)
            if next_token.item() == tokenizer.eos_id():
                break
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=-1)
    model.train()
    return tokenizer.decode(tokens[0, len(input_ids):].tolist())

def generate_with_rag(model, tokenizer, query: str, kb: KnowledgeBase,
                      k: int = 1, max_new_tokens: int = 20) -> tuple:
    """Retrieve context, build augmented prompt, then generate."""
    retrieved = kb.retrieve(query, k=k)
    rag_prompt = build_rag_prompt(query, retrieved)
    response   = generate(model, tokenizer, rag_prompt, max_new_tokens)
    return response, retrieved

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    "Q: A car travels at 30 mph for 4 hours. How far does it travel?\nA:",
    "Q: Oranges cost $3 each. How much do 6 oranges cost?\nA:",
    "Q: A room is 5 m long and 4 m wide. What is its area?\nA:",
    "Q: You invest $1000 at 5% annual interest for 3 years. How much interest?\nA:",
    "Q: A box has 6 rows with 8 items each. How many items total?\nA:",
]
EVAL_ANSWERS = [" 120 miles", " $18", " 20 square meters", " $150", " 48 items"]

def evaluate(model, model_name: str, tokenizer,
             kb: KnowledgeBase = None, use_rag: bool = False) -> int:
    tag = f"{model_name} {'+ RAG' if use_rag else '      '}"
    print(f"\n{'='*55}\n  {tag}\n{'='*55}")
    correct = 0
    for prompt, answer in zip(EVAL_PROMPTS, EVAL_ANSWERS):
        if use_rag:
            response, retrieved = generate_with_rag(model, tokenizer, prompt, kb)
            retrieved_str = f"  Retrieved: Q: {retrieved[0]['q']} → A: {retrieved[0]['a']}"
        else:
            response = generate(model, tokenizer, prompt)
            retrieved_str = ""

        hit      = answer.strip().lower() in response.strip().lower()
        correct += int(hit)
        print(f"\n  Prompt  : {prompt.strip()}")
        if retrieved_str:
            print(retrieved_str)
        print(f"  Expected: {answer.strip()}")
        print(f"  Got     : {response.strip()}")
        print(f"  Correct : {'YES' if hit else 'NO'}")
    return correct

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer = SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    data = load_preference_data()
    print(f"Preference pairs: {len(data)}")

    # Build models
    policy    = build_tiny_model(tokenizer.vocab_size()) if LOCAL_TEST else build_model(tokenizer)
    ref_model = copy.deepcopy(policy)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # DPO fine-tuning
    policy = train_dpo(policy, ref_model, tokenizer, data)

    # Build knowledge base
    kb = KnowledgeBase(KNOWLEDGE_BASE)
    print(f"\nKnowledge base: {len(KNOWLEDGE_BASE)} documents indexed.")
    # Quick retrieval sanity check
    sample_q = "A car travels at 30 mph. How far in 4 hours?"
    hits = kb.retrieve(sample_q, k=1)
    print(f"Sample retrieval for '{sample_q}':")
    print(f"  → Q: {hits[0]['q']}  A: {hits[0]['a']}")

    # 4-way evaluation
    n = len(EVAL_ANSWERS)
    r1 = evaluate(ref_model, "Reference (no DPO)", tokenizer,                   use_rag=False)
    r2 = evaluate(ref_model, "Reference (no DPO)", tokenizer, kb=kb,            use_rag=True)
    r3 = evaluate(policy,    "DPO policy         ", tokenizer,                  use_rag=False)
    r4 = evaluate(policy,    "DPO policy         ", tokenizer, kb=kb,           use_rag=True)

    print(f"\n{'='*55}")
    print(f"  Summary ({n} prompts)")
    print(f"{'='*55}")
    print(f"  Reference  (no RAG) : {r1}/{n}")
    print(f"  Reference  (+ RAG)  : {r2}/{n}  ← context without fine-tuning")
    print(f"  DPO policy (no RAG) : {r3}/{n}  ← fine-tuning without context")
    print(f"  DPO policy (+ RAG)  : {r4}/{n}  ← best of both")


if __name__ == "__main__":
    main()
