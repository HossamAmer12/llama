import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

df = pd.DataFrame({
    'prompt_id':   range(n),
    'annotator_1': np.random.choice(['A', 'B'], n, p=[0.6, 0.4]),
    'annotator_2': np.random.choice(['A', 'B'], n, p=[0.55, 0.45]),
    'judge_score': np.random.randn(n),   # + favors A, - favors B
    'human_label': np.random.choice(['A', 'B'], n, p=[0.6, 0.4]),
})

print(df.head())

# Task 1 — Compute the raw agreement rate between annotator_1 and annotator_2.
agreement_rate = (df['annotator_1'] == df['annotator_2']).mean()
print(f"Agreement rate: {agreement_rate:.2%}")


# Task 2 — Compute Cohen's Kappa to measure inter-annotator agreement beyond chance.
def cohen_kappa(df, col1, col2):
    # Compute observed agreement
    observed_agreement = (df[col1] == df[col2]).mean()

    # Compute expected agreement by chance
    categories = df[col1].unique()
    expected_agreement = 0.0
    for category in categories:
        p1 = (df[col1] == category).mean()
        p2 = (df[col2] == category).mean()
        expected_agreement += p1 * p2

    # Compute Cohen's Kappa
    if expected_agreement == 1.0:
        return 1.0  # Avoid division by zero when perfect agreement
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa


kappa = cohen_kappa(df, 'annotator_1', 'annotator_2')
print(f"Cohen's Kappa: {kappa:.3f}")

# Task 3 — now look at the judge_score column. The judge outputs a continuous score where positive favors A and negative favors B. Compute how well the judge aligns with human_label. What metric would you use and how would you compute it?

judge_label = np.where(df['judge_score'] > 0, 'A', 'B')
accuracy = (judge_label == df['human_label']).mean()
print(f"Judge accuracy against human labels: {accuracy:.2%}")


# Also worth: AUROC — measures ranking quality without committing to a threshold.
# from sklearn.metrics import roc_auc_score
# auroc = roc_auc_score((df['human_label'] == 'A').astype(int), df['judge_score'])


# ── Richer dataset for tasks 4–10 (model-pair rows with ties) ─────────────────
from itertools import combinations

MODEL_STRENGTH = {"gpt4": 0.80, "claude": 0.60, "llama": 0.35}

def _noisy_label(true, noise, pos_bias=0.0):
    if np.random.random() < noise:
        return np.random.choice(["A", "B", "tie"], p=[0.4, 0.4, 0.2])
    if true != "tie" and np.random.random() < pos_bias:
        return "A"   # always favour position A regardless of which model is there
    return true

def make_pairwise_dataset(n=300):
    models, rows = list(MODEL_STRENGTH), []
    for i in range(n):
        ma, mb = np.random.choice(models, size=2, replace=False)
        p_a = 1 / (1 + np.exp(-(MODEL_STRENGTH[ma] - MODEL_STRENGTH[mb]) * 4))
        r = np.random.random()
        if   r < p_a * 0.80:          true = "A"
        elif r < p_a:                  true = "tie"
        elif r < p_a + (1-p_a)*0.80:  true = "B"
        else:                          true = "tie"
        rows.append({
            "prompt_id":   f"p{i:03d}",
            "model_a": ma, "model_b": mb, "true_label": true,
            "annotator_1": _noisy_label(true, noise=0.15),
            "annotator_2": _noisy_label(true, noise=0.25),
            "annotator_3": _noisy_label(true, noise=0.20),
            "judge_llm":   _noisy_label(true, noise=0.18, pos_bias=0.10),  # has position bias
            "judge_rm":    _noisy_label(true, noise=0.32),
        })
    return pd.DataFrame(rows)

print("\nSample pairwise dataset:")
pdf = make_pairwise_dataset(300)

print(pdf.head())

# Task 4 — Position bias. The judge_llm was constructed with a bias toward position A.
# Detect it with a single statistic. Compare against annotator_1.
# Ideal = 0. Positive = judge systematically favours whichever response is listed first.

def position_bias(labels: pd.Series) -> float:
    counts = labels.value_counts(normalize=True)
    return counts.get("A", 0.0) - counts.get("B", 0.0)

for col in ["annotator_1", "judge_llm", "judge_rm"]:
    print(f"Position bias {col:14s}: {position_bias(pdf[col]):+.3f}")


# Task 5 — Consistency under order flip.
# Show each pair twice: original (A, B) and flipped (B, A). A good judge gives
# the same verdict regardless of order. Measure the fraction of consistent pairs.

def make_flip_dataset(n_pairs=80):
    models, rows = list(MODEL_STRENGTH), []
    for i in range(n_pairs):
        ma, mb = np.random.choice(models, size=2, replace=False)
        p_a = 1 / (1 + np.exp(-(MODEL_STRENGTH[ma] - MODEL_STRENGTH[mb]) * 4))
        r = np.random.random()
        true = "A" if r < p_a*0.85 else ("tie" if r < p_a else ("B" if r < p_a+(1-p_a)*0.85 else "tie"))
        for is_flipped in [False, True]:
            label = {"A":"B","B":"A","tie":"tie"}[true] if is_flipped else true
            rows.append({
                "pair_id": f"p{i:03d}", "is_flipped": is_flipped,
                "judge_llm": _noisy_label(label, noise=0.18, pos_bias=0.10),
                "judge_rm":  _noisy_label(label, noise=0.32),
            })
    return pd.DataFrame(rows)

def consistency_rate(df, judge_col):
    orig   = df[~df["is_flipped"]].set_index("pair_id")[judge_col]
    flip   = df[ df["is_flipped"]].set_index("pair_id")[judge_col]
    shared = orig.index.intersection(flip.index)
    # After flip: a correct judge should swap A and B. Normalize then compare.
    corrected = flip.loc[shared].map({"A":"B","B":"A","tie":"tie"})
    return (orig.loc[shared] == corrected).mean()


print("\nConsistency under order flip:")
fdf = make_flip_dataset(80)
print(fdf.head())

for col in ["judge_llm", "judge_rm"]:
    print(f"Consistency {col}: {consistency_rate(fdf, col):.3f}")
# Low consistency + high position bias go together: the judge picks A
# because it is in position A, not because it is the better response.


# Task 6 — Majority vote. Aggregate 3 annotators into a single gold label.
# Show that the aggregate improves kappa against the true label vs any single annotator.

def majority_vote(df, cols):
    def _vote(row):
        counts  = row.value_counts()
        winners = counts[counts == counts.max()].index.tolist()
        return winners[0] if len(winners) == 1 else "tie"
    return df[cols].apply(_vote, axis=1)

pdf["majority"] = majority_vote(pdf, ["annotator_1","annotator_2","annotator_3"])
k_single = cohen_kappa(pdf, "true_label", "annotator_1")
k_mv     = cohen_kappa(pdf, "true_label", "majority")
print(f"Kappa vs true — single: {k_single:.3f}  majority vote: {k_mv:.3f}")


# Task 7 — Win rate matrix. For every model pair compute P(A beats B).
# Ties count as 0.5 for each side.

def win_rate_matrix(df, label_col):
    models = sorted(set(df["model_a"]) | set(df["model_b"]))
    matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for ma, mb in combinations(models, 2):
        mask = (((df["model_a"]==ma)&(df["model_b"]==mb)) |
                ((df["model_a"]==mb)&(df["model_b"]==ma)))
        subset = df[mask]
        if subset.empty: continue
        def _score(row):
            if row["model_a"] == ma: return {"A":1.0,"B":0.0,"tie":0.5}[row[label_col]]
            return {"A":0.0,"B":1.0,"tie":0.5}[row[label_col]]
        wr = subset.apply(_score, axis=1).mean()
        matrix.loc[ma, mb] = wr
        matrix.loc[mb, ma] = 1.0 - wr
    return matrix

print("\nWin rate matrix (annotator_1):")
print(win_rate_matrix(pdf, "annotator_1").round(3))


# Task 8 — Bradley-Terry. Unlike win rate, BT accounts for opponent strength —
# beating a strong model is worth more than beating a weak one.
# Implement the MM iterative update:
#   strength[i] ← wins[i] / Σ_j [ n(i,j) / (strength[i] + strength[j]) ]
# Normalise so strengths sum to 1 after each iteration.

def bradley_terry(df, label_col, n_iter=200):
    models   = sorted(set(df["model_a"]) | set(df["model_b"]))
    strength = {m: 1.0 for m in models}
    wins     = {m: 0.0 for m in models}
    n_games  = {(ma,mb): 0 for ma in models for mb in models}
    for _, row in df.iterrows():
        ma, mb, label = row["model_a"], row["model_b"], row[label_col]
        n_games[(ma,mb)] += 1; n_games[(mb,ma)] += 1
        if label=="A":   wins[ma] += 1.0
        elif label=="B": wins[mb] += 1.0
        else:            wins[ma] += 0.5; wins[mb] += 0.5
    for _ in range(n_iter):
        for m in models:
            denom = sum(n_games[(m,o)]/(strength[m]+strength[o])
                        for o in models if o!=m and n_games[(m,o)]>0)
            if denom > 0: strength[m] = wins[m] / denom
        total = sum(strength.values())
        strength = {m: v/total for m,v in strength.items()}
    return pd.Series(strength).sort_values(ascending=False)

print("\nBradley-Terry rankings:")
for col in ["true_label", "annotator_1", "judge_llm", "judge_rm"]:
    r = bradley_terry(pdf, col)
    print(f"  {col:14s}: " + "  ".join(f"{m}={v:.3f}" for m,v in r.items()))


# Task 9 — Bootstrap CI for win rate. Wide CI = unreliable ranking.
# Resample matchup rows with replacement n_boot times, compute win rate each
# time, then take the 2.5th and 97.5th percentile as the 95% interval.

def bootstrap_win_rate(df, label_col, model_a, model_b, n_boot=1000):
    mask = (((df["model_a"]==model_a)&(df["model_b"]==model_b)) |
            ((df["model_a"]==model_b)&(df["model_b"]==model_a)))
    subset = df[mask]
    def _score(row):
        if row["model_a"]==model_a: return {"A":1.0,"B":0.0,"tie":0.5}[row[label_col]]
        return {"A":0.0,"B":1.0,"tie":0.5}[row[label_col]]
    scores = subset.apply(_score, axis=1).values
    point  = scores.mean()
    boots  = [np.random.choice(scores, len(scores), replace=True).mean()
              for _ in range(n_boot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

print("\nBootstrap win rate gpt4 vs claude:")
for col in ["annotator_1", "judge_llm", "judge_rm"]:
    pt, lo, hi = bootstrap_win_rate(pdf, col, "gpt4", "claude")
    print(f"  {col:14s}: {pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  width={hi-lo:.3f}")


# Task 10 — Rank correlation. A noisy judge can still be useful if it preserves
# the relative ordering of models even when individual pair labels are wrong.
# Compute Spearman ρ between BT rankings from two judge configurations.

from scipy.stats import spearmanr

def rank_correlation(df, col_a, col_b):
    bt_a = bradley_terry(df, col_a)
    bt_b = bradley_terry(df, col_b)
    shared = bt_a.index.intersection(bt_b.index)
    rho, _ = spearmanr(bt_a.loc[shared].values, bt_b.loc[shared].values)
    return float(rho)

print("\nRank correlation (Spearman ρ) vs annotator_1:")
for col in ["annotator_2", "judge_llm", "judge_rm"]:
    rho = rank_correlation(pdf, "annotator_1", col)
    print(f"  {col:14s}: ρ = {rho:.3f}")
# Key insight: judge_rm may have low kappa (poor per-pair agreement) but still
# high ρ (correct final ranking). That is often acceptable for model selection.
