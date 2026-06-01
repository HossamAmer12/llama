# RAG Evaluation — Faithfulness, Context Relevance, Answer Relevance, Context Recall
#
# Each metric follows the RAGAS decomposition:
#   1. An LLM extracts atomic statements from the text being scored.
#   2. A second LLM call verifies each statement against a reference.
#   3. Score = verified / total
#
# Stub LLMs return hardcoded verdicts so the pipeline runs end-to-end without
# a real API key. Swap _stub_eval_llm for a real ChatModel to get live scores.

from dataclasses import dataclass, field


# ── LLM stub ──────────────────────────────────────────────────────────────────

def _stub_eval_llm(prompt: str) -> str:
    """Stub — replace with your real LLM call (e.g. ChatAnthropic)."""
    p = prompt.lower()

    # faithfulness: LLM lists claims and marks each SUPPORTED / NOT SUPPORTED
    if "supported or not supported" in p:
        return (
            "1. Uses Counter from collections. SUPPORTED\n"
            "2. Calls .most_common(k). SUPPORTED\n"
            "3. Returns a list of integers. SUPPORTED\n"
        )

    # context relevance: LLM marks each context sentence RELEVANT / IRRELEVANT
    if "relevant or irrelevant" in p:
        return (
            "Sentence 1: RELEVANT\n"
            "Sentence 2: RELEVANT\n"
            "Sentence 3: IRRELEVANT\n"
        )

    # answer relevance: LLM generates questions the answer implicitly answers
    if "generate questions" in p:
        return (
            "Q1: How to find the top-k frequent elements in a list?\n"
            "Q2: How to count element frequencies in Python?\n"
        )

    # context recall: LLM checks if each ground-truth statement is in context
    if "attributable" in p:
        return (
            "1. Counter usage is attributable to the context. YES\n"
            "2. most_common(k) is attributable to the context. YES\n"
            "3. List comprehension is not explicitly in the context. NO\n"
        )

    return "SUPPORTED"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    name: str
    score: float                        # 0.0 – 1.0
    verdicts: list[str] = field(default_factory=list)
    explanation: str = ""


# ── RagEvaluator ──────────────────────────────────────────────────────────────

class RagEvaluator:
    """
    Evaluates a RAG pipeline output along four axes.

    Parameters
    ----------
    llm_fn : callable, optional
        Function (prompt: str) -> str.  Defaults to the stub.
    """

    def __init__(self, llm_fn=None):
        self._llm = llm_fn or _stub_eval_llm

    # ── 1. Faithfulness ───────────────────────────────────────────────────────

    def faithfulness(self, context: str, answer: str) -> MetricResult:
        """
        Fraction of claims in `answer` that are supported by `context`.

        Steps:
          a) Extract atomic claims from the answer.
          b) For each claim, ask the LLM: SUPPORTED or NOT SUPPORTED.
          c) Score = supported / total.
        """
        claims_prompt = (
            f"List every atomic factual claim made in the following code/answer "
            f"as numbered sentences.\n\nAnswer:\n{answer}"
        )
        claims_raw = self._llm(claims_prompt)
        claims = [l.strip() for l in claims_raw.strip().splitlines() if l.strip()]

        verify_prompt = (
            f"Context:\n{context}\n\n"
            "For each claim below, reply 'SUPPORTED' or 'NOT SUPPORTED' "
            "based only on the context.\n\n"
            + "\n".join(claims)
        )
        verdicts_raw = self._llm(verify_prompt)
        verdicts = [l.strip() for l in verdicts_raw.strip().splitlines() if l.strip()]

        supported = sum(1 for v in verdicts if "NOT SUPPORTED" not in v.upper() and "SUPPORTED" in v.upper())
        total = max(len(verdicts), 1)
        return MetricResult(
            name="faithfulness",
            score=round(supported / total, 3),
            verdicts=verdicts,
            explanation=f"{supported}/{total} claims supported by the retrieved context.",
        )

    # ── 2. Context Relevance ──────────────────────────────────────────────────

    def context_relevance(self, query: str, context: str) -> MetricResult:
        """
        Fraction of the retrieved context that is relevant to `query`.

        Steps:
          a) Split context into sentences/chunks.
          b) Ask the LLM: RELEVANT or IRRELEVANT for each chunk.
          c) Score = relevant / total.
        """
        sentences = [s.strip() for s in context.split("\n") if s.strip()]

        prompt = (
            f"Query: {query}\n\n"
            "For each sentence below, reply 'RELEVANT' or 'IRRELEVANT' "
            "with respect to the query.\n\n"
            + "\n".join(f"Sentence {i+1}: {s}" for i, s in enumerate(sentences))
        )
        verdicts_raw = self._llm(prompt)
        verdicts = [l.strip() for l in verdicts_raw.strip().splitlines() if l.strip()]

        relevant = sum(1 for v in verdicts if "IRRELEVANT" not in v.upper() and "RELEVANT" in v.upper())
        total = max(len(verdicts), 1)
        return MetricResult(
            name="context_relevance",
            score=round(relevant / total, 3),
            verdicts=verdicts,
            explanation=f"{relevant}/{total} context sentences relevant to the query.",
        )

    # ── 3. Answer Relevance ───────────────────────────────────────────────────

    def answer_relevance(self, query: str, answer: str) -> MetricResult:
        """
        How well `answer` addresses `query`.

        Steps:
          a) Ask the LLM to generate N questions that `answer` would answer.
          b) Score = fraction of generated questions that resemble the original query.

        In production: embed the questions and compute mean cosine similarity.
        Here: a simple keyword-overlap proxy.
        """
        gen_prompt = (
            f"Given the answer below, generate 2-3 questions that it answers.\n\n"
            f"Answer:\n{answer}"
        )
        questions_raw = self._llm(gen_prompt)
        questions = [l.strip() for l in questions_raw.strip().splitlines() if l.strip() and "?" in l]

        query_words = set(query.lower().split())
        scores = []
        for q in questions:
            q_words = set(q.lower().split())
            overlap = len(query_words & q_words) / max(len(query_words), 1)
            scores.append(overlap)

        score = round(sum(scores) / max(len(scores), 1), 3)
        return MetricResult(
            name="answer_relevance",
            score=score,
            verdicts=questions,
            explanation=f"Mean keyword overlap between generated questions and original query: {score}.",
        )

    # ── 4. Context Recall ─────────────────────────────────────────────────────

    def context_recall(self, context: str, answer: str) -> MetricResult:
        """
        Fraction of `answer` statements attributable to `context`.

        Steps:
          a) Extract statements from the answer.
          b) For each, ask: is it attributable to the context? YES / NO.
          c) Score = YES / total.
        """
        extract_prompt = (
            f"Break the following answer into numbered atomic statements.\n\n"
            f"Answer:\n{answer}"
        )
        statements_raw = self._llm(extract_prompt)
        statements = [l.strip() for l in statements_raw.strip().splitlines() if l.strip()]

        attr_prompt = (
            f"Context:\n{context}\n\n"
            "For each statement, reply YES if it is attributable to the context, "
            "NO otherwise.\n\n"
            + "\n".join(statements)
        )
        verdicts_raw = self._llm(attr_prompt)
        verdicts = [l.strip() for l in verdicts_raw.strip().splitlines() if l.strip()]

        yes_count = sum(1 for v in verdicts if v.upper().endswith("YES") or ". YES" in v.upper())
        total = max(len(verdicts), 1)
        return MetricResult(
            name="context_recall",
            score=round(yes_count / total, 3),
            verdicts=verdicts,
            explanation=f"{yes_count}/{total} answer statements attributable to the context.",
        )

    # ── Full report ───────────────────────────────────────────────────────────

    def evaluate_all(self, query: str, context: str, answer: str) -> dict[str, MetricResult]:
        """Run all four metrics and return a keyed dict."""
        return {
            "faithfulness":      self.faithfulness(context, answer),
            "context_relevance": self.context_relevance(query, context),
            "answer_relevance":  self.answer_relevance(query, answer),
            "context_recall":    self.context_recall(context, answer),
        }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_report(results: dict[str, MetricResult]) -> None:
    print("\n" + "=" * 52)
    print("  RAG Evaluation Report")
    print("=" * 52)
    for result in results.values():
        bar_len = int(result.score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"\n{result.name:<22} {bar}  {result.score:.3f}")
        print(f"  {result.explanation}")
    print("=" * 52)
