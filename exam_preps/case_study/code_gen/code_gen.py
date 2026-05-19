# Code Generation Pipeline — pure Python
#
# Same features as code_gen_mcp.py but zero external dependencies:
# no LangChain, no LangGraph, no MCP.
#
# Pipeline:
#   input_guard → retrieve → generate → code_guard → execute → evaluate
#        ↓ UNSAFE                            ↓ UNSAFE
#       END                                 END
#
# Three simulated scenarios (see __main__):
#   1. Safe task     — exercises the 3-attempt retry loop
#   2. Unsafe task   — input guard blocks before generation
#   3. Dangerous code — code guard blocks before execution

import subprocess
import sys
from dataclasses import dataclass

from rag_eval import RagEvaluator, print_report


TASK = "Write a function that takes a list of integers and returns the top-k most frequent elements."
MAX_RETRIES = 3
_DANGEROUS = ["os.system", "subprocess", "shutil.rmtree", "eval(", "exec(", "__import__"]


# ── Toy knowledge base (same entries as code_exec_server.py) ──────────────────

_KB: list[tuple[list[str], str, str]] = [
    (
        ["frequent", "top-k", "topk", "counter", "count", "most common"],
        "Top-K frequent elements using Counter",
        """\
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    return [item for item, _ in Counter(nums).most_common(k)]
""",
    ),
    (
        ["sort", "sorted", "key", "lambda", "order"],
        "Sort a list by a custom key",
        """\
items = [3, 1, 4, 1, 5]
sorted_items = sorted(items, key=lambda x: -x)  # descending
""",
    ),
    (
        ["heap", "heapq", "nlargest", "nsmallest", "priority"],
        "Find largest/smallest elements with heapq",
        """\
import heapq

nums = [3, 1, 4, 1, 5, 9, 2, 6]
top2 = heapq.nlargest(2, nums)
bot2 = heapq.nsmallest(2, nums)
""",
    ),
    (
        ["dict", "dictionary", "defaultdict", "group", "accumulate"],
        "Group / accumulate values with defaultdict",
        """\
from collections import defaultdict

groups: dict[str, list] = defaultdict(list)
for key, val in [("a", 1), ("b", 2), ("a", 3)]:
    groups[key].append(val)
""",
    ),
]


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    task: str
    code: str = ""
    exec_status: str = ""       # success | syntax_error | runtime_error | timeout
    exec_output: str = ""
    exec_error: str = ""
    eval_passed: bool = False
    eval_feedback: str = ""
    previous_error: str | None = None
    attempt: int = 0
    guard_blocked: bool = False
    guard_reason: str = ""
    retrieved_context: str = ""


# ── LLM stubs ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Stub simulating 3 attempts: syntax error → logic error → correct.

    Attempt is inferred from prompt content:
      no 'previous error'  → attempt 1 (syntax error)
      'syntaxerror'        → attempt 2 (runs but wrong logic)
      anything else        → attempt 3 (correct)

    Evaluator checks whether the code uses Counter + most_common.
    Swap for a real ChatModel to go live.
    """
    p = prompt.lower()

    # Evaluator
    if "pass or fail" in p:
        if "counter" in p and "most_common" in p:
            return "PASS — correctly uses Counter.most_common to return top-k elements."
        return "FAIL — function returns elements sorted by value, not by frequency."

    # Attempt 1 — syntax error (missing colon)
    if "previous error" not in p:
        return """
```python
def top_k_frequent(nums: list[int], k: int) -> list[int]
    return sorted(nums)[:k]

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
```
"""

    # Attempt 2 — runs but wrong: sorts by value, not frequency
    if "syntaxerror" in p:
        return """
```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    return sorted(set(nums), reverse=True)[:k]

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
print(top_k_frequent([1], 1))
```
"""

    # Attempt 3 — correct
    return """
```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    return [item for item, _ in Counter(nums).most_common(k)]

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
print(top_k_frequent([1], 1))
```
"""


def _call_guard_llm(prompt: str) -> str:
    """Stub for both guardrail checks.

    Distinguishes calls by prompt content:
      - Input guard prompt contains "appropriate coding task"
      - Code guard prompt contains "dangerous operations"
    """
    p = prompt.lower()

    # Input guard
    if "appropriate coding task" in p:
        if any(w in p for w in ["delete", "rm -rf", "drop table", "hack", "exploit", "malware"]):
            return "UNSAFE — task requests a potentially harmful operation."
        return "SAFE — standard algorithmic coding task, no harmful intent detected."

    # Code guard
    if any(pat.lower() in p for pat in _DANGEROUS):
        return "UNSAFE — code contains potentially dangerous operations."
    return "SAFE — no dangerous patterns detected in the generated code."


def _extract_code(text: str) -> str:
    if "```python" in text:
        s = text.index("```python") + len("```python")
        return text[s : text.index("```", s)].strip()
    if "```" in text:
        s = text.index("```") + 3
        return text[s : text.index("```", s)].strip()
    return text.strip()


# ── RAG ───────────────────────────────────────────────────────────────────────

def _retrieve(query: str, top_k: int = 2) -> str:
    """Keyword-match retrieval over the in-memory knowledge base."""
    query_lower = query.lower()
    scored = [
        (sum(kw in query_lower for kw in keywords), title, snippet)
        for keywords, title, snippet in _KB
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [item for item in scored if item[0] > 0][:top_k]

    if not hits:
        return ""

    parts = []
    for _, title, snippet in hits:
        parts.append(f"### {title}\n```python\n{snippet.strip()}\n```")
    return "\n\n".join(parts)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def input_guard(state: PipelineState) -> PipelineState:
    prompt = (
        f"Is this a safe, appropriate coding task?\n\nTask: {state.task}\n\n"
        "Reply SAFE or UNSAFE followed by a one-sentence reason."
    )
    response = _call_guard_llm(prompt).strip()
    blocked = not response.upper().startswith("SAFE")
    verdict = "BLOCKED" if blocked else "PASSED"
    print(f"[input_guard]  {verdict} | {response}")
    state.guard_blocked = blocked
    state.guard_reason = response
    return state


def retrieve(state: PipelineState) -> PipelineState:
    context = _retrieve(state.task)
    print(f"[retrieve]     found context ({len(context)} chars)")
    state.retrieved_context = context
    return state


def generate(state: PipelineState) -> PipelineState:
    prompt = f"Task: {state.task}\n\nWrite a complete, runnable Python solution."
    if state.retrieved_context:
        prompt += f"\n\nRelevant examples for reference:\n{state.retrieved_context}"
    if state.previous_error:
        prompt += f"\n\nPrevious error:\n{state.previous_error}\n\nFix it."
    raw = _call_llm(prompt)
    state.code = _extract_code(raw)
    state.attempt += 1
    return state


def code_guard(state: PipelineState) -> PipelineState:
    prompt = (
        "Does the following code contain dangerous operations "
        "(file deletion, arbitrary shell commands, network calls, eval/exec)?\n\n"
        f"Code:\n{state.code}\n\n"
        "Reply SAFE or UNSAFE followed by a one-sentence reason."
    )
    response = _call_guard_llm(prompt).strip()
    blocked = not response.upper().startswith("SAFE")
    verdict = "BLOCKED" if blocked else "PASSED"
    print(f"[code_guard]   {verdict} | {response}")
    state.guard_blocked = blocked
    state.guard_reason = response
    return state


def execute(state: PipelineState) -> PipelineState:
    try:
        result = subprocess.run(
            [sys.executable, "-c", state.code],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            state.exec_status = "success"
            state.exec_output = result.stdout
            state.exec_error = ""
        else:
            state.exec_status = "syntax_error" if "SyntaxError" in result.stderr else "runtime_error"
            state.exec_output = ""
            state.exec_error = result.stderr
            state.previous_error = result.stderr
    except subprocess.TimeoutExpired:
        state.exec_status = "timeout"
        state.exec_error = "Execution timed out after 10s"
        state.previous_error = state.exec_error
    print(f"[execute]      status={state.exec_status}")
    return state


def evaluate(state: PipelineState) -> PipelineState:
    prompt = (
        f"Task: {state.task}\n\nCode:\n{state.code}\n\n"
        f"Output:\n{state.exec_output}\n\n"
        "Does this correctly solve the task? Reply PASS or FAIL + one sentence."
    )
    response = _call_llm(prompt).strip()
    passed = response.upper().startswith("PASS")
    state.eval_passed = passed
    state.eval_feedback = response
    if not passed:
        state.previous_error = f"Evaluation failed: {response}"
    print(f"[evaluate]     {'PASS' if passed else 'FAIL'} | {response}")
    return state


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(task: str = TASK, llm_override=None) -> PipelineState:
    """Run the full pipeline and return the final state.

    llm_override: optionally replace _call_llm for scenario testing.
    """
    import code_gen as _self
    _orig = _self._call_llm
    if llm_override:
        _self._call_llm = llm_override

    state = PipelineState(task=task)

    # ── Input guard ───────────────────────────────────────────────────────
    state = input_guard(state)
    if state.guard_blocked:
        if llm_override:
            _self._call_llm = _orig
        return state

    # ── Retrieve ──────────────────────────────────────────────────────────
    state = retrieve(state)

    # ── Generate → code_guard → execute → evaluate  (with retries) ───────
    for _ in range(MAX_RETRIES):
        state = generate(state)

        state = code_guard(state)
        if state.guard_blocked:
            break

        state = execute(state)
        if state.exec_status != "success":
            continue                         # error fed back; retry generate

        state = evaluate(state)
        if state.eval_passed:
            break                            # done

    if llm_override:
        _self._call_llm = _orig
    return state


# ── Diagram ───────────────────────────────────────────────────────────────────

def save_graph(path: str = "code_gen_graph"):
    """Write the pipeline diagram as a Mermaid file (no extra deps needed)."""
    mermaid = """\
flowchart TD
    START([START]) --> input_guard
    input_guard -->|SAFE| retrieve
    input_guard -->|UNSAFE| END_GUARD([END])
    retrieve --> generate
    generate --> code_guard
    code_guard -->|SAFE| execute
    code_guard -->|UNSAFE| END_CODE([END])
    execute -->|success| evaluate
    execute -->|error / timeout| generate
    evaluate -->|PASS| END_PASS([END])
    evaluate -->|FAIL + retries left| generate
    evaluate -->|max retries| END_RETRY([END])
"""
    mermaid_path = f"{path}.mermaid"
    with open(mermaid_path, "w") as f:
        f.write(mermaid)
    print(f"Mermaid diagram saved → {mermaid_path}")


# ── Scenarios ─────────────────────────────────────────────────────────────────

def _print_result(state: PipelineState) -> None:
    if state.guard_blocked:
        print(f"\nPipeline BLOCKED: {state.guard_reason}")
        return
    print(f"\nPassed   : {state.eval_passed}")
    print(f"Attempts : {state.attempt}")
    print(f"Feedback : {state.eval_feedback}")
    print(f"\nFinal code:\n{state.code}")

    if state.retrieved_context and state.code:
        evaluator = RagEvaluator()
        results = evaluator.evaluate_all(
            query=state.task,
            context=state.retrieved_context,
            answer=state.code,
        )
        print_report(results)


if __name__ == "__main__":
    save_graph()

    # ── Scenario 1: safe task — 3-attempt retry loop ──────────────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 1 — Safe task (3-attempt retry loop)")
    print("=" * 60)
    _print_result(run_pipeline(TASK))

    # ── Scenario 2: unsafe task — input guard blocks ──────────────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 2 — Unsafe task (input guard blocks)")
    print("=" * 60)
    unsafe_task = "Write a script that deletes all .py files in the current directory."
    _print_result(run_pipeline(unsafe_task))

    # ── Scenario 3: dangerous generated code — code guard blocks ─────────
    print("\n" + "=" * 60)
    print("  SCENARIO 3 — Dangerous generated code (code guard blocks)")
    print("=" * 60)

    def _dangerous_llm(prompt: str) -> str:
        if "pass or fail" in prompt.lower():
            return "PASS"
        return """
```python
import os
os.system("rm -rf /tmp/test")

def top_k_frequent(nums, k):
    return []
```
"""

    _print_result(run_pipeline(TASK, llm_override=_dangerous_llm))
