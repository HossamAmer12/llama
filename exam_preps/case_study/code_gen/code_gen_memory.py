# Code Generation Pipeline — with short-term and long-term memory
#
# Short-term memory: every attempt (code + error + eval feedback) is recorded
# in PipelineState.history and injected into the next generation prompt so the
# LLM can see exactly what went wrong across ALL prior attempts, not just the
# last one.
#
# Long-term memory: a JSON file (code_gen_ltm.json) that persists successful
# solutions and user preferences across sessions.  Retrieved at the start of
# each generate() call to give the LLM a head-start for familiar tasks, and
# updated on success so future runs benefit immediately.
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
#   4. LTM hit       — past solution shortens retry count to 1

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rag_eval import RagEvaluator, print_report


TASK = "Write a function that takes a list of integers and returns the top-k most frequent elements."
MAX_RETRIES = 3
_DANGEROUS = ["os.system", "subprocess", "shutil.rmtree", "eval(", "exec(", "__import__"]

LTM_PATH = Path(__file__).parent / "code_gen_ltm.json"


# ── Toy knowledge base ────────────────────────────────────────────────────────

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


# ── Long-term memory ──────────────────────────────────────────────────────────

class LongTermMemory:
    """JSON-backed store for successful solutions and user preferences.

    Solutions are keyed by the exact task string.
    Preferences are free-form key/value pairs (e.g. "style": "functional").
    """

    def __init__(self, path: Path = LTM_PATH):
        self.path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                pass
        return {"solutions": {}, "preferences": {}}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))

    # ── solutions ─────────────────────────────────────────────────────────────

    def store_solution(self, task: str, code: str, attempts: int) -> None:
        self._data["solutions"][task] = {
            "code": code,
            "attempts": attempts,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        print(f"[ltm]          stored solution for task (attempts={attempts})")

    def recall_solution(self, task: str) -> str | None:
        entry = self._data["solutions"].get(task)
        if entry:
            print(f"[ltm]          recalled past solution (was solved in {entry['attempts']} attempt(s))")
            return entry["code"]
        return None

    # ── preferences ───────────────────────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        self._data["preferences"][key] = value
        self._save()

    def get_preference(self, key: str) -> str | None:
        return self._data["preferences"].get(key)

    def all_preferences(self) -> dict:
        return dict(self._data["preferences"])

    def summary(self) -> str:
        n_sol = len(self._data["solutions"])
        prefs = self._data["preferences"]
        lines = [f"LTM: {n_sol} stored solution(s)"]
        if prefs:
            lines.append("Preferences: " + ", ".join(f"{k}={v}" for k, v in prefs.items()))
        return " | ".join(lines)


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
    previous_error: str | None = None   # kept for backward-compat; history is richer
    attempt: int = 0
    guard_blocked: bool = False
    guard_reason: str = ""
    retrieved_context: str = ""
    # Short-term memory: one entry per completed attempt
    history: list[dict] = field(default_factory=list)


# ── LLM stubs ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Stub simulating 3 attempts: syntax error → logic error → correct.

    Attempt number is inferred from how many '### attempt' history blocks the
    prompt contains (written by _build_history_section):
      0 history entries → attempt 1 (syntax error)
      1 history entry   → attempt 2 (runs but wrong logic)
      2+ history entries → attempt 3 (correct)
    """
    p = prompt.lower()

    if "pass or fail" in p:
        if "counter" in p and "most_common" in p:
            return "PASS — correctly uses Counter.most_common to return top-k elements."
        return "FAIL — function returns elements sorted by value, not by frequency."

    history_count = p.count("### attempt")

    if history_count == 0:
        return """
```python
def top_k_frequent(nums: list[int], k: int) -> list[int]
    return sorted(nums)[:k]

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
```
"""

    if history_count == 1:
        return """
```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    return sorted(set(nums), reverse=True)[:k]

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
print(top_k_frequent([1], 1))
```
"""

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
    p = prompt.lower()
    if "appropriate coding task" in p:
        if any(w in p for w in ["delete", "rm -rf", "drop table", "hack", "exploit", "malware"]):
            return "UNSAFE — task requests a potentially harmful operation."
        return "SAFE — standard algorithmic coding task, no harmful intent detected."
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


# ── Short-term memory helpers ─────────────────────────────────────────────────

def _build_history_section(history: list[dict]) -> str:
    """Render the full attempt history as a prompt section."""
    if not history:
        return ""
    lines = ["\n## Previous Attempts (fix based on all history below)\n"]
    for entry in history:
        lines.append(f"### Attempt {entry['attempt']}")
        lines.append(f"**Code:**\n```python\n{entry['code']}\n```")
        if entry.get("error"):
            lines.append(f"**Execution error:**\n{entry['error'].strip()}")
        if entry.get("feedback"):
            lines.append(f"**Evaluation feedback:** {entry['feedback'].strip()}")
        lines.append("")
    return "\n".join(lines)


def _record_attempt(state: PipelineState, error: str = "", feedback: str = "") -> None:
    state.history.append({
        "attempt": state.attempt,
        "code": state.code,
        "error": error,
        "feedback": feedback,
    })


# ── Pipeline steps ────────────────────────────────────────────────────────────

def input_guard(state: PipelineState) -> PipelineState:
    prompt = (
        f"Is this a safe, appropriate coding task?\n\nTask: {state.task}\n\n"
        "Reply SAFE or UNSAFE followed by a one-sentence reason."
    )
    response = _call_guard_llm(prompt).strip()
    blocked = not response.upper().startswith("SAFE")
    print(f"[input_guard]  {'BLOCKED' if blocked else 'PASSED'} | {response}")
    state.guard_blocked = blocked
    state.guard_reason = response
    return state


def retrieve(state: PipelineState) -> PipelineState:
    context = _retrieve(state.task)
    print(f"[retrieve]     found context ({len(context)} chars)")
    state.retrieved_context = context
    return state


def generate(state: PipelineState, ltm: LongTermMemory | None = None) -> PipelineState:
    prompt = f"Task: {state.task}\n\nWrite a complete, runnable Python solution."

    # Apply user preferences from long-term memory
    if ltm:
        prefs = ltm.all_preferences()
        if prefs:
            pref_str = "; ".join(f"{k}: {v}" for k, v in prefs.items())
            prompt += f"\n\nUser preferences: {pref_str}"

    if state.retrieved_context:
        prompt += f"\n\nRelevant examples for reference:\n{state.retrieved_context}"

    # Long-term memory: inject a past successful solution as a hint
    if ltm:
        past_solution = ltm.recall_solution(state.task)
        if past_solution:
            prompt += f"\n\nPreviously successful solution for this task (use as reference):\n```python\n{past_solution}\n```"

    # Short-term memory: full interaction history
    history_section = _build_history_section(state.history)
    if history_section:
        prompt += history_section

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
    print(f"[code_guard]   {'BLOCKED' if blocked else 'PASSED'} | {response}")
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
            # Record failed attempt in short-term memory
            _record_attempt(state, error=result.stderr)
    except subprocess.TimeoutExpired:
        state.exec_status = "timeout"
        state.exec_error = "Execution timed out after 10s"
        state.previous_error = state.exec_error
        _record_attempt(state, error=state.exec_error)
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
        # Record evaluation failure in short-term memory
        _record_attempt(state, feedback=response)
    print(f"[evaluate]     {'PASS' if passed else 'FAIL'} | {response}")
    return state


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    task: str = TASK,
    ltm: LongTermMemory | None = None,
    llm_override=None,
) -> PipelineState:
    """Run the full pipeline and return the final state.

    ltm: optional LongTermMemory instance — consulted during generation,
         updated on success.
    llm_override: optionally replace _call_llm for scenario testing.
    """
    import code_gen_memory as _self
    _orig = _self._call_llm
    if llm_override:
        _self._call_llm = llm_override

    state = PipelineState(task=task)

    if ltm:
        print(f"[ltm]          {ltm.summary()}")

    state = input_guard(state)
    if state.guard_blocked:
        if llm_override:
            _self._call_llm = _orig
        return state

    state = retrieve(state)

    for _ in range(MAX_RETRIES):
        state = generate(state, ltm=ltm)

        state = code_guard(state)
        if state.guard_blocked:
            break

        state = execute(state)
        if state.exec_status != "success":
            continue

        state = evaluate(state)
        if state.eval_passed:
            # Persist successful solution to long-term memory
            if ltm:
                ltm.store_solution(task, state.code, state.attempt)
            break

    if llm_override:
        _self._call_llm = _orig
    return state


# ── Diagram ───────────────────────────────────────────────────────────────────

def save_graph(path: str = "code_gen_graph"):
    mermaid = """\
flowchart TD
    START([START]) --> ltm_load[Load LTM]
    ltm_load --> input_guard
    input_guard -->|SAFE| retrieve
    input_guard -->|UNSAFE| END_GUARD([END])
    retrieve --> generate
    generate -->|LTM hint + STM history| generate
    generate --> code_guard
    code_guard -->|SAFE| execute
    code_guard -->|UNSAFE| END_CODE([END])
    execute -->|success| evaluate
    execute -->|error → record STM| generate
    evaluate -->|PASS → store LTM| END_PASS([END])
    evaluate -->|FAIL → record STM| generate
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

    if state.history:
        print(f"\nShort-term memory ({len(state.history)} recorded failure(s)):")
        for entry in state.history:
            err = entry.get("error", "").strip().splitlines()[0] if entry.get("error") else ""
            fb = entry.get("feedback", "").strip()
            detail = err or fb or "(no detail)"
            print(f"  attempt {entry['attempt']}: {detail[:80]}")

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

    ltm = LongTermMemory()

    # ── Scenario 1: safe task — 3-attempt retry loop ──────────────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 1 — Safe task (3-attempt retry loop, LTM enabled)")
    print("=" * 60)
    _print_result(run_pipeline(TASK, ltm=ltm))

    # ── Scenario 2: unsafe task — input guard blocks ──────────────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 2 — Unsafe task (input guard blocks)")
    print("=" * 60)
    unsafe_task = "Write a script that deletes all .py files in the current directory."
    _print_result(run_pipeline(unsafe_task, ltm=ltm))

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

    _print_result(run_pipeline(TASK, ltm=ltm, llm_override=_dangerous_llm))

    # ── Scenario 4: LTM hit — past solution reduces retries ──────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 4 — LTM hit (same task, solution recalled from disk)")
    print("=" * 60)
    print("  (Re-running Scenario 1 task — LTM should now hold the solution)")
    _print_result(run_pipeline(TASK, ltm=ltm))
