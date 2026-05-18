# Code Generation Pipeline — LangGraph version
#
# pip install langgraph
#
# Graph:
#   START → generate → execute ──success──► evaluate ──pass──► END
#                        ↑                               │
#                        └──────── error / fail ─────────┘

import subprocess
import sys
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


TASK = "Write a function that takes a list of integers and returns the top-k most frequent elements."
MAX_RETRIES = 3


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task: str
    code: str
    exec_status: str        # success | syntax_error | runtime_error | timeout
    exec_output: str
    exec_error: str
    eval_passed: bool
    eval_feedback: str
    previous_error: str | None
    attempt: int


# ── LLM stub ───────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Stub — swap in ChatAnthropic / ChatOpenAI / your model here."""
    if "pass or fail" in prompt.lower():
        return "PASS — correctly returns top-k frequent elements."
    return """
```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    return [item for item, _ in Counter(nums).most_common(k)]

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
print(top_k_frequent([1], 1))
```
"""


def _extract_code(text: str) -> str:
    if "```python" in text:
        s = text.index("```python") + len("```python")
        return text[s : text.index("```", s)].strip()
    if "```" in text:
        s = text.index("```") + 3
        return text[s : text.index("```", s)].strip()
    return text.strip()


# ── Nodes ──────────────────────────────────────────────────────────────────────

def generate_node(state: AgentState) -> dict:
    prompt = f"Task: {state['task']}\n\nWrite a complete, runnable Python solution."
    if state["previous_error"]:
        prompt += f"\n\nPrevious error:\n{state['previous_error']}\n\nFix it."
    raw = _call_llm(prompt)
    return {"code": _extract_code(raw), "attempt": state["attempt"] + 1}


def execute_node(state: AgentState) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-c", state["code"]],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return {"exec_status": "success", "exec_output": result.stdout, "exec_error": ""}
        status = "syntax_error" if "SyntaxError" in result.stderr else "runtime_error"
        return {"exec_status": status, "exec_output": "", "exec_error": result.stderr,
                "previous_error": result.stderr}
    except subprocess.TimeoutExpired:
        err = "Execution timed out after 10s"
        return {"exec_status": "timeout", "exec_output": "", "exec_error": err,
                "previous_error": err}


def evaluate_node(state: AgentState) -> dict:
    prompt = (
        f"Task: {state['task']}\n\nCode:\n{state['code']}\n\n"
        f"Output:\n{state['exec_output']}\n\n"
        "Does this correctly solve the task? Reply PASS or FAIL + one sentence."
    )
    response = _call_llm(prompt).strip()
    passed = response.upper().startswith("PASS")
    updates: dict = {"eval_passed": passed, "eval_feedback": response}
    if not passed:
        updates["previous_error"] = f"Evaluation failed: {response}"
    return updates


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_execute(state: AgentState) -> Literal["evaluate", "generate", "__end__"]:
    if state["exec_status"] == "success":
        return "evaluate"
    if state["attempt"] >= MAX_RETRIES:
        return END
    return "generate"


def route_after_evaluate(state: AgentState) -> Literal["__end__", "generate"]:
    if state["eval_passed"] or state["attempt"] >= MAX_RETRIES:
        return END
    return "generate"


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("generate", generate_node)
    builder.add_node("execute", execute_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "execute")
    builder.add_conditional_edges("execute", route_after_execute)
    builder.add_conditional_edges("evaluate", route_after_evaluate)
    return builder.compile()


graph = build_graph()


# ── Graph export ───────────────────────────────────────────────────────────────

def save_graph(path: str = "code_gen_graph"):
    """Save the workflow graph as a .mermaid file and optionally a .png.

    PNG requires:  pip install playwright && playwright install
    """
    mermaid_text = graph.get_graph().draw_mermaid()
    mermaid_path = f"{path}.mermaid"
    with open(mermaid_path, "w") as f:
        f.write(mermaid_text)
    print(f"Mermaid diagram saved → {mermaid_path}")

    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path = f"{path}.png"
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"PNG diagram saved      → {png_path}")
    except Exception:
        print("PNG skipped (run: pip install playwright && playwright install)")


if __name__ == "__main__":
    save_graph()

    initial: AgentState = {
        "task": TASK,
        "code": "",
        "exec_status": "",
        "exec_output": "",
        "exec_error": "",
        "eval_passed": False,
        "eval_feedback": "",
        "previous_error": None,
        "attempt": 0,
    }

    final = graph.invoke(initial)

    print(f"\nPassed   : {final['eval_passed']}")
    print(f"Attempts : {final['attempt']}")
    print(f"Feedback : {final['eval_feedback']}")
    print(f"\nFinal code:\n{final['code']}")
