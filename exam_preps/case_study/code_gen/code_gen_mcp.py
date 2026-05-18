# Code Generation Pipeline — LangGraph + MCP + Guardrails + RAG
#
# pip install langgraph langchain-mcp-adapters fastmcp
#
# Graph:
#   START → input_guard → retrieve → generate → code_guard → execute → evaluate → END
#                 ↓ UNSAFE                              ↓ UNSAFE
#                END                                    END

import asyncio
import json
from typing import Literal, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from rag_eval import RagEvaluator, print_report


TASK = "Write a function that takes a list of integers and returns the top-k most frequent elements."
MAX_RETRIES = 3

MCP_SERVERS = {
    "code-executor": {
        "command": "python",
        "args": ["code_exec_server.py"],
        "transport": "stdio",
    }
}

# Patterns the code guardrail treats as unsafe
_DANGEROUS = ["os.system", "subprocess", "shutil.rmtree", "eval(", "exec(", "__import__"]


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task: str
    code: str
    exec_result: dict
    eval_passed: bool
    eval_feedback: str
    previous_error: str | None
    attempt: int
    guard_blocked: bool         # set to True by either guardrail on failure
    guard_reason: str
    retrieved_context: str      # injected into the generation prompt by retrieve_node


# ── LLM stubs ──────────────────────────────────────────────────────────────────

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

    # Evaluator: PASS only when code is actually correct
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
    """Stub for both guardrail checks — swap in a real model here.

    Distinguishes between the two guard calls by prompt content:
      - Input guard prompt contains "appropriate coding task"
      - Code guard prompt contains "dangerous operations"

    Input guard:  UNSAFE for tasks asking to delete/hack/exploit.
    Code guard:   UNSAFE if generated code contains a known dangerous pattern.
    """
    p = prompt.lower()

    # ── Input guard ────────────────────────────────────────────────────────
    if "appropriate coding task" in p:
        if any(w in p for w in ["delete", "rm -rf", "drop table", "hack", "exploit", "malware"]):
            return "UNSAFE — task requests a potentially harmful operation."
        return "SAFE — standard algorithmic coding task, no harmful intent detected."

    # ── Code guard ─────────────────────────────────────────────────────────
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


# ── Guardrail nodes ────────────────────────────────────────────────────────────

def input_guard_node(state: AgentState) -> dict:
    """Validates the task before any code is generated."""
    prompt = (
        f"Is this a safe, appropriate coding task?\n\nTask: {state['task']}\n\n"
        "Reply SAFE or UNSAFE followed by a one-sentence reason."
    )
    response = _call_guard_llm(prompt).strip()
    blocked = not response.upper().startswith("SAFE")
    verdict = "BLOCKED" if blocked else "PASSED"
    print(f"[input_guard]  {verdict} | {response}")
    return {"guard_blocked": blocked, "guard_reason": response}


def code_guard_node(state: AgentState) -> dict:
    """Validates generated code before it is executed via MCP."""
    prompt = (
        "Does the following code contain dangerous operations "
        "(file deletion, arbitrary shell commands, network calls, eval/exec)?\n\n"
        f"Code:\n{state['code']}\n\n"
        "Reply SAFE or UNSAFE followed by a one-sentence reason."
    )
    response = _call_guard_llm(prompt).strip()
    blocked = not response.upper().startswith("SAFE")
    verdict = "BLOCKED" if blocked else "PASSED"
    print(f"[code_guard]   {verdict} | {response}")
    return {"guard_blocked": blocked, "guard_reason": response}


# ── Core nodes ─────────────────────────────────────────────────────────────────

def make_retrieve_node(retrieve_tool):
    """Fetches relevant examples from the MCP knowledge base before generation."""
    async def retrieve_node(state: AgentState) -> dict:
        raw = await retrieve_tool.ainvoke({"query": state["task"]})
        # Normalise the same way as execute_node
        if isinstance(raw, list):
            context = raw[0].text if hasattr(raw[0], "text") else raw[0].get("text", str(raw[0]))
        else:
            context = str(raw)
        print(f"[retrieve] found context ({len(context)} chars)")
        return {"retrieved_context": context}
    return retrieve_node


def generate_node(state: AgentState) -> dict:
    prompt = f"Task: {state['task']}\n\nWrite a complete, runnable Python solution."
    if state["retrieved_context"]:
        prompt += f"\n\nRelevant examples for reference:\n{state['retrieved_context']}"
    if state["previous_error"]:
        prompt += f"\n\nPrevious error:\n{state['previous_error']}\n\nFix it."
    raw = _call_llm(prompt)
    return {"code": _extract_code(raw), "attempt": state["attempt"] + 1}


def make_execute_node(run_tool):
    async def execute_node(state: AgentState) -> dict:
        raw = await run_tool.ainvoke({"code": state["code"]})
        if isinstance(raw, dict):
            result = raw
        elif isinstance(raw, str):
            result = json.loads(raw)
        elif isinstance(raw, list):
            text = raw[0].text if hasattr(raw[0], "text") else raw[0].get("text", str(raw[0]))
            result = json.loads(text)
        else:
            result = {"status": "runtime_error", "output": "", "error": str(raw)}
        updates: dict = {"exec_result": result}
        if result["status"] != "success":
            updates["previous_error"] = result["error"]
        return updates
    return execute_node


def evaluate_node(state: AgentState) -> dict:
    prompt = (
        f"Task: {state['task']}\n\nCode:\n{state['code']}\n\n"
        f"Output:\n{state['exec_result'].get('output', '')}\n\n"
        "Does this correctly solve the task? Reply PASS or FAIL + one sentence."
    )
    response = _call_llm(prompt).strip()
    passed = response.upper().startswith("PASS")
    updates: dict = {"eval_passed": passed, "eval_feedback": response}
    if not passed:
        updates["previous_error"] = f"Evaluation failed: {response}"
    return updates


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_input_guard(state: AgentState) -> Literal["retrieve", "__end__"]:
    return END if state["guard_blocked"] else "retrieve"


def route_after_code_guard(state: AgentState) -> Literal["execute", "__end__"]:
    return END if state["guard_blocked"] else "execute"


def route_after_execute(state: AgentState) -> Literal["evaluate", "generate", "__end__"]:
    if state["exec_result"].get("status") == "success":
        return "evaluate"
    if state["attempt"] >= MAX_RETRIES:
        return END
    return "generate"


def route_after_evaluate(state: AgentState) -> Literal["__end__", "generate"]:
    if state["eval_passed"] or state["attempt"] >= MAX_RETRIES:
        return END
    return "generate"


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph(run_tool, retrieve_tool):
    builder = StateGraph(AgentState)

    builder.add_node("input_guard", input_guard_node)
    builder.add_node("retrieve", make_retrieve_node(retrieve_tool))  # RAG node
    builder.add_node("generate", generate_node)
    builder.add_node("code_guard", code_guard_node)
    builder.add_node("execute", make_execute_node(run_tool))
    builder.add_node("evaluate", evaluate_node)

    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges("input_guard", route_after_input_guard)
    builder.add_edge("retrieve", "generate")            # retrieval always feeds into generate
    builder.add_edge("generate", "code_guard")
    builder.add_conditional_edges("code_guard", route_after_code_guard)
    builder.add_conditional_edges("execute", route_after_execute)
    builder.add_conditional_edges("evaluate", route_after_evaluate)

    return builder.compile()


# ── Graph export ───────────────────────────────────────────────────────────────

def save_graph(graph, path: str = "code_gen_mcp_graph"):
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


# ── Entry point ────────────────────────────────────────────────────────────────

def _make_initial_state(task: str) -> AgentState:
    return AgentState(
        task=task,
        code="",
        exec_result={},
        eval_passed=False,
        eval_feedback="",
        previous_error=None,
        attempt=0,
        guard_blocked=False,
        guard_reason="",
        retrieved_context="",
    )


async def run_pipeline(task: str = TASK, graph=None):
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"MCP tools loaded: {[t.name for t in tools]}\n")

    run_tool = next(t for t in tools if t.name == "run_python_code")
    retrieve_tool = next(t for t in tools if t.name == "retrieve_code_examples")

    if graph is None:
        graph = build_graph(run_tool, retrieve_tool)
        save_graph(graph)

    final = await graph.ainvoke(_make_initial_state(task))

    if final["guard_blocked"]:
        print(f"\nPipeline BLOCKED by guardrail: {final['guard_reason']}")
        return

    print(f"\nPassed   : {final['eval_passed']}")
    print(f"Attempts : {final['attempt']}")
    print(f"Feedback : {final['eval_feedback']}")
    print(f"\nFinal code:\n{final['code']}")

    # ── RAG evaluation ────────────────────────────────────────────────────────
    if final["retrieved_context"] and final["code"]:
        evaluator = RagEvaluator()
        results = evaluator.evaluate_all(
            query=final["task"],
            context=final["retrieved_context"],
            answer=final["code"],
        )
        print_report(results)


async def main():
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    run_tool = next(t for t in tools if t.name == "run_python_code")
    retrieve_tool = next(t for t in tools if t.name == "retrieve_code_examples")
    graph = build_graph(run_tool, retrieve_tool)
    save_graph(graph)

    # ── Scenario 1: safe task — exercises 3-attempt retry loop ───────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 1 — Safe task (3-attempt retry loop)")
    print("=" * 60)
    await run_pipeline(task=TASK, graph=graph)

    # ── Scenario 2: unsafe task — input guard blocks immediately ─────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 2 — Unsafe task (input guard blocks)")
    print("=" * 60)
    unsafe_task = "Write a script that deletes all .py files in the current directory."
    await run_pipeline(task=unsafe_task, graph=graph)

    # ── Scenario 3: safe task but generated code is unsafe ───────────────────
    print("\n" + "=" * 60)
    print("  SCENARIO 3 — Code guard blocks dangerous generated code")
    print("=" * 60)
    # Temporarily patch _call_llm to return code with os.system so code_guard fires
    import code_gen_mcp as _self
    _original_llm = _self._call_llm

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
    _self._call_llm = _dangerous_llm
    await run_pipeline(task=TASK, graph=graph)
    _self._call_llm = _original_llm          # restore


if __name__ == "__main__":
    asyncio.run(main())
