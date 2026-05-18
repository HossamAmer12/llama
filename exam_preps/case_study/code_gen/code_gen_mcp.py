# Code Generation Pipeline — LangGraph + MCP + Guardrails
#
# pip install langgraph langchain-mcp-adapters fastmcp
#
# Graph:
#   START → input_guard → generate → code_guard → execute → evaluate → END
#                 ↓ UNSAFE                 ↓ UNSAFE
#                END                       END

import asyncio
import json
from typing import Literal, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph


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
    guard_blocked: bool     # set to True by either guardrail on failure
    guard_reason: str


# ── LLM stubs ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Stub for the generator / evaluator — swap in a real model here."""
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


def _call_guard_llm(prompt: str) -> str:
    """Stub for both guardrail checks — swap in a real model here.

    Input guard:  returns SAFE for normal coding tasks, UNSAFE otherwise.
    Code guard:   returns SAFE unless a dangerous pattern is detected in the prompt.
    """
    lowered = prompt.lower()
    if any(p in lowered for p in [p.lower() for p in _DANGEROUS]):
        return "UNSAFE — code contains potentially dangerous operations."
    return "SAFE"


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
    if blocked:
        print(f"[input_guard] BLOCKED: {response}")
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
    if blocked:
        print(f"[code_guard] BLOCKED: {response}")
    return {"guard_blocked": blocked, "guard_reason": response}


# ── Core nodes ─────────────────────────────────────────────────────────────────

def generate_node(state: AgentState) -> dict:
    prompt = f"Task: {state['task']}\n\nWrite a complete, runnable Python solution."
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

def route_after_input_guard(state: AgentState) -> Literal["generate", "__end__"]:
    return END if state["guard_blocked"] else "generate"


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

def build_graph(run_tool):
    builder = StateGraph(AgentState)

    builder.add_node("input_guard", input_guard_node)
    builder.add_node("generate", generate_node)
    builder.add_node("code_guard", code_guard_node)
    builder.add_node("execute", make_execute_node(run_tool))
    builder.add_node("evaluate", evaluate_node)

    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges("input_guard", route_after_input_guard)
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

async def run_pipeline():
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"MCP tools loaded: {[t.name for t in tools]}\n")

    run_tool = next(t for t in tools if t.name == "run_python_code")
    graph = build_graph(run_tool)
    save_graph(graph)

    initial: AgentState = {
        "task": TASK,
        "code": "",
        "exec_result": {},
        "eval_passed": False,
        "eval_feedback": "",
        "previous_error": None,
        "attempt": 0,
        "guard_blocked": False,
        "guard_reason": "",
    }

    final = await graph.ainvoke(initial)

    if final["guard_blocked"]:
        print(f"Pipeline blocked by guardrail: {final['guard_reason']}")
        return

    print(f"Passed   : {final['eval_passed']}")
    print(f"Attempts : {final['attempt']}")
    print(f"Feedback : {final['eval_feedback']}")
    print(f"\nFinal code:\n{final['code']}")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
