# Code Generation Pipeline — LangGraph + MCP version
#
# pip install langgraph langchain-mcp-adapters fastmcp
#
# Difference from code_gen_agents.py:
#   The execute node calls run_python_code via a local MCP server (code_exec_server.py)
#   instead of running subprocess directly. The server starts/stops automatically.
#
# Graph: same shape as code_gen_agents.py, but execute_node is async.

import asyncio
import json
from typing import Literal, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph


TASK = "Write a function that takes a list of integers and returns the top-k most frequent elements."
MAX_RETRIES = 3

# Points at code_exec_server.py in the same directory — no config file needed
MCP_SERVERS = {
    "code-executor": {
        "command": "python",
        "args": ["code_exec_server.py"],
        "transport": "stdio",
    }
}


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task: str
    code: str
    exec_result: dict       # {"status": ..., "output": ..., "error": ...}
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


def make_execute_node(run_tool):
    """Closes over the MCP tool so the node can be added to the graph normally."""
    async def execute_node(state: AgentState) -> dict:
        raw = await run_tool.ainvoke({"code": state["code"]})
        # ainvoke may return: str, dict, or list of MCP content items
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
    builder.add_node("generate", generate_node)
    builder.add_node("execute", make_execute_node(run_tool))   # async node
    builder.add_node("evaluate", evaluate_node)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "execute")
    builder.add_conditional_edges("execute", route_after_execute)
    builder.add_conditional_edges("evaluate", route_after_evaluate)
    return builder.compile()


# ── Entry point ────────────────────────────────────────────────────────────────

async def run_pipeline():
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"MCP tools loaded: {[t.name for t in tools]}\n")

    run_tool = next(t for t in tools if t.name == "run_python_code")
    graph = build_graph(run_tool)

    initial: AgentState = {
        "task": TASK,
        "code": "",
        "exec_result": {},
        "eval_passed": False,
        "eval_feedback": "",
        "previous_error": None,
        "attempt": 0,
    }

    final = await graph.ainvoke(initial)

    print(f"Passed   : {final['eval_passed']}")
    print(f"Attempts : {final['attempt']}")
    print(f"Feedback : {final['eval_feedback']}")
    print(f"\nFinal code:\n{final['code']}")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
