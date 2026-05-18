# MCP server — exposes run_python_code and retrieve_code_examples as local tools
#
# pip install fastmcp
#
# Run standalone to test:
#   python code_exec_server.py
#
# Used automatically as a subprocess by code_gen_mcp.py

import subprocess
import sys

from fastmcp import FastMCP


mcp = FastMCP("code-executor")


# ── Toy knowledge base ─────────────────────────────────────────────────────────
# Each entry: (keywords, title, code_snippet)
# In production replace with a real vector store (FAISS, Chroma, etc.)

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
top2 = heapq.nlargest(2, nums)   # [9, 6]
bot2 = heapq.nsmallest(2, nums)  # [1, 1]
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
# {"a": [1, 3], "b": [2]}
""",
    ),
]


@mcp.tool()
def retrieve_code_examples(query: str, top_k: int = 2) -> str:
    """Return the most relevant code examples from the knowledge base for a given query.

    Uses keyword matching — replace with embedding similarity for production.
    """
    query_lower = query.lower()
    scored = [
        (sum(kw in query_lower for kw in keywords), title, snippet)
        for keywords, title, snippet in _KB
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [item for item in scored if item[0] > 0][:top_k]

    if not hits:
        return "No relevant examples found."

    parts = []
    for _, title, snippet in hits:
        parts.append(f"### {title}\n```python\n{snippet.strip()}\n```")
    return "\n\n".join(parts)


@mcp.tool()
def run_python_code(code: str, timeout: int = 10) -> dict:
    """Execute Python code in a subprocess and return status + output."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return {"status": "success", "output": result.stdout, "error": ""}
        status = "syntax_error" if "SyntaxError" in result.stderr else "runtime_error"
        return {"status": status, "output": "", "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output": "", "error": f"Timed out after {timeout}s"}


if __name__ == "__main__":
    mcp.run()
