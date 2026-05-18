# MCP server — exposes run_python_code as a local tool
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
