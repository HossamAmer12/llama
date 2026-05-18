# Question 1: Code Generation Pipeline

# You're building a simple autonomous coding agent. Given a natural language task, the system should:

# Generate a Python solution
# Execute it and decide whether to accept or retry based on execution results
# Evaluate the final output for correctness against the original task

# Implement this as a working pipeline. The task input is:
# "Write a function that takes a list of integers and returns the top-k most frequent elements."
# Start with the core loop working end-to-end, then we'll discuss extensions.

# What they're watching for:

# Clean separation between generator, executor, evaluator
# Handling execution failures (syntax error, runtime error, timeout) differently from logical failures
# Retry logic that feeds the actual error back to the generator, not just "try again"
# Evaluator prompt that tests correctness, not just whether code runs

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum


TASK = "Write a function that takes a list of integers and returns the top-k most frequent elements."
MAX_RETRIES = 3


class ExecutionStatus(Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    output: str
    error: str


@dataclass
class GeneratedCode:
    code: str


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Stub returning a hardcoded solution. Replace with a real LLM call later."""
    return """
```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    return [item for item, _ in count.most_common(k)]

# smoke test
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))  # [1, 2]
print(top_k_frequent([1], 1))                   # [1]
```
"""


def _extract_code(text: str) -> str:
    """Pull Python out of markdown fences if present."""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    return text.strip()


def generate_code(task: str, previous_error: str | None = None) -> GeneratedCode:
    prompt = f"Task: {task}\n\nWrite a complete, runnable Python solution."
    if previous_error:
        prompt += (
            f"\n\nYour previous attempt failed with the following error:\n"
            f"{previous_error}\n\n"
            "Fix the issue and return the corrected code."
        )
    raw = _call_llm(prompt)
    return GeneratedCode(code=_extract_code(raw))


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_code(code: str, timeout: int = 10) -> ExecutionResult:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return ExecutionResult(ExecutionStatus.SUCCESS, result.stdout, result.stderr)

        stderr = result.stderr
        if "SyntaxError" in stderr:
            return ExecutionResult(ExecutionStatus.SYNTAX_ERROR, result.stdout, stderr)
        return ExecutionResult(ExecutionStatus.RUNTIME_ERROR, result.stdout, stderr)

    except subprocess.TimeoutExpired:
        return ExecutionResult(ExecutionStatus.TIMEOUT, "", f"Timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_code(task: str, code: str, execution_result: ExecutionResult) -> tuple[bool, str]:
    prompt = (
        f"Task: {task}\n\n"
        f"Code:\n{code}\n\n"
        f"Execution output:\n{execution_result.output}\n\n"
        "Does this code correctly and completely solve the task?\n"
        "Reply with PASS or FAIL followed by a one-sentence explanation."
    )
    response = _call_llm(prompt).strip()
    passed = response.upper().startswith("PASS")
    return passed, response


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(task: str = TASK, max_retries: int = MAX_RETRIES) -> str | None:
    previous_error: str | None = None

    for attempt in range(1, max_retries + 1):
        print(f"\n--- Attempt {attempt}/{max_retries} ---")

        # 1. Generate
        generated = generate_code(task, previous_error)
        print(f"Generated code:\n{generated.code}\n")

        # 2. Execute
        result = execute_code(generated.code)
        print(f"Execution status: {result.status.value}")

        if result.status == ExecutionStatus.TIMEOUT:
            previous_error = result.error
            continue

        if result.status in (ExecutionStatus.SYNTAX_ERROR, ExecutionStatus.RUNTIME_ERROR):
            previous_error = result.error
            print(f"Error:\n{result.error}")
            continue

        # 3. Evaluate — only reached on clean execution
        passed, explanation = evaluate_code(task, generated.code, result)
        print(f"Evaluation: {explanation}")

        if passed:
            print("\nPipeline succeeded.")
            return generated.code

        # Logical failure: feed the evaluator's verdict back to the generator
        previous_error = f"Code ran without errors but failed evaluation: {explanation}"

    print("\nMax retries reached. Pipeline failed.")
    return None


if __name__ == "__main__":
    run_pipeline()
