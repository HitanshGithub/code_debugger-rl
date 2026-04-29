"""
test_runner.py
Standalone utility to run test assertions against Python code.
Can be used independently of the RL environment.
"""

import json
import sys
from typing import List, Dict
from env.code_executor import run_code_with_tests_individually


def evaluate_code(code: str, tests: List[Dict], verbose: bool = False) -> Dict:
    """
    Evaluate code against a list of test cases.

    Args:
        code: Python code string to evaluate
        tests: List of test dicts with 'assertion' and 'description' keys
        verbose: If True, print detailed results

    Returns:
        dict with passed, total, errors, pass_rate
    """
    result = run_code_with_tests_individually(code, tests)
    result["pass_rate"] = result["passed"] / result["total"] if result["total"] > 0 else 0.0

    if verbose:
        print(f"\n{'='*40}")
        print(f"Tests passed: {result['passed']}/{result['total']} ({result['pass_rate']*100:.0f}%)")
        if result["errors"]:
            print("Errors:")
            for err in result["errors"]:
                print(f"  ❌ {err}")
        else:
            print("  ✅ All tests passed!")
        print(f"{'='*40}\n")

    return result


def compare_buggy_vs_fixed(buggy_code: str, fixed_code: str, tests: List[Dict]) -> None:
    """
    Compare test results between buggy and fixed versions of code.

    Args:
        buggy_code: The original buggy code
        fixed_code: The agent's proposed fix
        tests: List of test cases
    """
    print("\n📋 COMPARISON REPORT")
    print("=" * 50)

    print("\n[BUGGY CODE]")
    for i, line in enumerate(buggy_code.split("\n")):
        print(f"  {i}: {line}")
    buggy_result = evaluate_code(buggy_code, tests, verbose=True)

    print("\n[FIXED CODE]")
    for i, line in enumerate(fixed_code.split("\n")):
        print(f"  {i}: {line}")
    fixed_result = evaluate_code(fixed_code, tests, verbose=True)

    improved = fixed_result["passed"] > buggy_result["passed"]
    solved = fixed_result["passed"] == fixed_result["total"]

    print(f"\n{'✅ SOLVED!' if solved else '⚠️  Improved' if improved else '❌ No improvement'}")
    print(f"Before: {buggy_result['passed']}/{buggy_result['total']} tests passing")
    print(f"After:  {fixed_result['passed']}/{fixed_result['total']} tests passing")


if __name__ == "__main__":
    # Test with a sample problem
    buggy = "def add(a, b):\n    return a - b"
    fixed = "def add(a, b):\n    return a + b"
    tests = [
        {"assertion": "assert add(2, 3) == 5", "description": "basic addition"},
        {"assertion": "assert add(0, 0) == 0", "description": "zero case"},
    ]

    compare_buggy_vs_fixed(buggy, fixed, tests)
