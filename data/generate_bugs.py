"""
generate_bugs.py
Utility to auto-generate additional buggy code problems.
Injects common bugs into correct Python functions.

Usage:
    python data/generate_bugs.py
"""

import json
import random
import os


def inject_wrong_operator(code: str) -> tuple:
    """Replace +, -, *, / with wrong operators."""
    ops = [("+", "-"), ("-", "+"), ("*", "+"), ("/", "*")]
    for correct, wrong in ops:
        if f" {correct} " in code:
            buggy = code.replace(f" {correct} ", f" {wrong} ", 1)
            return buggy, "wrong_operator"
    return None, None


def inject_off_by_one(code: str) -> tuple:
    """Change range(n) to range(n-1) or range(n+1) to range(n)."""
    if "range(n)" in code:
        return code.replace("range(n)", "range(n - 1)", 1), "off_by_one"
    if "range(len(lst))" in code:
        return code.replace("range(len(lst))", "range(len(lst) - 1)", 1), "off_by_one"
    return None, None


SAMPLE_FUNCTIONS = [
    {
        "description": "Square a number",
        "correct_code": "def square(n):\n    return n * n",
        "tests": [
            {"assertion": "assert square(4) == 16", "description": "square 4"},
            {"assertion": "assert square(3) == 9", "description": "square 3"},
            {"assertion": "assert square(0) == 0", "description": "square 0"},
        ]
    },
    {
        "description": "Check if a number is negative",
        "correct_code": "def is_negative(n):\n    if n < 0:\n        return True\n    return False",
        "tests": [
            {"assertion": "assert is_negative(-1) == True", "description": "negative"},
            {"assertion": "assert is_negative(1) == False", "description": "positive"},
            {"assertion": "assert is_negative(0) == False", "description": "zero"},
        ]
    },
    {
        "description": "Get string length without built-in",
        "correct_code": "def str_length(s):\n    count = 0\n    for c in s:\n        count += 1\n    return count",
        "tests": [
            {"assertion": "assert str_length('hello') == 5", "description": "basic"},
            {"assertion": "assert str_length('') == 0", "description": "empty string"},
            {"assertion": "assert str_length('ab') == 2", "description": "two chars"},
        ]
    },
]


def generate_additional_problems(base_id: int = 21, count: int = 5) -> list:
    """Generate additional buggy problems from sample functions."""
    problems = []
    problem_id = base_id

    for func in random.sample(SAMPLE_FUNCTIONS, min(count, len(SAMPLE_FUNCTIONS))):
        correct_code = func["correct_code"]

        # Try to inject a bug
        buggy_code, bug_type = inject_wrong_operator(correct_code)
        if not buggy_code:
            buggy_code, bug_type = inject_off_by_one(correct_code)
        if not buggy_code:
            continue

        problems.append({
            "id": problem_id,
            "description": func["description"],
            "buggy_code": buggy_code,
            "correct_code": correct_code,
            "bug_type": bug_type,
            "tests": func["tests"]
        })
        problem_id += 1

    return problems


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "problems.json")

    with open(output_path) as f:
        existing = json.load(f)

    new_problems = generate_additional_problems(base_id=len(existing) + 1, count=5)

    if new_problems:
        all_problems = existing + new_problems
        with open(output_path, "w") as f:
            json.dump(all_problems, f, indent=2)
        print(f"✅ Added {len(new_problems)} new problems. Total: {len(all_problems)}")
    else:
        print("⚠️  No new problems generated.")
