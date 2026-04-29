"""
code_executor.py
Safely executes Python code in a subprocess sandbox with timeout.
"""

import subprocess
import sys
import json
import tempfile
import os
from typing import List, Dict


def run_code_with_tests(code: str, tests: List[Dict]) -> Dict:
    """
    Safely run code + assertions in a subprocess.

    Args:
        code: The Python code string to execute
        tests: List of test dicts with 'assertion' keys

    Returns:
        dict with keys: passed (int), total (int), errors (list of str)
    """
    assertions = "\n".join(t["assertion"] for t in tests)
    full_code = f"{code}\n\n# --- TESTS ---\n{assertions}\n"

    # Write to temp file and run in subprocess for safety
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return {"passed": len(tests), "total": len(tests), "errors": []}
        else:
            error_msg = result.stderr.strip()
            return {"passed": 0, "total": len(tests), "errors": [error_msg]}
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": len(tests), "errors": ["Timeout: code took too long"]}
    except Exception as e:
        return {"passed": 0, "total": len(tests), "errors": [str(e)]}
    finally:
        os.unlink(tmp_path)


def run_code_with_tests_individually(code: str, tests: List[Dict]) -> Dict:
    """
    Run each test assertion individually to count partial passes.

    Args:
        code: The Python code string to execute
        tests: List of test dicts with 'assertion' keys

    Returns:
        dict with keys: passed (int), total (int), errors (list of str)
    """
    passed = 0
    errors = []

    for test in tests:
        full_code = f"{code}\n\n{test['assertion']}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                passed += 1
            else:
                errors.append(f"[{test['description']}] {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            errors.append(f"[{test['description']}] Timeout")
        except Exception as e:
            errors.append(f"[{test['description']}] {str(e)}")
        finally:
            os.unlink(tmp_path)

    return {"passed": passed, "total": len(tests), "errors": errors}


if __name__ == "__main__":
    # Quick test
    code = "def add(a, b):\n    return a + b"
    tests = [
        {"assertion": "assert add(2, 3) == 5", "description": "basic"},
        {"assertion": "assert add(0, 0) == 0", "description": "zero"},
    ]
    result = run_code_with_tests_individually(code, tests)
    print(json.dumps(result, indent=2))
