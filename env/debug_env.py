"""
debug_env.py
Custom Gymnasium environment where an LLM agent debugs Python code.

State:  current code + error messages + test results + action history
Action: edit line / add print / delete line / run tests / done
Reward: +10 per new test passing, +100 for fully solving, -1 per step, -20 for timeout
"""

import json
import random
from typing import Dict, List, Tuple, Any, Optional
import gymnasium as gym

from env.code_executor import run_code_with_tests_individually


class CodeDebugEnv(gym.Env):
    """
    RL Environment for LLM-based code debugging.

    The agent receives buggy Python code and must fix it by taking
    discrete actions (edit lines, add prints, delete lines) until
    all test cases pass or max steps are exceeded.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, problems: List[Dict], max_steps: int = 10):
        """
        Args:
            problems: List of problem dicts loaded from problems.json
            max_steps: Maximum actions per episode before timeout penalty
        """
        super().__init__()
        self.problems = problems
        self.max_steps = max_steps

        # These are set on reset()
        self.current_problem: Dict = {}
        self.code: str = ""
        self.tests: List[Dict] = []
        self.step_count: int = 0
        self.prev_passed: int = 0
        self.action_history: List[Dict] = []

    def reset(self, problem_idx: Optional[int] = None, seed=None, options=None) -> Tuple[Dict, Dict]:
        """
        Reset environment with a new (or specified) problem.

        Returns:
            state: dict representing the current observation
            info: additional metadata
        """
        super().reset(seed=seed)

        # Pick a random problem or a specific one
        if problem_idx is not None:
            self.current_problem = self.problems[problem_idx]
        else:
            self.current_problem = random.choice(self.problems)

        self.code = self.current_problem["buggy_code"]
        self.tests = self.current_problem["tests"]
        self.step_count = 0
        self.action_history = []

        # Run tests on the initial buggy code
        result = run_code_with_tests_individually(self.code, self.tests)
        self.prev_passed = result["passed"]

        state = self._get_state(result["errors"])
        info = {
            "problem_id": self.current_problem["id"],
            "description": self.current_problem["description"],
            "bug_type": self.current_problem["bug_type"],
            "initial_passed": self.prev_passed,
            "total_tests": len(self.tests),
        }
        return state, info

    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Apply an action and return (new_state, reward, terminated, truncated, info).

        Args:
            action: dict like {"type": "EDIT_LINE", "line": 2, "new_code": "return a + b"}

        Returns:
            state, reward, terminated, truncated, info
        """
        self.step_count += 1
        self.action_history.append(action)

        # Apply the action to the code
        action_applied = self._apply_action(action)

        # Run tests on the updated code
        result = run_code_with_tests_individually(self.code, self.tests)
        curr_passed = result["passed"]
        total_tests = len(self.tests)

        # Compute reward
        reward = self._compute_reward(curr_passed, total_tests, action)

        # Episode ends if all tests pass, or agent says DONE, or max steps reached
        terminated = curr_passed == total_tests or action.get("type") == "DONE"
        truncated = self.step_count >= self.max_steps and not terminated

        self.prev_passed = curr_passed
        state = self._get_state(result["errors"])

        info = {
            "tests_passed": curr_passed,
            "total_tests": total_tests,
            "action_applied": action_applied,
            "solved": curr_passed == total_tests,
            "errors": result["errors"],
        }

        return state, reward, terminated, truncated, info

    def _apply_action(self, action: Dict) -> bool:
        """
        Apply the given action to self.code.

        Returns:
            True if action was applied successfully, False otherwise
        """
        action_type = action.get("type", "")
        lines = self.code.split("\n")

        try:
            if action_type == "EDIT_LINE":
                line_idx = int(action.get("line", 0))
                new_code = action.get("new_code", "")
                if 0 <= line_idx < len(lines):
                    lines[line_idx] = new_code
                    self.code = "\n".join(lines)
                    return True

            elif action_type == "ADD_PRINT":
                line_idx = int(action.get("line", 0))
                var = action.get("var", "")
                indent = "    " if line_idx > 0 and lines[line_idx - 1].startswith("    ") else ""
                print_stmt = f"{indent}print(f'DEBUG {var}={{repr({var})}}')"
                lines.insert(line_idx, print_stmt)
                self.code = "\n".join(lines)
                return True

            elif action_type == "DELETE_LINE":
                line_idx = int(action.get("line", 0))
                if 0 <= line_idx < len(lines):
                    lines.pop(line_idx)
                    self.code = "\n".join(lines)
                    return True

            elif action_type in ("RUN_TESTS", "DONE"):
                return True  # No code change, just observe

        except Exception:
            pass

        return False

    def _compute_reward(self, curr_passed: int, total_tests: int, action: Dict) -> float:
        """
        Compute the reward for the current step.

        Reward components:
        - Progress reward: +10 per additional test now passing
        - Solve bonus: +100 if all tests pass
        - Step penalty: -1 per step (encourages efficiency)
        - Timeout penalty: -20 if max steps hit with unsolved code
        """
        reward = 0.0

        # Progress reward
        reward += (curr_passed - self.prev_passed) * 10

        # Solve bonus
        if curr_passed == total_tests:
            reward += 100.0

        # Step penalty (encourage fewer steps)
        reward -= 1.0

        # Timeout penalty
        if self.step_count >= self.max_steps and curr_passed < total_tests:
            reward -= 20.0

        return reward

    def _get_state(self, errors: List[str]) -> Dict:
        """Build the state dictionary returned to the agent."""
        return {
            "code": self.code,
            "tests": self.tests,
            "error_messages": errors[:3],       # Show up to 3 errors
            "action_history": self.action_history[-5:],  # Last 5 actions
            "step_count": self.step_count,
            "tests_passed": self.prev_passed,
            "total_tests": len(self.tests),
            "lines": self.code.split("\n"),      # For easy line-indexed editing
        }

    def render(self, mode: str = "human") -> None:
        """Print current code state to console."""
        print("\n" + "=" * 50)
        print(f"Step {self.step_count}/{self.max_steps} | "
              f"Tests passing: {self.prev_passed}/{len(self.tests)}")
        print("-" * 50)
        print("CURRENT CODE:")
        for i, line in enumerate(self.code.split("\n")):
            print(f"  {i:2d}: {line}")
        print("=" * 50)


if __name__ == "__main__":
    import json

    with open("data/problems.json") as f:
        problems = json.load(f)

    env = CodeDebugEnv(problems)
    state, info = env.reset(problem_idx=0)

    print(f"Problem: {info['description']}")
    print(f"Bug type: {info['bug_type']}")
    env.render()

    # Simulate a manual fix
    action = {"type": "EDIT_LINE", "line": 1, "new_code": "    return a + b"}
    state, reward, terminated, truncated, info = env.step(action)
    print(f"\nAction: {action}")
    print(f"Reward: {reward} | Tests passed: {info['tests_passed']}/{info['total_tests']}")
    env.render()
