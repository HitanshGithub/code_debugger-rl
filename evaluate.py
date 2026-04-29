"""
evaluate.py
Evaluate the LLM Code Debugger agent on held-out problems.

Shows before/after code diffs, solve rate, and step efficiency.

Usage:
    python evaluate.py
    python evaluate.py --problems_file data/problems.json --held_out 5
"""

import argparse
import json
import os
import sys
import difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.debug_env import CodeDebugEnv
from agent.llm_policy import LLMPolicy
from agent.ppo_trainer import REINFORCETrainer


def show_diff(original: str, modified: str, label_a: str = "BUGGY", label_b: str = "FIXED") -> None:
    """Show a colorized unified diff between two code strings."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        orig_lines, mod_lines,
        fromfile=label_a, tofile=label_b, lineterm=""
    ))

    if not diff:
        print("  (no changes made)")
        return

    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            print(f"  \033[92m{line}\033[0m")   # green
        elif line.startswith("-") and not line.startswith("---"):
            print(f"  \033[91m{line}\033[0m")   # red
        else:
            print(f"  {line}")


def evaluate(problems_file: str, held_out: int, max_steps: int) -> None:
    """
    Run the agent on held-out problems and report results.

    Args:
        problems_file: Path to problems JSON
        held_out: Number of problems at the end of the list to use for eval
        max_steps: Max steps per episode
    """
    with open(problems_file) as f:
        all_problems = json.load(f)

    eval_problems = all_problems[-held_out:]
    print(f"\n🧪 Evaluating on {len(eval_problems)} held-out problems...\n")

    policy = LLMPolicy()
    env = CodeDebugEnv(eval_problems, max_steps=max_steps)
    trainer = REINFORCETrainer()

    results = []

    for i, problem in enumerate(eval_problems):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{len(eval_problems)}: {problem['description']}")
        print(f"Bug type: {problem['bug_type']}")
        print(f"{'='*60}")

        episode = trainer.run_episode(env, policy, problem_idx=i)

        print(f"\n📝 CODE DIFF:")
        show_diff(episode.initial_code, episode.final_code)

        print(f"\n📊 RESULT:")
        print(f"  Solved:       {'✅ YES' if episode.solved else '❌ NO'}")
        print(f"  Steps taken:  {episode.steps_taken}")
        print(f"  Tests passed: {episode.tests_passed_final}/{episode.total_tests}")
        print(f"  Total reward: {episode.total_reward:.1f}")

        # Show actions taken
        print(f"\n🔍 ACTIONS TAKEN:")
        for t in episode.transitions:
            print(f"  Step {t.step+1}: {json.dumps(t.action)} → reward={t.reward:.1f}")

        results.append(episode)

    # Summary
    solved = sum(e.solved for e in results)
    total = len(results)
    avg_steps = sum(e.steps_taken for e in results) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"🏁 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Solve rate    : {solved}/{total} ({solved/total*100:.1f}%)")
    print(f"  Avg steps     : {avg_steps:.1f}")
    print(f"  Avg reward    : {sum(e.total_reward for e in results)/total:.2f}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM Code Debugger Agent")
    parser.add_argument("--problems_file", type=str, default="data/problems.json")
    parser.add_argument("--held_out", type=int, default=5,
                        help="Number of held-out problems from end of list (default: 5)")
    parser.add_argument("--max_steps", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.problems_file, args.held_out, args.max_steps)
