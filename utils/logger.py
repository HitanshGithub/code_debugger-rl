"""
logger.py
Logs training episodes to CSV and prints live console summaries.
"""

import csv
import os
from typing import List, Dict
from datetime import datetime


class EpisodeLogger:
    """
    Logs RL training episodes to a CSV file and prints periodic console summaries.
    """

    FIELDNAMES = [
        "episode", "problem_id", "bug_type", "solved",
        "steps", "total_reward", "discounted_return",
        "tests_passed", "total_tests", "timestamp"
    ]

    def __init__(self, log_path: str = "training_log.csv", print_every: int = 5):
        """
        Args:
            log_path: Path to write CSV log
            print_every: Print summary table every N episodes
        """
        self.log_path = log_path
        self.print_every = print_every
        self.records: List[Dict] = []

        # Initialize CSV with headers
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def log(self, episode) -> None:
        """
        Log a completed episode.

        Args:
            episode: Episode dataclass from ppo_trainer.py
        """
        record = {
            "episode": episode.episode_id,
            "problem_id": episode.problem_id,
            "bug_type": episode.bug_type,
            "solved": int(episode.solved),
            "steps": episode.steps_taken,
            "total_reward": round(episode.total_reward, 2),
            "discounted_return": round(episode.discounted_return, 2),
            "tests_passed": episode.tests_passed_final,
            "total_tests": episode.total_tests,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.records.append(record)

        # Write to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(record)

        # Print summary periodically
        if episode.episode_id % self.print_every == 0:
            self._print_summary()

    def _print_summary(self) -> None:
        """Print a rolling summary of the last N episodes."""
        last = self.records[-self.print_every:]
        if not last:
            return

        solve_rate = sum(r["solved"] for r in last) / len(last) * 100
        avg_reward = sum(r["total_reward"] for r in last) / len(last)
        avg_steps = sum(r["steps"] for r in last) / len(last)

        total_solved = sum(r["solved"] for r in self.records)
        total_eps = len(self.records)

        print(f"\n{'─'*60}")
        print(f"  📊 TRAINING SUMMARY  (Episodes 1–{total_eps})")
        print(f"{'─'*60}")
        print(f"  Overall solve rate : {total_solved}/{total_eps} ({total_solved/total_eps*100:.1f}%)")
        print(f"  Last {self.print_every} ep solve rate: {solve_rate:.1f}%")
        print(f"  Last {self.print_every} avg reward   : {avg_reward:.2f}")
        print(f"  Last {self.print_every} avg steps    : {avg_steps:.1f}")
        print(f"{'─'*60}\n")

    def final_report(self) -> Dict:
        """Print and return the final training report."""
        if not self.records:
            return {}

        total = len(self.records)
        solved = sum(r["solved"] for r in self.records)
        avg_reward = sum(r["total_reward"] for r in self.records) / total
        avg_steps = sum(r["steps"] for r in self.records) / total

        # Stats by bug type
        by_bug = {}
        for r in self.records:
            bt = r["bug_type"]
            if bt not in by_bug:
                by_bug[bt] = {"total": 0, "solved": 0}
            by_bug[bt]["total"] += 1
            by_bug[bt]["solved"] += r["solved"]

        print("\n" + "=" * 60)
        print("  🏁 FINAL TRAINING REPORT")
        print("=" * 60)
        print(f"  Total episodes  : {total}")
        print(f"  Solved          : {solved} ({solved/total*100:.1f}%)")
        print(f"  Avg reward      : {avg_reward:.2f}")
        print(f"  Avg steps/ep    : {avg_steps:.1f}")
        print(f"\n  By bug type:")
        for bt, stats in sorted(by_bug.items()):
            rate = stats["solved"] / stats["total"] * 100
            print(f"    {bt:<25} {stats['solved']}/{stats['total']} ({rate:.0f}%)")
        print("=" * 60)

        return {
            "total_episodes": total,
            "solve_rate": solved / total,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "by_bug_type": by_bug,
        }
