"""
ppo_trainer.py
REINFORCE (policy gradient) trainer for the LLM debugging agent.

Since the LLM IS the policy (not a neural net we backprop through),
we implement REINFORCE to:
  1. Collect trajectories (state, action, reward) per episode
  2. Compute discounted returns G_t
  3. Log everything for future fine-tuning or analysis
  4. Save trajectories as JSONL for supervised fine-tuning later
"""

import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Transition:
    """Single (state, action, reward) transition in a trajectory."""
    step: int
    state_summary: str      # Compact string summary of state (not full state for storage)
    action: Dict
    reward: float
    tests_passed: int
    total_tests: int


@dataclass
class Episode:
    """Full episode trajectory with computed returns."""
    episode_id: int
    problem_id: int
    problem_description: str
    bug_type: str
    initial_code: str
    final_code: str
    transitions: List[Transition]
    total_reward: float
    discounted_return: float
    steps_taken: int
    tests_passed_final: int
    total_tests: int
    solved: bool


class REINFORCETrainer:
    """
    Collects trajectories from the LLM+env interaction and logs them.

    Since we can't backpropagate through the LLM directly, this trainer:
    - Tracks episode statistics for monitoring
    - Saves trajectories as JSONL (useful for fine-tuning later)
    - Computes discounted returns for analysis
    - Reports running stats every N episodes
    """

    def __init__(self, gamma: float = 0.99, output_path: str = "trajectories.jsonl"):
        """
        Args:
            gamma: Discount factor for computing returns
            output_path: File to save trajectory data for fine-tuning
        """
        self.gamma = gamma
        self.output_path = output_path
        self.episodes: List[Episode] = []
        self.episode_count = 0

        # Clear/create output file
        with open(self.output_path, "w") as f:
            pass  # Empty the file

    def run_episode(self, env, policy, problem_idx: int = None) -> Episode:
        """
        Run a single full episode: reset env → agent acts → collect trajectory.

        Args:
            env: CodeDebugEnv instance
            policy: LLMPolicy instance
            problem_idx: specific problem index (or None for random)

        Returns:
            Episode dataclass with full trajectory
        """
        self.episode_count += 1
        state, info = env.reset(problem_idx=problem_idx)
        initial_code = state["code"]

        transitions = []
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # LLM selects action
            action = policy.select_action(state)

            # Environment steps
            next_state, reward, terminated, truncated, step_info = env.step(action)

            # Record transition
            transition = Transition(
                step=state["step_count"],
                state_summary=f"code_lines={len(state['lines'])} passed={state['tests_passed']}/{state['total_tests']}",
                action=action,
                reward=reward,
                tests_passed=step_info["tests_passed"],
                total_tests=step_info["total_tests"],
            )
            transitions.append(transition)
            total_reward += reward
            state = next_state

        # Compute discounted return from the trajectory
        discounted_return = self._compute_discounted_return([t.reward for t in transitions])

        episode = Episode(
            episode_id=self.episode_count,
            problem_id=info["problem_id"],
            problem_description=info["description"],
            bug_type=info["bug_type"],
            initial_code=initial_code,
            final_code=state["code"],
            transitions=transitions,
            total_reward=total_reward,
            discounted_return=discounted_return,
            steps_taken=len(transitions),
            tests_passed_final=state["tests_passed"],
            total_tests=state["total_tests"],
            solved=state["tests_passed"] == state["total_tests"],
        )

        self.episodes.append(episode)
        self._save_episode(episode)
        return episode

    def _compute_discounted_return(self, rewards: List[float]) -> float:
        """
        Compute the discounted return G_0 = sum(gamma^t * r_t).

        Args:
            rewards: list of rewards per step

        Returns:
            Discounted return from step 0
        """
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
        return G

    def _save_episode(self, episode: Episode) -> None:
        """Append episode to JSONL file for later fine-tuning."""
        record = {
            "episode_id": episode.episode_id,
            "problem_id": episode.problem_id,
            "description": episode.problem_description,
            "bug_type": episode.bug_type,
            "solved": episode.solved,
            "steps": episode.steps_taken,
            "total_reward": episode.total_reward,
            "discounted_return": episode.discounted_return,
            "initial_code": episode.initial_code,
            "final_code": episode.final_code,
            "trajectory": [
                {
                    "step": t.step,
                    "action": t.action,
                    "reward": t.reward,
                    "tests_passed": t.tests_passed,
                }
                for t in episode.transitions
            ],
        }
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_stats(self, last_n: int = None) -> Dict:
        """
        Compute summary statistics over all (or last N) episodes.

        Args:
            last_n: if set, only consider the last N episodes

        Returns:
            dict with solve_rate, avg_reward, avg_steps, avg_return
        """
        episodes = self.episodes if last_n is None else self.episodes[-last_n:]
        if not episodes:
            return {}

        return {
            "total_episodes": len(self.episodes),
            "solve_rate": sum(e.solved for e in episodes) / len(episodes),
            "avg_reward": sum(e.total_reward for e in episodes) / len(episodes),
            "avg_steps": sum(e.steps_taken for e in episodes) / len(episodes),
            "avg_discounted_return": sum(e.discounted_return for e in episodes) / len(episodes),
            "solved_count": sum(e.solved for e in episodes),
            "total_count": len(episodes),
        }

    def print_episode_summary(self, episode: Episode) -> None:
        """Print a one-line episode summary to console."""
        status = "✅ SOLVED" if episode.solved else "❌ FAILED"
        print(
            f"  Ep {episode.episode_id:3d} | {status} | "
            f"Reward: {episode.total_reward:6.1f} | "
            f"Steps: {episode.steps_taken:2d} | "
            f"Tests: {episode.tests_passed_final}/{episode.total_tests} | "
            f"[{episode.bug_type}]"
        )
