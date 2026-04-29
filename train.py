"""
train.py
Main training script for the LLM Code Debugger RL Agent.

Usage:
    python train.py
    python train.py --episodes 50 --model Qwen/Qwen2.5-Coder-1.5B-Instruct
    python train.py --episodes 100 --max_steps 15 --device cuda
"""

import argparse
import json
import os
import sys

# ── Load .env file before anything else ───────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Reading from shell environment only.")
    print("   Install it with: pip install python-dotenv")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.debug_env import CodeDebugEnv
from agent.llm_policy import LLMPolicy
from agent.ppo_trainer import REINFORCETrainer
from utils.logger import EpisodeLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM Code Debugger RL Agent")

    # All defaults fall back to .env values, then hardcoded defaults
    parser.add_argument("--episodes", type=int,
                        default=int(os.getenv("NUM_EPISODES", 20)),
                        help="Number of training episodes (default: 20)")
    parser.add_argument("--max_steps", type=int,
                        default=int(os.getenv("MAX_STEPS", 10)),
                        help="Max steps per episode (default: 10)")
    parser.add_argument("--problems_file", type=str,
                        default=os.getenv("PROBLEMS_FILE", "data/problems.json"),
                        help="Path to problems JSON file")
    parser.add_argument("--trajectories_file", type=str,
                        default=os.getenv("TRAJECTORIES_FILE", "trajectories.jsonl"),
                        help="Output file for trajectory data")
    parser.add_argument("--log_file", type=str,
                        default=os.getenv("LOG_FILE", "training_log.csv"),
                        help="Output CSV log file")
    parser.add_argument("--print_every", type=int,
                        default=int(os.getenv("PRINT_EVERY", 5)),
                        help="Print summary every N episodes (default: 5)")
    parser.add_argument("--gamma", type=float,
                        default=float(os.getenv("GAMMA", 0.99)),
                        help="Discount factor for returns (default: 0.99)")
    parser.add_argument("--model", type=str,
                        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
                        help="HuggingFace model ID to use as policy")
    parser.add_argument("--device", type=str,
                        default=os.getenv("DEVICE") or None,
                        help="Device: 'cuda', 'cpu', or None for auto-detect")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load problems ──────────────────────────────────────────────
    print(f"\n🔧 Loading problems from {args.problems_file}...")
    with open(args.problems_file) as f:
        problems = json.load(f)
    print(f"   Loaded {len(problems)} problems.")

    # ── Initialize components ──────────────────────────────────────
    print(f"🤖 Initializing LLM policy: {args.model}")
    policy = LLMPolicy(model_name=args.model, device=args.device)

    env = CodeDebugEnv(problems, max_steps=args.max_steps)
    trainer = REINFORCETrainer(gamma=args.gamma, output_path=args.trajectories_file)
    logger = EpisodeLogger(log_path=args.log_file, print_every=args.print_every)

    # ── Training loop ──────────────────────────────────────────────
    print(f"\n🚀 Starting training for {args.episodes} episodes...\n")
    print(f"{'─'*60}")
    print(f"  {'Ep':>4}  {'Status':<12}  {'Reward':>8}  {'Steps':>6}  {'Tests':<10}  Bug Type")
    print(f"{'─'*60}")

    for ep in range(1, args.episodes + 1):
        episode = trainer.run_episode(env, policy)
        logger.log(episode)
        trainer.print_episode_summary(episode)

    # ── Final report ───────────────────────────────────────────────
    stats = logger.final_report()

    print(f"\n📁 Trajectories saved to: {args.trajectories_file}")
    print(f"📊 Training log saved to:  {args.log_file}")
    print("\nTo evaluate the agent, run:")
    print("  python evaluate.py\n")


if __name__ == "__main__":
    main()
