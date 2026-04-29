# 🤖 LLM Code Debugger RL Agent

An RL environment where a Claude LLM agent receives buggy Python code
and learns to fix it by taking sequential debugging actions.

## Architecture

```
Buggy Code + Tests
        │
        ▼
  ┌─────────────┐
  │  LLM Policy │  ← Claude (claude-sonnet-4-20250514)
  │  (Agent)    │
  └─────┬───────┘
        │  Action: EDIT_LINE / ADD_PRINT / DELETE_LINE / DONE
        ▼
  ┌─────────────┐
  │ CodeDebugEnv│  ← Custom Gymnasium Environment
  │  (Env)      │
  └─────┬───────┘
        │  State + Reward
        ▼
  ┌─────────────┐
  │  REINFORCE  │  ← Trajectory Collection + Logging
  │  Trainer    │
  └─────────────┘
```

## Reward Function

| Event                          | Reward  |
|-------------------------------|---------|
| Each additional test passing  | +10     |
| All tests pass (solved)       | +100    |
| Each step taken               | -1      |
| Timeout (max steps exceeded)  | -20     |

## Action Space

```json
{"type": "EDIT_LINE",   "line": 1, "new_code": "    return a + b"}
{"type": "ADD_PRINT",   "line": 2, "var": "result"}
{"type": "DELETE_LINE", "line": 3}
{"type": "RUN_TESTS"}
{"type": "DONE"}
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here
```

## Run Training

```bash
# Train for 20 episodes (default)
python train.py

# Train for 100 episodes with custom settings
python train.py --episodes 100 --max_steps 15 --print_every 10
```

## Evaluate

```bash
# Evaluate on last 5 held-out problems
python evaluate.py

# Evaluate on more problems
python evaluate.py --held_out 8
```

## Generate More Problems

```bash
python data/generate_bugs.py
```

## Output Files

- `trajectories.jsonl` — All episode trajectories (for fine-tuning)
- `training_log.csv`   — Per-episode stats (reward, steps, solve rate)

## Example Output

```
Ep   1 | ✅ SOLVED  | Reward:   89.0 | Steps:  2 | Tests: 3/3 | [wrong_operator]
Ep   2 | ❌ FAILED  | Reward:  -31.0 | Steps: 10 | Tests: 0/3 | [off_by_one]
Ep   3 | ✅ SOLVED  | Reward:   98.0 | Steps:  2 | Tests: 3/3 | [wrong_return_value]
```

## Dataset

20 hand-crafted buggy problems across 7 bug types:
- `wrong_operator` — e.g., `-` instead of `+`
- `off_by_one` — e.g., `range(n-1)` instead of `range(n)`
- `wrong_return_value` — returns wrong variable
- `missing_condition` — wrong comparison operator
- `wrong_loop_bound` — incorrect loop range
- `string_method_error` — wrong string method
- `index_error` — wrong list index
