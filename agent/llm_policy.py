"""
llm_policy.py
Uses a HuggingFace model as the RL policy.

Fixes applied:
  - Removed pipeline() entirely — use model.generate() directly to avoid
    max_length/max_new_tokens conflicts from pipeline internals
  - Added truncation of long prompts to prevent OOM on CPU
  - Cleaner generation kwargs with no conflicting flags
  - Better JSON extraction from noisy model output
"""

import json
import os
import re
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Login to HuggingFace if token is set (required for gated models like Llama)
_hf_token = os.getenv("HUGGINGFACE_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
        print("  [LLMPolicy] ✅ Logged in to HuggingFace Hub")
    except Exception as e:
        print(f"  [LLMPolicy] ⚠️  HuggingFace login failed: {e}")


SYSTEM_PROMPT = """You are a Python debugger. Fix buggy code by choosing ONE action.

ACTIONS (respond with exactly one valid JSON object, nothing else):
{"type": "EDIT_LINE", "line": <int>, "new_code": "<string>"}
{"type": "ADD_PRINT", "line": <int>, "var": "<string>"}
{"type": "DELETE_LINE", "line": <int>}
{"type": "RUN_TESTS"}
{"type": "DONE"}

Rules:
- Output ONLY the JSON. No explanation. No markdown.
- Line numbers are 0-indexed.
- Fix wrong operators (- vs +), off-by-one (range(n-1) vs range(n)), wrong return variables.
- Output DONE if all tests pass."""


class LLMPolicy:
    """
    HuggingFace-based policy for the CodeDebugEnv.
    Uses model.generate() directly (no pipeline) to avoid config conflicts.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        max_new_tokens: int = 128,
        max_retries: int = 2,
        device: str = None,
        max_prompt_tokens: int = 512,   # Truncate long prompts to save memory
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.max_prompt_tokens = max_prompt_tokens

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"  [LLMPolicy] Loading model : {model_name}")
        print(f"  [LLMPolicy] Device        : {self.device}")
        if self.device == "cpu":
            print("  [LLMPolicy] ⚠️  CPU mode — each inference step will take 10-60s.")

        # ── Tokenizer ────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,   # suppresses BPE warning
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model ────────────────────────────────────────────────
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,                                          # use dtype not torch_dtype
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        print(f"  [LLMPolicy] ✅ Model loaded.\n")

    def select_action(self, state: Dict) -> Dict:
        """Select the next debugging action given current env state."""
        prompt = self._build_prompt(state)

        for attempt in range(self.max_retries + 1):
            try:
                raw_text = self._generate(prompt)
                action = self._parse_action(raw_text)
                return action
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < self.max_retries:
                    print(f"  [LLMPolicy] Parse error attempt {attempt+1}: {e} — retrying")
                else:
                    print(f"  [LLMPolicy] Could not parse after {self.max_retries} retries → DONE")
                    return {"type": "DONE"}
            except Exception as e:
                print(f"  [LLMPolicy] Generation error: {e} → DONE")
                return {"type": "DONE"}

    def _generate(self, prompt: str) -> str:
        """Tokenize prompt and run model.generate() directly."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = f"{SYSTEM_PROMPT}\n\nUser:\n{prompt}\n\nAssistant:"

        # Tokenize with truncation to avoid OOM
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,            # greedy — deterministic
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        new_tokens = output_ids[0][input_len:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return generated

    def _build_prompt(self, state: Dict) -> str:
        """Build a compact prompt to keep token count low on CPU."""
        code_with_lines = "\n".join(
            f"{i}: {line}" for i, line in enumerate(state["lines"])
        )
        tests_str = "\n".join(f"  {t['assertion']}" for t in state["tests"])
        errors_str = state["error_messages"][0] if state["error_messages"] else "None"
        last_action = (
            json.dumps(state["action_history"][-1])
            if state["action_history"] else "None"
        )

        return (
            f"Steps left: {10 - state['step_count']} | "
            f"Passing: {state['tests_passed']}/{state['total_tests']}\n\n"
            f"CODE:\n{code_with_lines}\n\n"
            f"TESTS:\n{tests_str}\n\n"
            f"LAST ERROR: {errors_str}\n"
            f"LAST ACTION: {last_action}\n\n"
            f"Your JSON action:"
        )

    def _parse_action(self, raw_text: str) -> Dict:
        """Extract and validate a JSON action from model output."""
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        # Try to find a JSON object anywhere in the output
        json_match = re.search(r'\{[^{}]+\}', clean, re.DOTALL)
        if json_match:
            clean = json_match.group(0)

        action = json.loads(clean)

        valid_types = {"EDIT_LINE", "ADD_PRINT", "DELETE_LINE", "RUN_TESTS", "DONE"}
        if action.get("type") not in valid_types:
            raise ValueError(f"Unknown action type: {action.get('type')}")

        if "line" in action:
            action["line"] = int(action["line"])

        return action


if __name__ == "__main__":
    policy = LLMPolicy(model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    test_state = {
        "code": "def add(a, b):\n    return a - b",
        "lines": ["def add(a, b):", "    return a - b"],
        "tests": [{"assertion": "assert add(2, 3) == 5", "description": "basic"}],
        "error_messages": ["AssertionError"],
        "action_history": [],
        "step_count": 0,
        "tests_passed": 0,
        "total_tests": 1,
    }
    action = policy.select_action(test_state)
    print(f"Action: {json.dumps(action, indent=2)}")
