#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


USER_PROMPTS = [
    "Summarize the deployment note for an engineer who needs the operational risk, not marketing language.",
    "Rewrite the following test result as a concise incident update with cause, impact, and next action.",
    "Explain why a baseline sanity check should run before comparing two cache implementations.",
    "Turn this benchmark observation into a decision record for a small inference team.",
    "List the checks you would run before claiming a quantized cache is production ready.",
    "Condense this debugging log into the three facts that change the next experiment.",
    "Draft a short response to a reviewer who is worried that a dataset is using the wrong chat template.",
    "Compare two possible explanations for an accuracy regression and say which evidence would decide it.",
]

MODEL_RESPONSES = [
    "The deployment risk is that the reference measurement is not yet trustworthy. The next action is to validate the f16 baseline on a fixture that matches the model template before using any KVarN delta as correctness evidence.",
    "The run showed a high perplexity baseline, so the comparison is blocked. Impact is limited to evaluation confidence, not necessarily runtime correctness. The next action is to replace the fixture and rerun the baseline alone.",
    "A baseline check prevents a broken corpus or model pairing from being mistaken for a KVarN failure. If f16 already scores badly, the KVarN ratio is measured against noise rather than a meaningful reference.",
    "Decision: do not accept throughput numbers until the same binary passes an f16-versus-KVarN accuracy gate. Rationale: speed is valuable only after the implementation is numerically close to the normal cache.",
    "The minimum checks are a sane f16 perplexity, matching tokenizer/template markers, no literal unknown-token artifacts, a bounded KVarN perplexity increase, cache engagement logs, and sequential runs that do not inflate memory pressure.",
    "The important facts are that the invalid fixture used another model's markers, the clean prose was still off-distribution for the instruction model, and a valid Gemma turn format scored normally through the same executable.",
    "The concern is valid. A chat-tuned Gemma model should be tested on text rendered with its own turn markers. A Qwen-style transcript can make the baseline look broken and should fail preflight before KVarN is invoked.",
    "If flash attention on fixes the issue and off fails, the cache layout is suspect. If both fail on invalid text but both pass on valid Gemma turns, the causal mechanism is the evaluation fixture rather than attention math.",
]


def build_fixture(turns: int) -> str:
    parts: list[str] = []
    for i in range(turns):
        user = USER_PROMPTS[i % len(USER_PROMPTS)]
        response = MODEL_RESPONSES[i % len(MODEL_RESPONSES)]
        parts.append(f"<|turn>user\n{user}\n<turn|>\n")
        parts.append(f"<|turn>model\n{response}\n<turn|>\n")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic Gemma4-formatted chat perplexity fixture.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--turns", type=int, default=64)
    args = ap.parse_args()

    if args.turns <= 0:
        raise SystemExit("--turns must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_fixture(args.turns), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
