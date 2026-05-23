#!/usr/bin/env python3
"""Focused ATX MoE throughput runner for llama.cpp local builds."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


GEN_RE = re.compile(r"Generation:\s*([0-9.]+)\s*t/s")
PROMPT_RE = re.compile(r"Prompt:\s*([0-9.]+)\s*t/s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llama-cli", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--policy")
    p.add_argument("--mode", default="off", choices=["off", "exact-v1", "direct", "hybrid", "auto", "layer"])
    p.add_argument("--ctx-size", type=int, default=512)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--prompt", default="Write a compact Python function that merges two sorted lists.")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--cold-copy-mode", default="auto")
    p.add_argument("--cold-coalesce-gap", type=int, default=0)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def run_once(args: argparse.Namespace, rep: int, out_dir: Path) -> dict:
    stats = out_dir / f"stats_{args.mode}_{rep}.json"
    log = out_dir / f"log_{args.mode}_{rep}.txt"
    cmd = [
        args.llama_cli,
        "-m", args.model,
        "-ngl", "all",
        "--ctx-size", str(args.ctx_size),
        "--seed", "1",
        "--temp", "0",
        "--no-warmup",
        "--no-display-prompt",
        "-p", args.prompt,
        "-n", str(args.tokens),
        "--moe-residency-stats", str(stats),
    ]
    if args.policy:
        cmd += ["--moe-residency-policy", args.policy]
    if args.mode != "off":
        cmd += ["--moe-residency-mode", args.mode]
    if args.mode in {"direct", "hybrid", "auto"}:
        cmd += ["--moe-direct-require", "--moe-direct-strict-hot-no-stage"]
    if args.mode != "off":
        cmd += ["--moe-cold-copy-mode", args.cold_copy_mode, "--moe-cold-coalesce-gap", str(args.cold_coalesce_gap)]
    cmd += args.extra

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.perf_counter() - t0
    text = proc.stdout + "\n" + proc.stderr
    log.write_text(text, encoding="utf-8", errors="replace")

    prompt_match = PROMPT_RE.search(text)
    gen_match = GEN_RE.search(text)
    stats_obj = {}
    if stats.exists():
        try:
            stats_obj = json.loads(stats.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stats_obj = {"parse_error": str(stats)}

    return {
        "rep": rep,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "prompt_tps": float(prompt_match.group(1)) if prompt_match else None,
        "generation_tps": float(gen_match.group(1)) if gen_match else None,
        "stats_path": str(stats),
        "log_path": str(log),
        "stats": stats_obj,
    }


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [run_once(args, i, out) for i in range(args.repetitions)]
    result = {
        "model": args.model,
        "policy": args.policy,
        "mode": args.mode,
        "ctx_size": args.ctx_size,
        "tokens": args.tokens,
        "runs": runs,
    }
    (out / f"bench_{args.mode}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    ok = all(r["returncode"] == 0 for r in runs)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
