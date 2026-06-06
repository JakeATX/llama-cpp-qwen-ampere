#!/usr/bin/env python3
"""Throughput-first bottleneck layer search for ATX MoE policies (macOS/Linux)."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    p.add_argument("--build-dir", default="build-atx-metal")
    p.add_argument("--llama-cli")
    p.add_argument("--model", required=True)
    p.add_argument("--policy-source", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-keep-layers", default="25,26,27,28,31,32,33,34,35,37,38,39")
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--ctx-size", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--target-decode-tps", type=float, default=80.0)
    return p.parse_args()


def run_bench(repo: Path, llama_cli: str, model: str, policy: Path, mode: str, out: Path, ctx: int, tokens: int) -> dict:
    bench = repo / "scripts" / "atx_moe_bench.py"
    cmd = [
        "python3", str(bench),
        "--llama-cli", llama_cli,
        "--model", model,
        "--mode", mode,
        "--policy", str(policy),
        "--ctx-size", str(ctx),
        "--tokens", str(tokens),
        "--repetitions", "1",
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
    (out / "runner.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8", errors="replace")
    bj = out / f"bench_{mode}.json"
    if not bj.exists():
        return {"returncode": proc.returncode, "generation_tps": None}
    data = json.loads(bj.read_text(encoding="utf-8"))
    vals = [r.get("generation_tps") for r in data.get("runs", []) if r.get("generation_tps") is not None]
    return {
        "returncode": proc.returncode,
        "generation_tps": statistics.median(vals) if vals else None,
        "stats": data.get("runs", [{}])[0].get("stats", {}),
    }


def compile_policy(repo: Path, source: Path, out_dir: Path, stats_json: Path | None, base_layers: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", str(repo / "scripts" / "atx_moe_policy_compile.py"),
        "--policy-source", str(source),
        "--out-dir", str(out_dir),
        "--mode", "auto",
        "--base-keep-layers", base_layers,
    ]
    if stats_json and stats_json.exists():
        cmd += ["--stats-json", str(stats_json)]
    subprocess.run(cmd, cwd=repo, check=True)
    candidates = sorted(out_dir.glob("bottleneck_top_*_layers.atx.json"))
    if not candidates:
        raise SystemExit(f"no bottleneck policies in {out_dir}")
    return candidates[0]


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    llama_cli = args.llama_cli or str(repo / args.build_dir / "bin" / "llama-cli")
    policy_source = Path(args.policy_source)

    baseline_policy = out / "baseline_layer.json"
    baseline_policy.write_text(
        json.dumps({"keep_layers": [int(x) for x in args.base_keep_layers.split(",") if x.strip()]}, indent=2),
        encoding="utf-8",
    )

    results = []
    best = None
    for i in range(args.max_iterations):
        iter_dir = out / f"iter_{i:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        stats_probe = iter_dir / "stats_probe.json"
        if i == 0:
            probe = run_bench(repo, llama_cli, args.model, baseline_policy, "layer", iter_dir / "baseline", args.ctx_size, args.tokens)
            if probe.get("stats"):
                stats_probe.write_text(json.dumps(probe["stats"], indent=2), encoding="utf-8")
        policy = compile_policy(
            repo,
            policy_source,
            iter_dir / "policies",
            stats_probe if stats_probe.exists() else None,
            args.base_keep_layers,
        )
        hybrid = run_bench(repo, llama_cli, args.model, policy, "hybrid", iter_dir / "hybrid", args.ctx_size, args.tokens)
        row = {
            "iteration": i,
            "policy": str(policy),
            "generation_tps": hybrid.get("generation_tps"),
            "host_bytes_copied": hybrid.get("stats", {}).get("host_bytes_copied"),
            "direct_kernel_dispatch_mmvq": hybrid.get("stats", {}).get("direct_kernel_dispatch_mmvq"),
        }
        results.append(row)
        if hybrid.get("generation_tps") and (best is None or hybrid["generation_tps"] > best["generation_tps"]):
            best = row
        if best and best["generation_tps"] and best["generation_tps"] >= args.target_decode_tps:
            break
        time.sleep(1)

    summary = {
        "target_decode_tps": args.target_decode_tps,
        "iterations": results,
        "best": best,
    }
    (out / "iterate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if best and best.get("generation_tps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
