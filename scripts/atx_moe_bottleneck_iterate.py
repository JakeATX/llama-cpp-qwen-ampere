#!/usr/bin/env python3
"""Throughput-first bottleneck layer search for ATX MoE policies (server MTP 64K)."""

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
    p.add_argument("--model", required=True)
    p.add_argument("--policy-source", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-keep-layers", default="25-28,31-39,40")
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--ctx-size", type=int, default=64000)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--target-decode-tps", type=float, default=80.0)
    p.add_argument("--swap-candidates", type=int, default=8)
    return p.parse_args()


def compile_policies(repo: Path, source: Path, out_dir: Path, stats_json: Path | None, base_layers: str, swap: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", str(repo / "scripts" / "atx_moe_policy_compile.py"),
        "--policy-source", str(source),
        "--out-dir", str(out_dir),
        "--mode", "auto",
        "--base-keep-layers", base_layers,
        "--swap-candidates", str(swap),
    ]
    if stats_json and stats_json.exists():
        cmd += ["--stats-json", str(stats_json)]
    subprocess.run(cmd, cwd=repo, check=True)
    swaps = sorted(out_dir.glob("bottleneck_swap_in_*_drop_*.atx.json"))
    hybrids = sorted(out_dir.glob("combined_top_*_layer_experts.atx.hybrid_top_*_layers.atx.json"))
    return swaps + hybrids


def run_server_scenario(repo: Path, model: str, policy: Path, mode: str, out: Path, ctx: int, tokens: int, build_dir: str) -> dict:
    cmd = [
        "python3", str(repo / "scripts" / "atx_moe_metal_server_acceptance.py"),
        "--repo", str(repo),
        "--build-dir", build_dir,
        "--model", model,
        "--out-dir", str(out),
        "--ctx-size", str(ctx),
        "--max-tokens", str(tokens),
        "--scenarios", "hybrid_probe",
        "--skip-compile",
    ]
    # harness uses fixed scenario names; run a one-off via bench fallback for swap policies
    bench = repo / "scripts" / "atx_moe_bench.py"
    completion = repo / build_dir / "bin" / "llama-completion"
    bcmd = [
        "python3", str(bench),
        "--llama-completion", str(completion),
        "--model", model,
        "--mode", mode,
        "--policy", str(policy),
        "--ctx-size", str(ctx),
        "--tokens", str(tokens),
        "--repetitions", "1",
        "--out", str(out),
        "--extra", "--moe-prewarm-experts", "eager",
    ]
    proc = subprocess.run(bcmd, cwd=repo, text=True, capture_output=True)
    (out / "runner.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    bj = out / f"bench_{mode}.json"
    if not bj.exists():
        return {"returncode": proc.returncode, "decode_tps": None}
    data = json.loads(bj.read_text(encoding="utf-8"))
    run = data.get("runs", [{}])[0]
    stats = run.get("stats", {})
    counters = stats.get("counters", stats) if isinstance(stats, dict) else {}
    return {
        "returncode": proc.returncode,
        "decode_tps": run.get("generation_tps"),
        "draft_acceptance_rate": None,
        "host_bytes_copied": counters.get("host_bytes_copied"),
        "host_expert_single_copy_calls": counters.get("host_expert_single_copy_calls"),
        "direct_kernel_dispatch_mmvq": counters.get("direct_kernel_dispatch_mmvq"),
        "direct_kernel_dispatch_mmq": counters.get("direct_kernel_dispatch_mmq"),
        "stats_path": run.get("stats_path"),
    }


def score_candidate(row: dict, baseline: dict) -> tuple:
    decode = row.get("decode_tps") or 0.0
    mtp = row.get("draft_acceptance_rate") or baseline.get("draft_acceptance_rate") or 0.0
    base_single = baseline.get("host_expert_single_copy_calls") or 0
    single = row.get("host_expert_single_copy_calls") or 0
    penalty = max(0, single - base_single)
    host_bytes = row.get("host_bytes_copied") or 0
    return (decode, mtp, -penalty, -host_bytes)


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    policy_source = Path(args.policy_source)

    baseline_dir = out / "baseline"
    baseline = run_server_scenario(
        repo, args.model, policy_source / "combined_top_10pct_layer_experts.atx.json",
        "hybrid", baseline_dir, args.ctx_size, args.max_tokens, args.build_dir,
    )
    baseline_mtp = baseline.get("draft_acceptance_rate") or 0.76

    results = []
    best = None
    stats_probe = None
    if baseline.get("stats_path"):
        stats_probe = Path(baseline["stats_path"])

    for i in range(args.max_iterations):
        iter_dir = out / f"iter_{i:02d}"
        policies = compile_policies(
            repo, policy_source, iter_dir / "policies",
            stats_probe, args.base_keep_layers, args.swap_candidates,
        )
        if i >= len(policies):
            break
        policy = policies[i]
        row = run_server_scenario(
            repo, args.model, policy, "auto", iter_dir / "probe",
            args.ctx_size, args.max_tokens, args.build_dir,
        )
        row["iteration"] = i
        row["policy"] = str(policy)
        row["score"] = score_candidate(row, baseline)
        if row.get("draft_acceptance_rate") is not None and row["draft_acceptance_rate"] < baseline_mtp - 0.05:
            row["rejected"] = "mtp_regression"
        else:
            row["rejected"] = None
            if row.get("decode_tps") and (best is None or score_candidate(row, baseline) > score_candidate(best, baseline)):
                best = row
        results.append(row)
        if best and best.get("decode_tps") and best["decode_tps"] >= args.target_decode_tps:
            break
        time.sleep(1)

    summary = {
        "target_decode_tps": args.target_decode_tps,
        "baseline": baseline,
        "iterations": results,
        "best": best,
    }
    (out / "iterate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if best and best.get("decode_tps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
