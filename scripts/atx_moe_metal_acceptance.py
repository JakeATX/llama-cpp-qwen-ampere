#!/usr/bin/env python3
"""ATX Metal MoE residency acceptance harness for Apple Silicon builds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    p.add_argument("--build-dir", default="build-atx-metal")
    p.add_argument("--llama-cli")
    p.add_argument("--model", required=True)
    p.add_argument("--policy")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ctx-size", type=int, default=2048)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--target-decode-tps", type=float, default=80.0)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--prompt", default="Write a compact Python function that merges two sorted lists.")
    return p.parse_args()


def median_tps(bench_json: Path, key: str = "generation_tps") -> float | None:
    data = json.loads(bench_json.read_text(encoding="utf-8"))
    vals = [r.get(key) for r in data.get("runs", []) if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def stats_field(stats_path: Path, field: str, default=0):
    if not stats_path.exists():
        return default
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return data.get(field, default)


def run_cmd(cmd: list[str], cwd: Path, log: Path) -> int:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8", errors="replace")
    return proc.returncode


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        rc = run_cmd(["cmake", "--build", args.build_dir, "-j", "8"], repo, out / "build.log")
        if rc != 0:
            (out / "blockers_build.md").write_text("# Metal build failed\n\nSee build.log.\n", encoding="utf-8")
            return rc

    llama_cli = args.llama_cli or str(repo / args.build_dir / "bin" / "llama-cli")
    bench = repo / "scripts" / "atx_moe_bench.py"
    rows = []
    blockers = []

    policy_path = args.policy
    if not policy_path:
        default_policy = out / "default_experts_0_31.json"
        default_policy.write_text(json.dumps({"keep_experts": list(range(32))}, indent=2), encoding="utf-8")
        policy_path = str(default_policy)

    scenarios = [
        ("off", None, "off"),
        ("exact-v1", policy_path, "exact-v1"),
        ("direct", policy_path, "direct"),
        ("hybrid", policy_path, "hybrid"),
    ]

    for label, policy, mode in scenarios:
        mode_out = out / mode
        cmd = [
            "python3", str(bench),
            "--llama-cli", llama_cli,
            "--model", args.model,
            "--mode", mode,
            "--ctx-size", str(args.ctx_size),
            "--tokens", str(args.tokens),
            "--repetitions", str(args.repetitions),
            "--prompt", args.prompt,
            "--out", str(mode_out),
        ]
        if policy:
            cmd += ["--policy", policy]
        if mode in {"direct", "hybrid"}:
            cmd += ["--extra", "--moe-keep-experts", "0-31"]
        rc = run_cmd(cmd, repo, out / f"{label}.runner.log")
        bj = mode_out / f"bench_{mode}.json"
        tps = median_tps(bj) if bj.exists() else None
        stats_path = mode_out / f"stats_{mode}_0.json"
        row = {
            "scenario": label,
            "mode": mode,
            "returncode": rc,
            "median_generation_tps": tps,
            "resident_staging_copy_calls": stats_field(stats_path, "resident_staging_copy_calls"),
            "hot_host_staging_bytes": stats_field(stats_path, "hot_host_staging_bytes"),
            "direct_kernel_dispatch_mmvq": stats_field(stats_path, "direct_kernel_dispatch_mmvq"),
            "host_bytes_copied": stats_field(stats_path, "host_bytes_copied"),
            "metal_prompt_staging_bytes": stats_field(stats_path, "metal_prompt_staging_bytes"),
        }
        rows.append(row)
        if rc != 0:
            blockers.append(f"- {label}: runner failed (see {label}.runner.log)")
        if mode == "direct" and row["direct_kernel_dispatch_mmvq"] == 0:
            blockers.append(f"- {label}: expected Metal direct MMVQ dispatches > 0")
        if mode == "direct" and row["resident_staging_copy_calls"] > 0:
            blockers.append(f"- {label}: expected resident_staging_copy_calls == 0 in strict direct decode")

    off_tps = next((r["median_generation_tps"] for r in rows if r["scenario"] == "off"), None)
    exact_tps = next((r["median_generation_tps"] for r in rows if r["scenario"] == "exact-v1"), None)
    for r in rows:
        r["speedup_vs_off"] = (r["median_generation_tps"] / off_tps) if off_tps and r["median_generation_tps"] else None
        r["speedup_vs_exact_v1"] = (r["median_generation_tps"] / exact_tps) if exact_tps and r["median_generation_tps"] else None
        if r["scenario"] == "off" and off_tps and off_tps < args.target_decode_tps * 0.97:
            blockers.append(f"- baseline regression: {off_tps:.2f} tok/s below 97% of {args.target_decode_tps} tok/s target")
        if r["scenario"] in {"direct", "hybrid"} and r["speedup_vs_exact_v1"] and r["speedup_vs_exact_v1"] < 1.25:
            blockers.append(f"- {r['scenario']}: {r['median_generation_tps']:.2f} tok/s < 1.25x exact-v1")

    summary = {"rows": rows, "blockers": blockers, "target_decode_tps": args.target_decode_tps}
    (out / "acceptance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with (out / "acceptance_table.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
