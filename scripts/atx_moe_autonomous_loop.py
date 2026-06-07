#!/usr/bin/env python3
"""Closed-loop ATX Metal MoE iteration orchestrator."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


GATES = {
    "decode_tps_min": 72.0,
    "decode_tps_stretch": 80.0,
    "no_policy_regression_pct": 0.03,
    "mtp_acceptance_drop_max": 0.05,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    p.add_argument("--build-dir", default="build-atx-metal")
    p.add_argument("--model", required=True)
    p.add_argument("--policy-source", default="")
    p.add_argument("--runs-root", default="")
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--ctx-size", type=int, default=64000)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--best-layers", default="", help="Inline whole-layer best candidate to include in acceptance, e.g. 0-40")
    p.add_argument("--best-policy", default="", help="Explicit best candidate policy path to include in acceptance")
    p.add_argument("--best-mode", default="layer", choices=["exact-v1", "direct", "hybrid", "auto", "layer"])
    p.add_argument("--prewarm-experts", default="", choices=["", "off", "lazy", "eager"])
    p.add_argument("--scenarios", nargs="+", default=[
        "reference", "known_fast_tail", "hybrid_top8", "hybrid_top16",
        "exact_v1_10pct", "direct_10pct", "bottleneck_auto",
    ])
    return p.parse_args()


def sync_and_build(repo: Path, build_dir: str, skip_build: bool) -> None:
    subprocess.run(["git", "fetch", "jakeatx", "atx-expert-residency"], cwd=repo, check=False)
    subprocess.run(["git", "merge", "--ff-only", "jakeatx/atx-expert-residency"], cwd=repo, check=False)
    if skip_build:
        return
    ncpu = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
    subprocess.run(["cmake", "--build", build_dir, "-j", ncpu], cwd=repo, check=True)


def next_iter_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("iter_*"))
    n = 0
    if existing:
        last = existing[-1].name
        try:
            n = int(last.split("_", 1)[1]) + 1
        except ValueError:
            n = len(existing)
    out = root / f"iter_{n:03d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def counter(row: dict, key: str) -> int | None:
    val = row.get(key)
    if val is None and isinstance(row.get("stats"), dict):
        c = row["stats"].get("counters", row["stats"])
        val = c.get(key)
    return int(val) if val is not None else None


def score_row(row: dict, baseline: dict | None) -> tuple:
    decode = row.get("decode_tps") or 0.0
    mtp = row.get("draft_acceptance_rate") or 0.0
    host_single = counter(row, "single_ranges") or counter(row, "host_expert_single_copy_calls") or 0
    host_bytes = counter(row, "host_bytes") or counter(row, "host_bytes_copied") or 0
    base_single = 0
    if baseline:
        base_single = counter(baseline, "single_ranges") or counter(baseline, "host_expert_single_copy_calls") or 0
    penalty = max(0, host_single - base_single)
    return (decode, mtp, -penalty, -host_bytes)


def triage(summary: dict, gates: dict) -> list[str]:
    blockers: list[str] = []
    rows = summary.get("rows", [])
    ref = next((r for r in rows if r.get("scenario") == "reference"), None)
    hybrid = next((r for r in rows if r.get("scenario") == "hybrid_top16"), None) or next(
        (r for r in rows if "hybrid" in r.get("scenario", "")), None
    )
    if not ref or not ref.get("decode_tps"):
        blockers.append("harness: reference scenario missing decode tok/s")
    for r in rows:
        if not Path(r.get("stats_path", "")).exists():
            blockers.append(f"harness: {r.get('scenario')} missing residency stats JSON")
    if hybrid:
        mmvq = counter(hybrid, "direct_mmvq") or counter(hybrid, "direct_kernel_dispatch_mmvq") or 0
        if mmvq <= 0:
            blockers.append("runtime: hybrid/direct direct_kernel_dispatch_mmvq is zero")
        staging = counter(hybrid, "resident_staging") or counter(hybrid, "resident_staging_copy_calls") or 0
        if staging > 0:
            blockers.append("runtime: resident_staging_copy_calls > 0 under strict direct/hybrid")
    best = summary.get("best_decode_tps") or 0.0
    if best < gates["decode_tps_min"]:
        blockers.append(f"policy: best decode {best:.2f} tok/s below minimum {gates['decode_tps_min']}")
    if ref and ref.get("decode_tps") and best:
        floor = ref["decode_tps"] * (1.0 - gates["no_policy_regression_pct"])
        if best < floor and hybrid and hybrid.get("scenario") != "reference":
            blockers.append(f"policy: best policy regressed more than {gates['no_policy_regression_pct']*100:.0f}% vs no-policy")
    ref_mtp = ref.get("draft_acceptance_rate") if ref else None
    if ref_mtp and hybrid and hybrid.get("draft_acceptance_rate") is not None:
        if hybrid["draft_acceptance_rate"] < ref_mtp - gates["mtp_acceptance_drop_max"]:
            blockers.append("mtp: hybrid acceptance dropped vs reference")
    return blockers


def gates_pass(summary: dict, gates: dict) -> bool:
    return len(triage(summary, gates)) == 0 and bool(summary.get("best_decode_tps", 0) >= gates["decode_tps_min"])


def write_checkpoint(out: Path, iteration: int, summary: dict, blockers: list[str]) -> None:
    checkpoint = {
        "iteration": iteration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "blockers": blockers,
        "minimum_gate_pass": bool(summary.get("best_decode_tps", 0) >= GATES["decode_tps_min"]),
        "stretch_gate_pass": bool(summary.get("best_decode_tps", 0) >= GATES["decode_tps_stretch"]),
        "work_item": (
            "harness" if any("harness" in b for b in blockers) else
            "runtime" if any("runtime" in b for b in blockers) else
            "policy" if any("policy" in b or "mtp" in b for b in blockers) else
            "none"
        ),
    }
    (out / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    lines = ["# ATX Metal autonomous blockers", ""]
    if blockers:
        lines.extend(f"- {b}" for b in blockers)
    else:
        lines.append("- none")
    (out / "blockers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diff = subprocess.run(["git", "diff"], capture_output=True, text=True)
    if diff.stdout.strip():
        (out / "git_diff.patch").write_text(diff.stdout, encoding="utf-8")


def run_acceptance(repo: Path, args: argparse.Namespace, out: Path) -> dict:
    cmd = [
        "python3", str(repo / "scripts" / "atx_moe_metal_server_acceptance.py"),
        "--repo", str(repo),
        "--build-dir", args.build_dir,
        "--model", args.model,
        "--out-dir", str(out),
        "--ctx-size", str(args.ctx_size),
        "--max-tokens", str(args.max_tokens),
        "--scenarios", *args.scenarios,
    ]
    if args.policy_source:
        cmd += ["--policy-source", args.policy_source]
    if args.best_layers:
        cmd += ["--layer-scenario", f"best_candidate={args.best_layers}"]
    if args.best_policy:
        cmd += ["--policy-scenario", f"best_candidate={args.best_mode}={args.best_policy}"]
    if args.prewarm_experts:
        cmd += ["--prewarm-experts", args.prewarm_experts]
    subprocess.run(cmd, cwd=repo, check=False)
    summary_path = out / "acceptance_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"acceptance failed to write {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    runs_root = Path(args.runs_root) if args.runs_root else repo.parent / "runs" / "atx_moe_metal" / "autonomous"
    policy_source = args.policy_source or str(repo.parent / "runs" / "hf_combined_saliency_upload" / "staging" / "policies")

    for i in range(args.max_iterations):
        sync_and_build(repo, args.build_dir, args.skip_build)
        out = next_iter_dir(runs_root)
        summary = run_acceptance(repo, argparse.Namespace(**{**vars(args), "policy_source": policy_source}), out)
        blockers = triage(summary, GATES)
        write_checkpoint(out, i, summary, blockers)
        if gates_pass(summary, GATES):
            print(f"All gates passed at {out}")
            return 0
        print(f"iter {i} blockers ({len(blockers)}): see {out / 'blockers.md'}")
        if i + 1 >= args.max_iterations:
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
