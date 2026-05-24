#!/usr/bin/env python3
"""
Adaptive layer bottleneck campaign tooling.

Subcommands:
  inventory  - write local environment inventory
  make-mini  - create a mixed coding/agent/control prompt manifest
  run        - execute a manifest with llama-cli/llama-completion and --layer-profile
  analyze    - aggregate layer_profile JSONL into layer rankings, policies, and HTML

This script is intentionally stdlib-only so it can run on the Mac and on a
future CUDA host without dependency setup.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LAYERS = 40
DEFAULT_FULL_ATTENTION_LAYERS = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def inventory(args: argparse.Namespace) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    git_rc, git_head, _ = run_cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    git_status_rc, git_status, _ = run_cmd(["git", "-C", str(ROOT), "status", "--short"])
    disk = shutil.disk_usage(ROOT)
    meta = {
        "type": "layer_bottleneck_inventory",
        "created_utc": utc_now(),
        "root": str(ROOT),
        "llama_cpp": str(ROOT),
        "git_head": git_head.strip() if git_rc == 0 else "",
        "git_status_short": git_status.splitlines() if git_status_rc == 0 else [],
        "model": str(args.model) if args.model else "",
        "model_exists": bool(args.model and Path(args.model).exists()),
        "llama_cli": str(args.llama_cli) if args.llama_cli else "",
        "llama_cli_exists": bool(args.llama_cli and Path(args.llama_cli).exists()),
        "disk_total_gib": round(disk.total / 2**30, 3),
        "disk_free_gib": round(disk.free / 2**30, 3),
        "python": sys.version,
        "existing_trace_artifacts": {
            "route_set_run": str(ROOT / "runs/moe_route_set_critical_path/latest_full"),
            "route_set_run_exists": (ROOT / "runs/moe_route_set_critical_path/latest_full/run_meta.json").exists(),
        },
    }
    write_json(out / "inventory.json", meta)
    print(json.dumps(meta, indent=2))


def make_mini(args: argparse.Namespace) -> None:
    prompts: list[dict[str, Any]] = []

    base_prompts = [
        ("HumanEval", "humaneval_fib", "Write a Python function `fib(n)` that returns the nth Fibonacci number. Include edge cases and a short explanation."),
        ("HumanEval", "humaneval_merge", "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Return intervals sorted by start."),
        ("MBPP", "mbpp_json", "Write a Python function that reads JSON lines from a string and returns the sum of the `value` fields for valid rows."),
        ("LiveCodeBench", "lcb_bugfix", "You are given a Python function that sometimes returns the wrong answer for duplicate values. Explain the bug and provide a corrected implementation."),
        ("LiveCodeBench", "lcb_datastructure", "Solve this coding problem: maintain a stream of integers and support insert, delete-one, and query median in O(log n) time."),
        ("SWE-style", "swe_patch_test", "A test suite fails because a cache key ignores the `locale` argument. Describe the minimal patch and write the changed function."),
        ("SWE-style", "swe_refactor", "Refactor a small command parser so quoted strings and escaped spaces are handled correctly. Provide tests."),
        ("Agentic", "agent_terminal_git", "In a terminal, diagnose why a git branch has diverged, preserve local changes, and produce a safe sequence of commands."),
        ("Agentic", "agent_terminal_tests", "You are an agent in a repo. Tests fail only on macOS path handling. Plan the investigation and propose the patch."),
        ("NonCodingControl", "control_summary", "Summarize the tradeoffs of urban rail expansion versus bus rapid transit for a city council memo."),
        ("NonCodingControl", "control_reasoning", "Explain why Simpson's paradox can reverse an aggregate trend using a concrete numerical example."),
    ]
    for benchmark, request_id, prompt in base_prompts:
        prompts.append({
            "request_id": request_id,
            "benchmark": benchmark,
            "task_id": request_id,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "max_tokens": args.max_tokens,
            "temperature": 0,
        })

    tbench_tasks = sorted((ROOT / "sources/terminal-bench/original-tasks").glob("*/task.yaml"))[: args.terminal_tasks]
    for task in tbench_tasks:
        text = task.read_text(encoding="utf-8", errors="replace")
        prompt = f"Terminal-Bench task:\n\n{text[:6000]}\n\nProvide a concise terminal-agent plan and the final commands you would run."
        task_id = task.parent.name
        prompts.append({
            "request_id": f"terminal_{task_id}",
            "benchmark": "Terminal-Bench Task Prompt",
            "task_id": task_id,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "max_tokens": args.max_tokens,
            "temperature": 0,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in prompts[: args.limit if args.limit else None]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.out), "requests": len(prompts[: args.limit if args.limit else None])}, indent=2))


TIMING_RE = re.compile(r"eval time =\s+(?P<eval_ms>[0-9.]+) ms /\s+(?P<decoded>[0-9]+) tokens.*?(?P<tps>[0-9.]+) tokens per second", re.S)
PROMPT_RE = re.compile(r"prompt eval time =\s+(?P<prompt_ms>[0-9.]+) ms /\s+(?P<prompt_tokens>[0-9]+) tokens", re.S)
CLI_BRACKET_RE = re.compile(r"\[\s*Prompt:\s*(?P<prompt_tps>[0-9.]+)\s*t/s\s*\|\s*Generation:\s*(?P<decode_tps>[0-9.]+)\s*t/s\s*\]")


def choose_runner(path: Path) -> Path:
    """Prefer llama-completion for one-shot profiling if it is built beside llama-cli."""
    if path.name == "llama-cli":
        completion = path.with_name("llama-completion")
        if completion.exists():
            return completion
    return path


def run_manifest(args: argparse.Namespace) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "profiles"
    raw_dir.mkdir(exist_ok=True)
    stdout_dir = out / "stdout"
    stdout_dir.mkdir(exist_ok=True)
    status_path = out / "request_status.jsonl"
    manifest_out = out / "request_manifest.jsonl"
    seen = set()
    if status_path.exists() and args.resume:
        for rec in iter_jsonl(status_path):
            if rec.get("status") == "ok":
                seen.add(rec.get("request_id"))

    with args.manifest.open("r", encoding="utf-8") as f, manifest_out.open("w", encoding="utf-8") as mf, status_path.open("a", encoding="utf-8") as sf:
        for line in f:
            row = json.loads(line)
            request_id = row["request_id"]
            mf.write(json.dumps(row, sort_keys=True) + "\n")
            if request_id in seen:
                continue
            profile_path = raw_dir / f"{request_id}.layer_profile.jsonl"
            prompt_path = out / f"{request_id}.prompt.txt"
            prompt_path.write_text(row["prompt"], encoding="utf-8")
            runner = choose_runner(args.llama_cli)
            cmd = [
                str(runner),
                "-m", str(args.model),
                "-f", str(prompt_path),
                "-n", str(row.get("max_tokens", args.max_tokens)),
                "--temp", str(row.get("temperature", 0)),
                "--seed", str(args.seed),
                "--no-display-prompt",
                "--layer-profile", str(profile_path),
                "--layer-profile-detail", args.detail,
                "--layer-profile-sync", args.sync,
            ]
            if runner.name == "llama-cli":
                cmd.extend(["--single-turn", "--simple-io"])
            started = time.time()
            try:
                p = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout_s)
                rc = p.returncode
                stdout = p.stdout
                stderr = p.stderr
                timed_out = False
            except subprocess.TimeoutExpired as e:
                rc = 124
                stdout = e.stdout or ""
                stderr = e.stderr or ""
                timed_out = True
            elapsed_ms = (time.time() - started) * 1000.0
            (stdout_dir / f"{request_id}.stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
            (stdout_dir / f"{request_id}.stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")
            pm = PROMPT_RE.search(stderr + "\n" + stdout)
            em = TIMING_RE.search(stderr + "\n" + stdout)
            bm = CLI_BRACKET_RE.search(stderr + "\n" + stdout)
            status = {
                "type": "layer_bottleneck_request_status",
                "request_id": request_id,
                "benchmark": row.get("benchmark"),
                "task_id": row.get("task_id"),
                "status": "ok" if rc == 0 and profile_path.exists() else "failed",
                "returncode": rc,
                "timed_out": timed_out,
                "elapsed_ms": elapsed_ms,
                "profile_path": str(profile_path),
                "stdout_path": str(stdout_dir / f"{request_id}.stdout.txt"),
                "stderr_path": str(stdout_dir / f"{request_id}.stderr.txt"),
                "prompt_ms": float(pm.group("prompt_ms")) if pm else "",
                "prompt_tokens": int(pm.group("prompt_tokens")) if pm else "",
                "decode_ms": float(em.group("eval_ms")) if em else "",
                "decoded_tokens": int(em.group("decoded")) if em else "",
                "prompt_tps": float(bm.group("prompt_tps")) if bm else "",
                "decode_tps": float(em.group("tps")) if em else (float(bm.group("decode_tps")) if bm else ""),
                "command": cmd,
            }
            sf.write(json.dumps(status, sort_keys=True) + "\n")
            sf.flush()
    print(json.dumps({"out_dir": str(out), "status_path": str(status_path)}, indent=2))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def layer_policy(name: str, layers: list[int], reason: str, layer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_layer = {int(r["layer"]): r for r in layer_rows}
    total_ms = sum(float(r.get("elapsed_ms", 0) or 0) for r in layer_rows)
    selected_ms = sum(float(row_by_layer.get(l, {}).get("elapsed_ms", 0) or 0) for l in layers)
    selected_decode = sum(float(row_by_layer.get(l, {}).get("decode_like_ms", 0) or 0) for l in layers)
    total_decode = sum(float(r.get("decode_like_ms", 0) or 0) for r in layer_rows)
    return {
        "schema_version": "atx-layer-hot-policy-v1",
        "policy_name": name,
        "policy_type": "layer",
        "keep_layers": layers,
        "selected_layers": len(layers),
        "expected_total_layer_time_capture_pct": round(100 * selected_ms / total_ms, 6) if total_ms else 0,
        "expected_decode_like_time_capture_pct": round(100 * selected_decode / total_decode, 6) if total_decode else 0,
        "measured_decode_tps": None,
        "measured_p95_token_latency_ms": None,
        "measurement_status": "candidate_not_ab_tested",
        "reason": reason,
    }


def analyze(args: argparse.Namespace) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    profiles = list(args.profile_dir.glob("*.jsonl")) if args.profile_dir else []
    if args.profile_jsonl:
        profiles.append(args.profile_jsonl)

    status_by_request = {}
    if args.status_jsonl and args.status_jsonl.exists():
        for rec in iter_jsonl(args.status_jsonl):
            status_by_request[Path(str(rec.get("profile_path", ""))).name] = rec

    per_layer: dict[int, dict[str, Any]] = {}
    op_rows_map: dict[tuple[int, str], dict[str, Any]] = {}
    token_rows: list[dict[str, Any]] = []
    raw_out = out / "layer_profile_raw.jsonl"
    total_nodes = 0

    with raw_out.open("w", encoding="utf-8") as raw:
        for profile in profiles:
            request_id = profile.name.replace(".layer_profile.jsonl", "")
            req_status = status_by_request.get(profile.name, {})
            per_request_layer_ms = Counter()
            for rec in iter_jsonl(profile):
                if rec.get("type") != "layer_profile_node":
                    continue
                raw.write(json.dumps(rec, sort_keys=True) + "\n")
                total_nodes += 1
                layer = int(rec.get("layer", -1))
                if layer < 0:
                    continue
                elapsed_ms = float(rec.get("elapsed_us", 0.0)) / 1000.0
                phase = rec.get("phase", "unknown")
                fam = rec.get("op_family", "unknown")
                row = per_layer.setdefault(layer, {
                    "layer": layer,
                    "node_count": 0,
                    "elapsed_ms": 0.0,
                    "decode_like_ms": 0.0,
                    "prefill_like_ms": 0.0,
                    "tensor_bytes": 0,
                    "src_bytes": 0,
                    "requests": set(),
                    "benchmarks": Counter(),
                    "samples": [],
                })
                row["node_count"] += 1
                row["elapsed_ms"] += elapsed_ms
                row["tensor_bytes"] += int(rec.get("tensor_bytes", 0) or 0)
                row["src_bytes"] += int(rec.get("src0_bytes", 0) or 0) + int(rec.get("src1_bytes", 0) or 0)
                row["requests"].add(request_id)
                if req_status.get("benchmark"):
                    row["benchmarks"][req_status.get("benchmark")] += 1
                if phase == "decode_like":
                    row["decode_like_ms"] += elapsed_ms
                elif phase == "prefill_like":
                    row["prefill_like_ms"] += elapsed_ms
                per_request_layer_ms[layer] += elapsed_ms
                op_key = (layer, fam)
                op = op_rows_map.setdefault(op_key, {"layer": layer, "op_family": fam, "node_count": 0, "elapsed_ms": 0.0})
                op["node_count"] += 1
                op["elapsed_ms"] += elapsed_ms
            for layer, ms in per_request_layer_ms.items():
                per_layer[layer]["samples"].append(ms)
            if per_request_layer_ms:
                token_rows.append({
                    "request_id": request_id,
                    "benchmark": req_status.get("benchmark", ""),
                    "total_profiled_layer_ms": round(sum(per_request_layer_ms.values()), 6),
                    "max_layer_ms": round(max(per_request_layer_ms.values()), 6),
                    "max_layer": max(per_request_layer_ms, key=lambda k: per_request_layer_ms[k]),
                    "decode_tps": req_status.get("decode_tps", ""),
                })

    total_ms = sum(float(v["elapsed_ms"]) for v in per_layer.values())
    total_decode = sum(float(v["decode_like_ms"]) for v in per_layer.values())
    layer_rows = []
    for layer in range(LAYERS):
        v = per_layer.get(layer, {
            "layer": layer, "node_count": 0, "elapsed_ms": 0.0, "decode_like_ms": 0.0, "prefill_like_ms": 0.0,
            "tensor_bytes": 0, "src_bytes": 0, "requests": set(), "benchmarks": Counter(), "samples": [],
        })
        samples = list(v["samples"])
        elapsed = float(v["elapsed_ms"])
        decode = float(v["decode_like_ms"])
        traffic_share = float(v["src_bytes"]) / max(1.0, sum(float(x["src_bytes"]) for x in per_layer.values()))
        decode_share = decode / total_decode if total_decode else 0.0
        time_share = elapsed / total_ms if total_ms else 0.0
        p95 = percentile(samples, 95)
        p99 = percentile(samples, 99)
        stability = len(v["requests"]) / max(1, len(profiles))
        score = time_share + decode_share + traffic_share + 0.25 * stability
        layer_rows.append({
            "layer": layer,
            "node_count": v["node_count"],
            "elapsed_ms": round(elapsed, 6),
            "elapsed_share_pct": round(100 * time_share, 6),
            "decode_like_ms": round(decode, 6),
            "decode_like_share_pct": round(100 * decode_share, 6),
            "prefill_like_ms": round(float(v["prefill_like_ms"]), 6),
            "p95_request_layer_ms": round(p95, 6),
            "p99_request_layer_ms": round(p99, 6),
            "tensor_bytes": v["tensor_bytes"],
            "src_bytes": v["src_bytes"],
            "traffic_share_pct": round(100 * traffic_share, 6),
            "active_request_count": len(v["requests"]),
            "benchmark_counts": json.dumps(dict(v["benchmarks"]), sort_keys=True),
            "bottleneck_score": round(score, 9),
            "measurement_basis": "synchronized ggml scheduler callback timing; node timings grouped by blk.N layer",
        })
    layer_rows.sort(key=lambda r: float(r["bottleneck_score"]), reverse=True)

    op_rows = list(op_rows_map.values())
    op_rows.sort(key=lambda r: float(r["elapsed_ms"]), reverse=True)
    for r in op_rows:
        r["elapsed_ms"] = round(float(r["elapsed_ms"]), 6)

    top_total = [int(r["layer"]) for r in sorted(layer_rows, key=lambda r: float(r["elapsed_ms"]), reverse=True)[:10]]
    top_decode = [int(r["layer"]) for r in sorted(layer_rows, key=lambda r: float(r["decode_like_ms"]), reverse=True)[:10]]
    top_tail = [int(r["layer"]) for r in sorted(layer_rows, key=lambda r: float(r["p95_request_layer_ms"]), reverse=True)[:10]]
    top_traffic = [int(r["layer"]) for r in sorted(layer_rows, key=lambda r: float(r["src_bytes"]), reverse=True)[:10]]
    top_hybrid = [int(r["layer"]) for r in layer_rows[:10]]
    rng = random.Random(1337)
    random_policies = [sorted(rng.sample(range(LAYERS), 10)) for _ in range(5)]

    policies = [
        layer_policy("first_10", list(range(10)), "baseline first ten layers", layer_rows),
        layer_policy("last_10", list(range(30, 40)), "baseline last ten layers", layer_rows),
        layer_policy("full_attention_10", DEFAULT_FULL_ATTENTION_LAYERS, "heuristic full-attention layer set", layer_rows),
        layer_policy("top10_total_time", sorted(top_total), "top ten layers by measured total layer time", layer_rows),
        layer_policy("top10_decode_time", sorted(top_decode), "top ten layers by measured decode-like time", layer_rows),
        layer_policy("top10_tail_p95", sorted(top_tail), "top ten layers by p95 request-layer time", layer_rows),
        layer_policy("top10_traffic", sorted(top_traffic), "top ten layers by profiled source/tensor bytes", layer_rows),
        layer_policy("top10_hybrid_bottleneck", sorted(top_hybrid), "top ten layers by weighted bottleneck score", layer_rows),
    ]
    for i, layers in enumerate(random_policies):
        policies.append(layer_policy(f"random_10_seed_{i}", layers, "random same-count control", layer_rows))

    write_csv(out / "layer_timing_summary.csv", layer_rows)
    write_csv(out / "layer_op_family_summary.csv", op_rows)
    write_csv(out / "token_latency.csv", token_rows)
    write_json(out / "layer_policy_candidates.json", policies)
    write_json(out / "recommended_top10_layers.json", policies[7])
    write_csv(out / "layer_policy_comparison.csv", [{
        "policy_name": p["policy_name"],
        "keep_layers": ",".join(map(str, p["keep_layers"])),
        "expected_total_layer_time_capture_pct": p["expected_total_layer_time_capture_pct"],
        "expected_decode_like_time_capture_pct": p["expected_decode_like_time_capture_pct"],
        "measured_decode_tps": "",
        "measured_p95_token_latency_ms": "",
        "measurement_status": p["measurement_status"],
        "reason": p["reason"],
    } for p in policies])

    stability_rows = []
    for r in layer_rows:
        counts = json.loads(r["benchmark_counts"] or "{}")
        stability_rows.append({
            "layer": r["layer"],
            "active_request_count": r["active_request_count"],
            "benchmark_count": len(counts),
            "benchmark_counts": r["benchmark_counts"],
            "bottleneck_score": r["bottleneck_score"],
        })
    write_csv(out / "benchmark_layer_stability.csv", stability_rows)

    manifest = []
    if args.manifest and args.manifest.exists():
        manifest = list(iter_jsonl(args.manifest))
        with (out / "request_manifest.jsonl").open("w", encoding="utf-8") as f:
            for rec in manifest:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    request_rows = []
    if args.status_jsonl and args.status_jsonl.exists():
        for rec in iter_jsonl(args.status_jsonl):
            request_rows.append(rec)
    write_csv(out / "request_summary.csv", request_rows)

    write_json(out / "run_meta.json", {
        "type": "layer_bottleneck_analysis",
        "created_utc": utc_now(),
        "profiles": [str(p) for p in profiles],
        "profile_count": len(profiles),
        "profiled_nodes": total_nodes,
        "layer_count": LAYERS,
        "recommended_policy": "top10_hybrid_bottleneck",
        "measurement_boundary": "direct synchronized scheduler callback timing; policy speedups not measured until A/B replay",
    })

    make_html(out, layer_rows, op_rows, policies)
    print(json.dumps({"out_dir": str(out), "profiles": len(profiles), "profiled_nodes": total_nodes, "recommended": policies[7]["keep_layers"]}, indent=2))


def make_html(out: Path, layer_rows: list[dict[str, Any]], op_rows: list[dict[str, Any]], policies: list[dict[str, Any]]) -> None:
    def table(rows: list[dict[str, Any]], cols: list[str], limit: int = 40) -> str:
        body = ["<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr></thead><tbody>"]
        for r in rows[:limit]:
            body.append("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
        body.append("</tbody></table>")
        return "\n".join(body)

    policy_rows = [{
        "policy_name": p["policy_name"],
        "keep_layers": ",".join(map(str, p["keep_layers"])),
        "total_capture_pct": p["expected_total_layer_time_capture_pct"],
        "decode_capture_pct": p["expected_decode_like_time_capture_pct"],
        "status": p["measurement_status"],
    } for p in policies]
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Layer Bottleneck Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;color:#17233c}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th,td{{border:1px solid #d7deea;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#eef3fb}}
.warn{{background:#fff5d6;border:1px solid #e7c75f;padding:12px;margin:16px 0}}
</style></head><body>
<h1>Layer Bottleneck Report</h1>
<div class="warn">Layer timings are directly measured through synchronized scheduler callback profiling. Candidate policy speedups remain unmeasured until A/B replay.</div>
<h2>Top Layers</h2>
{table(layer_rows, ['layer','bottleneck_score','elapsed_ms','elapsed_share_pct','decode_like_ms','decode_like_share_pct','p95_request_layer_ms','traffic_share_pct','active_request_count'], 40)}
<h2>Policy Candidates</h2>
{table(policy_rows, ['policy_name','keep_layers','total_capture_pct','decode_capture_pct','status'], 20)}
<h2>Top Op Families</h2>
{table(op_rows, ['layer','op_family','node_count','elapsed_ms'], 80)}
</body></html>"""
    (out / "layer_bottleneck_report.html").write_text(doc, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inventory")
    pi.add_argument("--out-dir", type=Path, required=True)
    pi.add_argument("--model", type=Path)
    pi.add_argument("--llama-cli", type=Path)
    pi.set_defaults(func=inventory)

    pm = sub.add_parser("make-mini")
    pm.add_argument("--out", type=Path, required=True)
    pm.add_argument("--max-tokens", type=int, default=128)
    pm.add_argument("--terminal-tasks", type=int, default=5)
    pm.add_argument("--limit", type=int, default=0)
    pm.set_defaults(func=make_mini)

    pr = sub.add_parser("run")
    pr.add_argument("--manifest", type=Path, required=True)
    pr.add_argument("--out-dir", type=Path, required=True)
    pr.add_argument("--llama-cli", type=Path, required=True)
    pr.add_argument("--model", type=Path, required=True)
    pr.add_argument("--max-tokens", type=int, default=128)
    pr.add_argument("--seed", type=int, default=1)
    pr.add_argument("--detail", default="summary")
    pr.add_argument("--sync", default="layer")
    pr.add_argument("--timeout-s", type=int, default=1800)
    pr.add_argument("--resume", action="store_true")
    pr.set_defaults(func=run_manifest)

    pa = sub.add_parser("analyze")
    pa.add_argument("--profile-dir", type=Path)
    pa.add_argument("--profile-jsonl", type=Path)
    pa.add_argument("--status-jsonl", type=Path)
    pa.add_argument("--manifest", type=Path)
    pa.add_argument("--out-dir", type=Path, required=True)
    pa.set_defaults(func=analyze)

    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
