#!/usr/bin/env python3
"""Metal llama-server MTP acceptance harness (macOS port of atx_moe_bottleneck_acceptance.ps1)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    p.add_argument("--build-dir", default="build-atx-metal")
    p.add_argument("--llama-server")
    p.add_argument("--model", required=True)
    p.add_argument("--policy-source", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ctx-size", type=int, default=64000)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt-file", default="", help="Use this prompt text instead of the built-in coding prompt")
    p.add_argument("--prompt-repeat", type=int, default=1, help="Repeat the selected prompt N times to increase prefill length")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--ubatch-size", type=int, default=512)
    p.add_argument("--ncpu-moe", type=int, default=34)
    p.add_argument("--reference-ncpu-moe", action="store_true", help="Add -ncmoe to the reference scenario for CPU-MoE/offload comparison")
    p.add_argument("--reference-no-ncpu-moe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--known-fast-layers", default="25-28,31-39,40")
    p.add_argument("--attention-layers", default="3,7,11,15,19,23,27,31,35,39")
    p.add_argument("--target-decode-tps", type=float, default=80.0)
    p.add_argument("--reference-decode-tps", type=float, default=72.0)
    p.add_argument("--scenarios", nargs="+", default=[
        "reference", "known_fast_tail", "hybrid_top8", "hybrid_top16",
        "exact_v1_10pct", "direct_10pct", "bottleneck_auto",
    ])
    p.add_argument(
        "--policy-scenario",
        action="append",
        default=[],
        help="Add an explicit policy scenario as NAME=MODE=PATH.",
    )
    p.add_argument(
        "--layer-scenario",
        action="append",
        default=[],
        help="Add an inline whole-layer policy scenario as NAME=LAYERS, where LAYERS accepts comma/range syntax.",
    )
    p.add_argument(
        "--sweep-policy-glob",
        action="append",
        default=[],
        help="Evaluate policies matching this glob under the compiled policy directory.",
    )
    p.add_argument("--sweep-policy-mode", default="hybrid", choices=["exact-v1", "direct", "hybrid", "auto", "layer"])
    p.add_argument("--sweep-limit", type=int, default=0, help="Limit total glob-sweep scenarios, 0 means unlimited")
    p.add_argument("--prewarm-experts", default="", choices=["", "off", "lazy", "eager"])
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--no-mtp", action="store_true")
    p.add_argument("--server-ready-minutes", type=int, default=20)
    p.add_argument("--hardening", action="store_true", help="Run context/correctness hardening matrix")
    p.add_argument("--hardening-policy", default="", help="Explicit policy path for hardening best-candidate runs")
    p.add_argument("--hardening-mode", default="", choices=["", "exact-v1", "direct", "hybrid", "auto", "layer"])
    p.add_argument("--hardening-layers", default="", help="Inline whole-layer policy for hardening best-candidate runs")
    return p.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(x) for x in item.split("-", 1)]
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def coding_prompt(repo: Path) -> str:
    repeat = max(1, int(os.environ.get("ATX_BENCH_PROMPT_REPEAT", "1")))
    prompt_file = os.environ.get("ATX_BENCH_PROMPT_FILE", "")
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8").strip()
        if text:
            return "\n\n".join([text] * repeat)
    path = repo / "prompts" / "atx_moe_coding_prompts.txt"
    if path.exists():
        for part in path.read_text(encoding="utf-8").split("\n\n"):
            if len(part.strip()) > 200:
                return "\n\n".join([part.strip()] * repeat)
    text = (
        "You are editing a Python service. Implement a small LRU cache class with get, put, "
        "delete, clear, and stats methods. Include edge-case handling for capacity 0, "
        "overwrites, and missing keys. Return only a unified diff."
    )
    return "\n\n".join([text] * repeat)


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 900) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_stats_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    c = obj.get("counters", obj)
    return {
        "host_bytes": c.get("host_bytes_copied"),
        "host_ranges": c.get("host_expert_range_copy_calls", c.get("host_expert_copy_calls")),
        "single_ranges": c.get("host_expert_single_copy_calls"),
        "direct_hits": c.get("resident_direct_hit_slices"),
        "resident_staging": c.get("resident_staging_copy_calls"),
        "direct_mmvq": c.get("direct_kernel_dispatch_mmvq"),
        "direct_mmq": c.get("direct_kernel_dispatch_mmq"),
        "direct_cache_registrations": c.get("direct_cache_registrations"),
        "direct_cache_query_hits": c.get("direct_cache_query_hits"),
        "direct_cache_query_misses": c.get("direct_cache_query_misses"),
        "hot_stage_violations": c.get("hot_staging_violations"),
        "hot_staging_bytes": c.get("hot_staging_bytes"),
        "fallback_dispatches": c.get("direct_kernel_fallbacks"),
        "moe_named_node_misses": c.get("moe_named_node_misses"),
    }


def read_spec_summary(stderr_path: Path, resp: dict | None) -> dict:
    draft_generated = None
    draft_accepted = None
    draft_rate = None
    if resp and isinstance(resp.get("timings"), dict):
        t = resp["timings"]
        draft_generated = t.get("draft_n")
        draft_accepted = t.get("draft_n_accepted")
        if draft_generated and draft_accepted is not None and draft_generated > 0:
            draft_rate = round(draft_accepted / draft_generated, 5)
    if stderr_path.exists():
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(
            r"draft acceptance rate\s*=\s*([0-9.]+)\s*\(\s*([0-9]+)\s+accepted\s*/\s*([0-9]+)\s+generated",
            text,
        )
        if m:
            draft_rate = float(m.group(1))
            draft_accepted = int(m.group(2))
            draft_generated = int(m.group(3))
    return {
        "draft_generated": draft_generated,
        "draft_accepted": draft_accepted,
        "draft_acceptance_rate": draft_rate,
    }


def write_layer_policy(path: Path, name: str, layers: list[int], basis: str) -> Path:
    obj = {
        "schema_version": "atx-moe-residency-policy-v1",
        "policy_name": name,
        "keep_layers": layers,
        "basis": basis,
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def compile_policies(repo: Path, policy_source: Path, out_dir: Path, stats_json: Path | None, known_fast: str) -> None:
    cmd = [
        "python3", str(repo / "scripts" / "atx_moe_policy_compile.py"),
        "--policy-source", str(policy_source),
        "--out-dir", str(out_dir),
        "--mode", "auto",
        "--base-keep-layers", known_fast,
        "--swap-candidates", "3",
    ]
    if stats_json and stats_json.exists():
        cmd += ["--stats-json", str(stats_json)]
    subprocess.run(cmd, cwd=repo, check=True)


def sanitize_scenario_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:120]


def policy_args(policy: Path, mode: str, prewarm: str = "") -> list[str]:
    args = ["--moe-residency-policy", str(policy), "--moe-residency-mode", mode]
    if mode in {"direct", "hybrid", "auto"}:
        args += ["--moe-direct-require", "--moe-direct-strict-hot-no-stage"]
    if prewarm:
        args += ["--moe-prewarm-experts", prewarm]
    return args


def run_scenario(
    repo: Path,
    server_bin: Path,
    model: Path,
    out_dir: Path,
    name: str,
    extra_args: list[str],
    *,
    ctx: int,
    max_tokens: int,
    threads: int,
    batch_size: int,
    ubatch_size: int,
    use_ncpu_moe: bool,
    ncpu_moe: int,
    no_mtp: bool,
    ready_minutes: int,
) -> dict:
    raw_dir = out_dir / "raw"
    stats_dir = out_dir / "stats"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    port = free_port()
    stdout_path = raw_dir / f"{name}.stdout.log"
    stderr_path = raw_dir / f"{name}.stderr.log"
    stats_path = stats_dir / f"{name}.residency.json"

    server_args = [
        str(server_bin),
        "-m", str(model),
        "--alias", f"atx-metal-{name}",
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(ctx),
        "-t", str(threads),
        "-tb", str(threads),
        "-ngl", "999",
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "--parallel", "1",
        "--cache-ram", "0",
        "--reasoning", "on",
        "--reasoning-format", "deepseek",
        "--temp", "0",
        "--top-p", "1",
        "--no-webui",
        "--moe-residency-stats", str(stats_path),
    ]
    if not no_mtp:
        server_args += ["--spec-type", "mtp", "--spec-draft-n-max", "2"]
    if use_ncpu_moe:
        server_args += ["-ncmoe", str(ncpu_moe)]
    server_args += extra_args

    for p in (stdout_path, stderr_path, stats_path):
        if p.exists():
            p.unlink()

    proc = subprocess.Popen(
        server_args,
        cwd=repo,
        stdout=stdout_path.open("w"),
        stderr=stderr_path.open("w"),
    )
    t0 = time.perf_counter()
    error = ""
    resp = None
    ready = False
    try:
        deadline = time.time() + ready_minutes * 60
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                http_json("GET", f"http://127.0.0.1:{port}/health", timeout=2)
                ready = True
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(1)
        if not ready:
            raise RuntimeError("server did not become ready")
        body = {
            "prompt": coding_prompt(repo),
            "n_predict": max_tokens,
            "temperature": 0.0,
            "cache_prompt": False,
            "stop": [],
        }
        resp = http_json("POST", f"http://127.0.0.1:{port}/completion", body)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        try:
            http_json("POST", f"http://127.0.0.1:{port}/shutdown", timeout=5)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    elapsed = time.perf_counter() - t0
    timings = (resp or {}).get("timings") or {}
    decode = timings.get("predicted_per_second") or timings.get("predicted_n_per_second")
    prefill = timings.get("prompt_per_second") or timings.get("prompt_n_per_second")
    stats = read_stats_summary(stats_path)
    spec = read_spec_summary(stderr_path, resp)
    return {
        "scenario": name,
        "decode_tps": decode,
        "prefill_tps": prefill,
        "predicted_tokens": timings.get("predicted_n"),
        "elapsed_sec": round(elapsed, 2),
        "error": error,
        "stats_path": str(stats_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        **stats,
        **spec,
    }


def main() -> int:
    args = parse_args()
    if args.prompt_file:
        os.environ["ATX_BENCH_PROMPT_FILE"] = str(Path(args.prompt_file).resolve())
    os.environ["ATX_BENCH_PROMPT_REPEAT"] = str(args.prompt_repeat)
    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    server_bin = Path(args.llama_server) if args.llama_server else repo / args.build_dir / "bin" / "llama-server"
    model = Path(args.model)
    policy_dir = out / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)

    known_fast_policy = write_layer_policy(
        policy_dir / "known_fast_tail_layers.atx.json",
        "known_fast_tail_layers",
        parse_layers(args.known_fast_layers),
        "Known fast whole-layer tail residency candidate.",
    )
    attention_policy = write_layer_policy(
        policy_dir / "attention_layer_baseline.atx.json",
        "attention_layer_baseline",
        parse_layers(args.attention_layers),
        "Attention-spaced layer baseline.",
    )

    policy_source = Path(args.policy_source) if args.policy_source else None
    hybrid_10 = policy_dir / "combined_top_10pct_layer_experts.atx.hybrid_top_16_layers.atx.json"
    global_10 = policy_dir / "combined_top_10pct_global_experts.atx.json"
    bottleneck_policy = repo / "policies" / "atx_bottleneck_swap_in_0_drop_36_q4kxl_64k.atx.json"

    rows: list[dict] = []
    blockers: list[str] = []

    def add_row(row: dict) -> None:
        rows.append(row)
        if row.get("error"):
            blockers.append(f"- {row['scenario']}: {row['error']}")
        if not Path(row.get("stats_path", "")).exists():
            blockers.append(f"- {row['scenario']}: missing residency stats")

    if "reference" in args.scenarios:
        add_row(run_scenario(
            repo, server_bin, model, out, "reference",
            [], ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=args.reference_ncpu_moe and not args.reference_no_ncpu_moe,
            ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    if "known_fast_tail" in args.scenarios:
        add_row(run_scenario(
            repo, server_bin, model, out, "known_fast_tail",
            ["--moe-residency-policy", str(known_fast_policy), "--moe-residency-mode", "layer"],
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    seed_stats = next((r.get("stats_path") for r in rows if r["scenario"] == "known_fast_tail"), None)
    if policy_source and policy_source.exists() and not args.skip_compile:
        compile_policies(
            repo, policy_source, policy_dir,
            Path(seed_stats) if seed_stats else None,
            args.known_fast_layers,
        )
        hybrid_10 = policy_dir / "combined_top_10pct_layer_experts.atx.hybrid_top_16_layers.atx.json"
        global_10 = policy_dir / "combined_top_10pct_global_experts.atx.json"

    hybrid8 = policy_dir / "combined_top_10pct_layer_experts.atx.hybrid_top_8_layers.atx.json"
    if hybrid8.exists() and "hybrid_top8" in args.scenarios:
        add_row(run_scenario(
            repo, server_bin, model, out, "hybrid_top8",
            policy_args(hybrid8, "hybrid", args.prewarm_experts),
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    if hybrid_10.exists() and "hybrid_top16" in args.scenarios:
        add_row(run_scenario(
            repo, server_bin, model, out, "hybrid_top16",
            policy_args(hybrid_10, "hybrid", args.prewarm_experts),
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    if global_10.exists() and "exact_v1_10pct" in args.scenarios:
        add_row(run_scenario(
            repo, server_bin, model, out, "exact_v1_10pct",
            ["--moe-residency-policy", str(global_10), "--moe-residency-mode", "exact-v1"],
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    if global_10.exists() and "direct_10pct" in args.scenarios:
        add_row(run_scenario(
            repo, server_bin, model, out, "direct_10pct",
            policy_args(global_10, "direct", args.prewarm_experts),
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    if args.hardening:
        if args.hardening_layers:
            best_policy = write_layer_policy(
                policy_dir / "hardening_best_candidate.atx.json",
                "hardening_best_candidate",
                parse_layers(args.hardening_layers),
                "Inline best-candidate policy supplied for hardening.",
            )
            best_mode = args.hardening_mode or "layer"
        elif args.hardening_policy:
            best_policy = Path(args.hardening_policy)
            best_mode = args.hardening_mode or "hybrid"
        else:
            best_policy = hybrid_10 if hybrid_10.exists() else known_fast_policy
            best_mode = "hybrid" if hybrid_10.exists() else "layer"
        for ctx in (4096, 8192, 16384, 32768, args.ctx_size):
            add_row(run_scenario(
                repo, server_bin, model, out, f"harden_ctx_{ctx}",
                policy_args(best_policy, best_mode, args.prewarm_experts or "eager"),
                ctx=ctx, max_tokens=min(args.max_tokens, 24),
                threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
                use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
                ready_minutes=args.server_ready_minutes,
            ))

    if bottleneck_policy.exists() and "bottleneck_auto" in args.scenarios:
        add_row(run_scenario(
            repo, server_bin, model, out, "bottleneck_auto",
            policy_args(bottleneck_policy, "auto", args.prewarm_experts) + ["--moe-cold-coalesce-gap", "0"],
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    for spec in args.layer_scenario:
        try:
            name, layers = spec.split("=", 1)
        except ValueError:
            raise SystemExit(f"invalid --layer-scenario {spec!r}; expected NAME=LAYERS")
        scenario_name = sanitize_scenario_name(name)
        policy = write_layer_policy(
            policy_dir / f"{scenario_name}.atx.json",
            scenario_name,
            parse_layers(layers),
            "Inline whole-layer policy scenario supplied to server acceptance harness.",
        )
        add_row(run_scenario(
            repo, server_bin, model, out, scenario_name,
            policy_args(policy, "layer", args.prewarm_experts),
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    for spec in args.policy_scenario:
        try:
            name, mode, policy_path = spec.split("=", 2)
        except ValueError:
            raise SystemExit(f"invalid --policy-scenario {spec!r}; expected NAME=MODE=PATH")
        policy = Path(policy_path)
        add_row(run_scenario(
            repo, server_bin, model, out, sanitize_scenario_name(name),
            policy_args(policy, mode, args.prewarm_experts),
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    sweep_paths: list[Path] = []
    for pattern in args.sweep_policy_glob:
        sweep_paths.extend(sorted(policy_dir.glob(pattern)))
    seen_sweep: set[Path] = set()
    for policy in sweep_paths:
        if policy in seen_sweep:
            continue
        seen_sweep.add(policy)
        if args.sweep_limit and len(seen_sweep) > args.sweep_limit:
            break
        add_row(run_scenario(
            repo, server_bin, model, out, sanitize_scenario_name(f"sweep_{policy.stem}"),
            policy_args(policy, args.sweep_policy_mode, args.prewarm_experts),
            ctx=args.ctx_size, max_tokens=args.max_tokens,
            threads=args.threads, batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            use_ncpu_moe=False, ncpu_moe=args.ncpu_moe, no_mtp=args.no_mtp,
            ready_minutes=args.server_ready_minutes,
        ))

    ref = next((r for r in rows if r["scenario"] == "reference"), None)
    best = max((r for r in rows if r.get("decode_tps")), key=lambda r: r["decode_tps"], default=None)
    ref_tps = ref.get("decode_tps") if ref else None
    for r in rows:
        r["speedup_vs_reference"] = (r["decode_tps"] / ref_tps) if ref_tps and r.get("decode_tps") else None

    summary = {
        "out_dir": str(out),
        "target_decode_tps": args.target_decode_tps,
        "reference_decode_tps": args.reference_decode_tps,
        "best_scenario": best["scenario"] if best else None,
        "best_decode_tps": best.get("decode_tps") if best else None,
        "reproduced_reference_decode_tps": ref_tps,
        "pass_80_tps": bool(best and best.get("decode_tps") and best["decode_tps"] >= args.target_decode_tps),
        "pass_reference_1_10x": bool(best and best.get("decode_tps") and best["decode_tps"] >= 1.10 * args.reference_decode_tps),
        "rows": rows,
        "blockers": blockers,
    }
    (out / "acceptance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with (out / "acceptance_table.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
