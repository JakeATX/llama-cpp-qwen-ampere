#!/usr/bin/env python3
"""ATX CUDA MoE residency acceptance harness."""

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
    p.add_argument("--build-dir", default="build-atx-cuda")
    p.add_argument("--llama-cli")
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--attention-policy")
    p.add_argument("--bottleneck-policy")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ctx-size", type=int, default=512)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--target-decode-tps", type=float, default=80.0)
    p.add_argument("--reference-layer-tps", type=float, default=72.0)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def median_tps(bench_json: Path) -> float | None:
    data = json.loads(bench_json.read_text(encoding="utf-8"))
    vals = [r.get("generation_tps") for r in data.get("runs", []) if r.get("generation_tps") is not None]
    return statistics.median(vals) if vals else None


def run(args: argparse.Namespace, cmd: list[str], log: Path) -> int:
    proc = subprocess.run(cmd, cwd=args.repo, text=True, capture_output=True)
    log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8", errors="replace")
    return proc.returncode


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        rc = run(args, ["cmake", "--build", args.build_dir, "--config", "Release", "--target", "llama-cli", "-j", "8"], out / "build.log")
        if rc != 0:
            (out / "blockers_build.md").write_text("# Build failed\n\nSee build.log.\n", encoding="utf-8")
            return rc

    llama_cli = args.llama_cli or str(repo / args.build_dir / "bin" / "llama-cli.exe")
    rows = []
    blockers = []
    bench = repo / "scripts" / "atx_moe_bench.py"
    for model in args.model:
        model_name = Path(model).stem
        scenarios = [
            ("off", None, "off"),
            ("exact-v1", args.policy, "exact-v1"),
            ("direct", args.policy, "direct"),
            ("hybrid", args.policy, "hybrid"),
        ]
        if args.attention_policy:
            scenarios.append(("attention-layer", args.attention_policy, "layer"))
        if args.bottleneck_policy:
            scenarios.append(("bottleneck-auto", args.bottleneck_policy, "auto"))
        for label, policy, mode in scenarios:
            mode_out = out / model_name / mode
            cmd = [
                "python", str(bench),
                "--llama-cli", llama_cli,
                "--model", model,
                "--mode", mode,
                "--ctx-size", str(args.ctx_size),
                "--tokens", str(args.tokens),
                "--repetitions", str(args.repetitions),
                "--out", str(mode_out),
            ]
            if policy:
                cmd += ["--policy", policy]
            rc = run(args, cmd, out / f"{model_name}_{mode}.runner.log")
            bj = mode_out / f"bench_{mode}.json"
            tps = median_tps(bj) if bj.exists() else None
            rows.append({"model": model_name, "scenario": label, "mode": mode, "returncode": rc, "median_generation_tps": tps})
            if rc != 0:
                blockers.append(f"- {model_name} {label}: runner failed, see `{model_name}_{mode}.runner.log`")

    off_by_model = {r["model"]: r for r in rows if r["scenario"] == "off"}
    exact_by_model = {r["model"]: r for r in rows if r["scenario"] == "exact-v1"}
    attention_by_model = {r["model"]: r for r in rows if r["scenario"] == "attention-layer"}
    for r in rows:
        base = off_by_model.get(r["model"], {}).get("median_generation_tps")
        exact = exact_by_model.get(r["model"], {}).get("median_generation_tps")
        attention = attention_by_model.get(r["model"], {}).get("median_generation_tps") or args.reference_layer_tps
        r["speedup_vs_off"] = (r["median_generation_tps"] / base) if base and r["median_generation_tps"] else None
        r["speedup_vs_exact_v1"] = (r["median_generation_tps"] / exact) if exact and r["median_generation_tps"] else None
        r["speedup_vs_layer_reference"] = (r["median_generation_tps"] / attention) if attention and r["median_generation_tps"] else None
        if r["scenario"] == "bottleneck-auto" and r["median_generation_tps"]:
            if r["median_generation_tps"] < args.target_decode_tps and r["median_generation_tps"] < 1.10 * attention:
                blockers.append(
                    f"- {r['model']} bottleneck-auto: {r['median_generation_tps']:.2f} tok/s did not reach "
                    f"{args.target_decode_tps:.2f} tok/s or 1.10x layer reference ({attention:.2f})")

    summary = {"rows": rows, "blockers": blockers}
    (out / "acceptance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "acceptance_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    html_rows = "\n".join("<tr>" + "".join(f"<td>{r.get(k)}</td>" for k in rows[0].keys()) + "</tr>" for r in rows) if rows else ""
    headers = "".join(f"<th>{k}</th>" for k in rows[0].keys()) if rows else ""
    (out / "acceptance_report.html").write_text(f"<table><thead><tr>{headers}</tr></thead><tbody>{html_rows}</tbody></table>\n", encoding="utf-8")
    if blockers:
        bdir = out / "blockers"
        bdir.mkdir(exist_ok=True)
        (bdir / "failed_gates.md").write_text("# Failed Gates\n\n" + "\n".join(blockers) + "\n", encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
