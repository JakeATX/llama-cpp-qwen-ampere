#!/usr/bin/env python3
"""Compile ATX MoE policy artifacts into runtime policy candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-source", required=True, help="Directory containing HF *.atx.json policies")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", default="hybrid", choices=["direct", "layer", "hybrid", "auto"])
    p.add_argument("--budget-gib", type=float, action="append", default=[])
    p.add_argument("--ctx-size", type=int, action="append", default=[])
    p.add_argument("--stats-json", action="append", default=[], help="Residency stats JSON used to rank bottleneck layers")
    p.add_argument("--max-promote-layers", type=int, default=24)
    p.add_argument("--host-byte-reduction-target", type=float, default=0.80)
    p.add_argument("--full-hot-top-layers", type=int, default=0)
    p.add_argument("--attention-baseline-layers", default="", help="Optional comma/range layer list to emit as attention baseline")
    p.add_argument("--base-keep-layers", default="", help="Known fast layer set to preserve or use as the starting point for bottleneck candidates")
    p.add_argument("--swap-candidates", type=int, default=0, help="Emit fixed-budget one-layer swap candidates from bottleneck rank into base keep layers")
    return p.parse_args()


def policy_summary(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    selection = obj.get("selection", {})
    cells = obj.get("keep_layer_experts", [])
    layers = sorted({int(c["layer"]) for c in cells if isinstance(c, dict) and "layer" in c})
    return {
        "name": obj.get("policy_name", path.stem),
        "source": str(path),
        "cells": len(cells),
        "layers": len(layers),
        "selection": selection,
    }


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(x) for x in item.split("-", 1)]
            layers.extend(range(lo, hi + 1))
        else:
            layers.append(int(item))
    return sorted(set(layers))


def write_layer_policy(out: Path, name: str, layers: list[int], basis: str, extra: dict | None = None) -> dict:
    obj = {
        "schema_version": "atx-moe-residency-policy-v1",
        "policy_name": name,
        "keep_layers": sorted(set(layers)),
        "basis": basis,
    }
    if extra:
        obj.update(extra)
    target = out / f"{name}.atx.json"
    target.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return {
        "name": name,
        "source": "compiler",
        "cells": 0,
        "layers": len(set(layers)),
        "compiled_policy": str(target),
        "runtime_mode": "layer",
        "promoted_layers": sorted(set(layers)),
    }


def main() -> int:
    args = parse_args()
    src = Path(args.policy_source)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policies = sorted(src.glob("combined_top_*pct_*_experts.atx.json"))
    if not policies:
        raise SystemExit(f"no ATX policies found under {src}")

    reports = []
    base_keep_layers = parse_layers(args.base_keep_layers)
    bottleneck_rank = []
    bottleneck_total_host_bytes = 0
    for stats_path in [Path(p) for p in args.stats_json]:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        counters = stats.get("counters", {})
        bottleneck_total_host_bytes += int(counters.get("host_bytes_copied", 0) or 0)
        layer_rows = stats.get("bottleneck_layers") or []
        if not layer_rows:
            layer_rows = [
                {"layer": int(layer), **values}
                for layer, values in stats.get("per_layer", {}).items()
                if int(values.get("host_bytes_copied", 0) or 0) > 0
            ]
        for row in layer_rows:
            host_bytes = int(row.get("host_bytes_copied", 0) or 0)
            single_ranges = int(row.get("host_expert_single_copy_calls", 0) or 0)
            cold_slices = int(row.get("cold_expert_miss_slices", 0) or 0)
            submit_ns = int(row.get("host_copy_submit_ns", 0) or 0)
            score = host_bytes + single_ranges * 1048576 + cold_slices * 262144 + submit_ns
            bottleneck_rank.append({
                "layer": int(row["layer"]),
                "score": score,
                "host_bytes_copied": host_bytes,
                "single_ranges": single_ranges,
                "cold_slices": cold_slices,
                "source_stats": str(stats_path),
            })
    merged_bottlenecks = {}
    for row in bottleneck_rank:
        if row["layer"] not in merged_bottlenecks:
            merged_bottlenecks[row["layer"]] = dict(row)
        else:
            dst = merged_bottlenecks[row["layer"]]
            dst["score"] += row["score"]
            dst["host_bytes_copied"] += row["host_bytes_copied"]
            dst["single_ranges"] += row["single_ranges"]
            dst["cold_slices"] += row["cold_slices"]
    ranked_bottleneck_layers = sorted(merged_bottlenecks.values(), key=lambda r: r["score"], reverse=True)
    if ranked_bottleneck_layers:
        cumulative = 0
        selected = []
        for row in ranked_bottleneck_layers:
            selected.append(row["layer"])
            cumulative += row["host_bytes_copied"]
            if len(selected) >= args.max_promote_layers:
                break
            if bottleneck_total_host_bytes > 0 and cumulative / bottleneck_total_host_bytes >= args.host_byte_reduction_target:
                break
        ranked_layer_ids = [row["layer"] for row in ranked_bottleneck_layers]
        for n_layers in (4, 8, 10, 12, 13, 16, args.max_promote_layers):
            if n_layers <= 0:
                continue
            top_layers = ranked_layer_ids[:n_layers]
            reports.append(write_layer_policy(
                out,
                f"bottleneck_top_{n_layers}_layers",
                top_layers,
                "Whole-layer bottleneck candidate ranked by measured host bytes, range fragmentation, and submit wait.",
                {
                    "bottleneck_rank": ranked_bottleneck_layers,
                    "bottleneck_source_stats": args.stats_json,
                    "runtime_mode": "auto",
                },
            ))
            if base_keep_layers:
                preserved = list(base_keep_layers[:n_layers])
                needed = max(0, n_layers - len(preserved))
                hybrid_layers = sorted(set(preserved + [layer for layer in ranked_layer_ids if layer not in preserved][:needed]))
                reports.append(write_layer_policy(
                    out,
                    f"bottleneck_base_preserve_{n_layers}_layers",
                    hybrid_layers,
                    "Known fast base layers preserved first, then measured bottleneck layers fill remaining residency budget.",
                    {
                        "base_keep_layers": base_keep_layers,
                        "bottleneck_rank": ranked_bottleneck_layers,
                        "bottleneck_source_stats": args.stats_json,
                        "runtime_mode": "auto",
                    },
                ))
        if base_keep_layers and args.swap_candidates > 0:
            base = list(base_keep_layers)
            base_set = set(base)
            incoming = [layer for layer in ranked_layer_ids if layer not in base_set][:args.swap_candidates]
            removable = list(reversed(base))
            for add_layer in incoming:
                for drop_layer in removable:
                    swapped = sorted((base_set - {drop_layer}) | {add_layer})
                    reports.append(write_layer_policy(
                        out,
                        f"bottleneck_swap_in_{add_layer}_drop_{drop_layer}",
                        swapped,
                        "Fixed-budget one-layer swap candidate for empirical throughput search.",
                        {
                            "base_keep_layers": base_keep_layers,
                            "swap_in": add_layer,
                            "swap_out": drop_layer,
                            "bottleneck_rank": ranked_bottleneck_layers,
                            "bottleneck_source_stats": args.stats_json,
                            "runtime_mode": "auto",
                        },
                    ))
        for policy in policies:
            if "layer_experts" not in policy.name:
                continue
            obj = json.loads(policy.read_text(encoding="utf-8"))
            out_obj = dict(obj)
            out_obj["policy_name"] = f"{obj.get('policy_name', policy.stem)}_bottleneck_first"
            out_obj["bottleneck_keep_layers"] = selected
            out_obj["keep_layers"] = selected
            out_obj["bottleneck_rank"] = ranked_bottleneck_layers
            out_obj["bottleneck_source_stats"] = args.stats_json
            out_obj["projected_host_byte_reduction_target"] = args.host_byte_reduction_target
            out_obj["runtime_mode"] = "auto"
            if args.full_hot_top_layers > 0:
                full_hot_layers = selected[:args.full_hot_top_layers]
                existing = out_obj.get("keep_layer_experts", [])
                full_hot_cells = [{"layer": layer, "expert": expert} for layer in full_hot_layers for expert in range(256)]
                out_obj["keep_layer_experts"] = existing + full_hot_cells
                out_obj["full_hot_layers"] = full_hot_layers
            target = out / f"{policy.stem}.bottleneck_first.atx.json"
            target.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
            reports.append(policy_summary(target) | {
                "compiled_policy": str(target),
                "runtime_mode": "auto",
                "bottleneck_keep_layers": selected,
                "bottleneck_total_host_bytes": bottleneck_total_host_bytes,
                "bottleneck_source_stats": args.stats_json,
            })
    if args.attention_baseline_layers:
        reports.append(write_layer_policy(
            out,
            "attention_layer_baseline",
            parse_layers(args.attention_baseline_layers),
            "User-provided known fast layer baseline for ATX comparison.",
        ))
    if base_keep_layers:
        reports.append(write_layer_policy(
            out,
            "known_fast_base_layers",
            base_keep_layers,
            "Known fast base layer set supplied to the bottleneck compiler.",
        ))

    for policy in policies:
        target = out / policy.name
        shutil.copyfile(policy, target)
        summary = policy_summary(target)
        reports.append(summary | {
            "compiled_policy": str(target),
            "runtime_mode": args.mode,
            "budgets_gib": args.budget_gib,
            "ctx_sizes": args.ctx_size,
        })
        if "layer_experts" in policy.name:
            obj = json.loads(target.read_text(encoding="utf-8"))
            counts = Counter(int(cell["layer"]) for cell in obj.get("keep_layer_experts", []) if isinstance(cell, dict) and "layer" in cell)
            ranked_layers = [layer for layer, _ in counts.most_common()]
            for n_layers in (4, 8, 12, 16):
                hybrid = dict(obj)
                hybrid["policy_name"] = f"{obj.get('policy_name', target.stem)}_hybrid_top_{n_layers}_layers"
                hybrid["keep_layers"] = ranked_layers[:n_layers]
                hybrid["hybrid_note"] = "ATX compiler: whole-layer promote highest-density salience layers and keep exact hot cells elsewhere."
                hybrid_target = out / f"{target.stem}.hybrid_top_{n_layers}_layers.atx.json"
                hybrid_target.write_text(json.dumps(hybrid, indent=2), encoding="utf-8")
                reports.append(policy_summary(hybrid_target) | {
                    "compiled_policy": str(hybrid_target),
                    "runtime_mode": "hybrid",
                    "promoted_layers": ranked_layers[:n_layers],
                    "budgets_gib": args.budget_gib,
                    "ctx_sizes": args.ctx_size,
                })

    (out / "compiled_policy_index.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    lines = ["# ATX MoE Policy Compile Report", ""]
    for r in reports:
        lines.append(f"- {r['name']}: {r['cells']} layer-expert cells across {r['layers']} layers -> `{Path(r['compiled_policy']).name}`")
    (out / "policy_compile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
