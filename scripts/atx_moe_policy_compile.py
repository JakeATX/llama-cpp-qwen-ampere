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


def main() -> int:
    args = parse_args()
    src = Path(args.policy_source)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policies = sorted(src.glob("combined_top_*pct_*_experts.atx.json"))
    if not policies:
        raise SystemExit(f"no ATX policies found under {src}")

    reports = []
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
