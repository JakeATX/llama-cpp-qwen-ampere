#!/usr/bin/env python3
"""Compare tensor dumps emitted by llama-perplexity KVarN diagnostics.

The tool intentionally reads only dump JSON/bin files. It does not import llama
or KVarN implementation code, so it can serve as an independent boundary check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DTYPES = {
    "f32": np.float32,
    "f16": np.float16,
}


def load_dump_index(path: Path) -> Dict[str, List[dict]]:
    if not path.is_dir():
        raise FileNotFoundError(f"dump directory not found: {path}")
    by_name: Dict[str, List[dict]] = {}
    for meta_path in sorted(path.glob("*.json")):
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["_json"] = meta_path
        meta["_bin"] = path / meta["bin"]
        by_name.setdefault(meta["name"], []).append(meta)
    return by_name


def parse_pair(spec: str) -> Tuple[str, str]:
    if "=" in spec:
        left, right = spec.split("=", 1)
    elif ":" in spec:
        left, right = spec.split(":", 1)
    else:
        left = right = spec
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"invalid pair spec: {spec!r}")
    return left, right


def tensor_array(meta: dict) -> np.ndarray:
    typ = meta.get("type")
    if typ not in DTYPES:
        raise ValueError(f"unsupported tensor type {typ!r} for {meta.get('name')}")
    data = np.fromfile(meta["_bin"], dtype=DTYPES[typ])
    expected = int(meta["n_bytes"]) // np.dtype(DTYPES[typ]).itemsize
    if data.size != expected:
        raise ValueError(f"{meta['_bin']} has {data.size} values, expected {expected}")
    return data.astype(np.float32, copy=False)


def compare_one(left: dict, right: dict) -> dict:
    if left["ne"] != right["ne"]:
        raise ValueError(f"shape mismatch for {left['name']} vs {right['name']}: {left['ne']} != {right['ne']}")
    l = tensor_array(left)
    r = tensor_array(right)
    if l.size != r.size:
        raise ValueError(f"value-count mismatch for {left['name']} vs {right['name']}: {l.size} != {r.size}")
    diff = r - l
    denom = float(np.dot(l, l)) + 1.0e-30
    nmse = float(np.dot(diff, diff) / denom)
    abs_diff = np.abs(diff)
    return {
        "left_name": left["name"],
        "right_name": right["name"],
        "left_index": left["index"],
        "right_index": right["index"],
        "n_values": int(l.size),
        "nmse": nmse,
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean()) if l.size else 0.0,
    }


def iter_pairs(left_idx: Dict[str, List[dict]], right_idx: Dict[str, List[dict]], specs: Iterable[str]) -> Iterable[Tuple[dict, dict]]:
    for spec in specs:
        left_name, right_name = parse_pair(spec)
        if left_name not in left_idx:
            raise KeyError(f"left dump missing tensor {left_name!r}")
        if right_name not in right_idx:
            raise KeyError(f"right dump missing tensor {right_name!r}")
        left_list = left_idx[left_name]
        right_list = right_idx[right_name]
        if len(left_list) != len(right_list):
            raise ValueError(
                f"occurrence mismatch for {left_name!r}/{right_name!r}: {len(left_list)} != {len(right_list)}")
        for left, right in zip(left_list, right_list):
            yield left, right


def common_pair_specs(left_idx: Dict[str, List[dict]], right_idx: Dict[str, List[dict]]) -> List[str]:
    return sorted(set(left_idx).intersection(right_idx))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--pair", action="append", default=[], help="tensor pair name, or left=right")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--max-nmse", type=float, default=math.inf)
    parser.add_argument("--max-abs", type=float, default=math.inf)
    args = parser.parse_args()

    left_idx = load_dump_index(args.left)
    right_idx = load_dump_index(args.right)
    specs = args.pair or common_pair_specs(left_idx, right_idx)
    if not specs:
        raise RuntimeError("no tensor pairs selected")

    rows = [compare_one(left, right) for left, right in iter_pairs(left_idx, right_idx, specs)]
    rows.sort(key=lambda r: (r["left_name"], r["left_index"], r["right_index"]))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    failed = False
    for row in rows:
        print(
            f"{row['left_name']} -> {row['right_name']} "
            f"nmse={row['nmse']:.6e} max_abs={row['max_abs']:.6e} mean_abs={row['mean_abs']:.6e}")
        if row["nmse"] > args.max_nmse or row["max_abs"] > args.max_abs:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
