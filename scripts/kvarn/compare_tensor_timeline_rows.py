#!/usr/bin/env python3
"""Compare named tensor rows across dumps with different batch splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_tensor_dumps import DTYPES
from compare_tensor_dumps import parse_pair


def load_occurrences(root: Path, name: str, skip_leading_short: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for meta_path in sorted(root.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("name") != name:
            continue
        typ = meta.get("type")
        if typ not in DTYPES:
            raise SystemExit(f"unsupported tensor type {typ!r} for {name}")
        ne = [int(x) for x in meta["ne"]]
        if len(ne) < 2:
            continue
        n_tok = int(ne[1])
        if n_tok <= 0:
            continue
        data = np.fromfile(root / meta["bin"], dtype=DTYPES[typ])
        expected = int(meta["n_bytes"]) // np.dtype(DTYPES[typ]).itemsize
        if data.size != expected:
            raise SystemExit(f"{meta['bin']} has {data.size} values, expected {expected}")
        arr = data.astype(np.float32).reshape(tuple(ne), order="F")
        if arr.ndim > 2:
            arr = np.squeeze(arr, axis=tuple(range(2, arr.ndim)))
        if arr.ndim != 2:
            raise SystemExit(f"{name} occurrence {meta_path} has unsupported shape {ne}")
        meta["_json"] = str(meta_path)
        meta["_arr"] = arr
        out.append(meta)
    if skip_leading_short and out and int(out[0]["ne"][1]) < 16:
        out = out[1:]
    if not out:
        raise SystemExit(f"missing tensor {name!r} under {root}")
    return out


def build_timeline(occurrences: list[dict[str, Any]], context_size: int) -> list[dict[str, Any]]:
    timeline = []
    chunk = 0
    pos = 0
    for meta in occurrences:
        n_tok = int(meta["ne"][1])
        if pos + n_tok > context_size:
            chunk += 1
            pos = 0
        timeline.append({
            "chunk": chunk,
            "pos0": pos,
            "pos1": pos + n_tok,
            "meta": meta,
        })
        pos += n_tok
        if pos == context_size:
            chunk += 1
            pos = 0
    return timeline


def select_row(timeline: list[dict[str, Any]], chunk: int, pos: int) -> tuple[np.ndarray, dict[str, Any], int]:
    for item in timeline:
        if int(item["chunk"]) != chunk:
            continue
        if int(item["pos0"]) <= pos < int(item["pos1"]):
            local = pos - int(item["pos0"])
            return item["meta"]["_arr"][:, local].astype(np.float32), item["meta"], local
    raise SystemExit(f"could not find chunk={chunk} pos={pos} in tensor timeline")


def nmse(ref: np.ndarray, got: np.ndarray) -> float:
    r = ref.astype(np.float64)
    g = got.astype(np.float64)
    d = g - r
    denom = float(np.dot(r, r))
    num = float(np.dot(d, d))
    if denom == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / denom


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return 1.0 if denom == 0.0 else float(np.dot(af, bf) / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, type=Path)
    ap.add_argument("--right", required=True, type=Path)
    ap.add_argument("--tensor", action="append", required=True,
                    help="Tensor name, or left=right pair when names differ across dumps.")
    ap.add_argument("--context-size", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--left-pos", type=int,
                    help="Override --pos for the left dump when left/right timelines use different origins.")
    ap.add_argument("--right-pos", type=int,
                    help="Override --pos for the right dump when left/right timelines use different origins.")
    ap.add_argument("--left-origin", type=int, default=0,
                    help="Native-position origin of the left tensor timeline. Ignored when --left-pos is set.")
    ap.add_argument("--right-origin", type=int, default=0,
                    help="Native-position origin of the right tensor timeline. Ignored when --right-pos is set.")
    ap.add_argument("--skip-leading-short", action="store_true")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    rows = []
    for spec in args.tensor:
        left_name, right_name = parse_pair(spec)
        left_tl = build_timeline(load_occurrences(args.left, left_name, args.skip_leading_short), args.context_size)
        right_tl = build_timeline(load_occurrences(args.right, right_name, args.skip_leading_short), args.context_size)
        left_pos = (args.pos - args.left_origin) if args.left_pos is None else args.left_pos
        right_pos = (args.pos - args.right_origin) if args.right_pos is None else args.right_pos
        if left_pos < 0 or right_pos < 0:
            raise SystemExit(
                f"negative timeline position after origin adjustment: "
                f"left_pos={left_pos} right_pos={right_pos} native_pos={args.pos}")
        left_row, left_meta, left_local = select_row(left_tl, args.chunk, left_pos)
        right_row, right_meta, right_local = select_row(right_tl, args.chunk, right_pos)
        if left_row.shape != right_row.shape:
            raise SystemExit(f"{left_name}->{right_name} row shape mismatch: {left_row.shape} != {right_row.shape}")
        diff = np.abs(right_row.astype(np.float64) - left_row.astype(np.float64))
        worst_d = int(np.argmax(diff)) if diff.size else 0
        row = {
            "left_name": left_name,
            "right_name": right_name,
            "chunk": args.chunk,
            "pos": args.pos,
            "left_pos": left_pos,
            "right_pos": right_pos,
            "left_index": int(left_meta["index"]),
            "right_index": int(right_meta["index"]),
            "left_local": left_local,
            "right_local": right_local,
            "n_values": int(left_row.size),
            "nmse": nmse(left_row, right_row),
            "cosine": cosine(left_row, right_row),
            "max_abs": float(diff[worst_d]) if diff.size else 0.0,
            "mean_abs": float(diff.mean()) if diff.size else 0.0,
            "worst_d": worst_d,
            "left_json": left_meta["_json"],
            "right_json": right_meta["_json"],
        }
        rows.append(row)
        print(
            f"{left_name}->{right_name} chunk={args.chunk} pos={args.pos} "
            f"left_pos={left_pos} right_pos={right_pos} "
            f"nmse={row['nmse']:.6e} cosine={row['cosine']:.9f} "
            f"max_abs={row['max_abs']:.6e} mean_abs={row['mean_abs']:.6e} "
            f"left_index={row['left_index']} right_index={row['right_index']} "
            f"left_local={left_local} right_local={right_local}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
