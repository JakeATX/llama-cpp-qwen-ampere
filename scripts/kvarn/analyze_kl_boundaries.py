#!/usr/bin/env python3
"""Annotate KL spikes with KVarN sink/body/tail boundary geometry."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any


def classify_pos(pos: int, n_ctx: int, sink: int, tail: int, group: int) -> dict[str, Any]:
    body_start = sink
    tail_start = max(sink, n_ctx - tail)
    body_end = tail_start - 1

    if pos < sink:
        region = "sink"
        record = None
        offset = None
    elif pos >= tail_start:
        region = "tail"
        record = None
        offset = None
    else:
        region = "body"
        body_pos = pos - sink
        record = body_pos // group
        offset = body_pos % group

    boundaries = [0, sink, tail_start, n_ctx]
    if tail_start > sink:
        boundaries.extend(range(sink + group, tail_start, group))
    distance = min(abs(pos - b) for b in boundaries)
    nearest = min(boundaries, key=lambda b: abs(pos - b))

    return {
        "region": region,
        "record": record,
        "offset": offset,
        "nearest_boundary": nearest,
        "distance_to_boundary": distance,
        "body_start": body_start,
        "body_end": body_end,
        "tail_start": tail_start,
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item: dict[str, Any] = dict(row)
            for key in ("chunk", "local_index", "logit_pos", "global_logit_index"):
                if key in row and row[key] != "":
                    item[key] = int(row[key])
            item["target_pos"] = int(row["target_pos"])
            item["global_target_index"] = int(row["global_target_index"])
            item["target_token"] = int(row["target_token"])
            item["kld"] = float(row["kld"])
            item["p_diff"] = float(row["p_diff"])
            rows.append(item)
    return rows


def annotate_capture_plan(
        row: dict[str, Any],
        *,
        boundary_min_tokens: int | None,
        query_block: int) -> None:
    logit_pos = row.get("logit_pos")
    if logit_pos is None:
        row["capture_tokens"] = None
        row["capture_iq"] = None
        row["capture_call_index"] = None
        return
    if query_block <= 0:
        raise SystemExit("--query-block must be positive")

    # Boundary dumps are emitted for the upper token count of the current query
    # block. A logit at position 2366 in a 128-query block belongs to the dump
    # with n_tokens=2432 and iq=62.
    capture_tokens = ((int(logit_pos) // query_block) + 1) * query_block
    if boundary_min_tokens is not None:
        capture_tokens = max(capture_tokens, boundary_min_tokens)
    block_start = capture_tokens - query_block
    row["capture_tokens"] = capture_tokens
    row["capture_iq"] = int(logit_pos) - block_start
    if boundary_min_tokens is None:
        row["capture_call_index"] = None
    else:
        row["capture_call_index"] = (capture_tokens - boundary_min_tokens) // query_block


def read_base_tokens(path: Path) -> list[int]:
    with path.open("rb") as f:
        magic = f.read(8)
        if magic not in (b"_logits_", b"_logp16_", b"_logp16n"):
            raise SystemExit(f"unsupported KL base magic {magic!r}")
        n_ctx = struct.unpack("<i", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        if n_ctx <= 1 or n_vocab <= 0 or n_chunk <= 0:
            raise SystemExit(f"invalid KL base header n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk}")
        raw = f.read(n_ctx * n_chunk * 4)
        if len(raw) != n_ctx * n_chunk * 4:
            raise SystemExit(f"truncated KL base token table in {path}")
    return list(struct.unpack("<" + "i" * (n_ctx * n_chunk), raw))


def token_class(token_id: int) -> str:
    if token_id in {105, 106}:
        return "turn-control"
    if token_id in {107, 251}:
        return "newline"
    if token_id < 0:
        return "invalid"
    return "ordinary"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if fraction <= 0:
        return values[0]
    if fraction >= 1:
        return values[-1]
    p = fraction * (len(values) - 1)
    i = int(p)
    f = p - i
    return (1.0 - f) * values[i] + f * values[min(i + 1, len(values) - 1)]


def split_stats(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(float(row["kld"]))
    for name, values in sorted(buckets.items()):
        result[name] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "p99": percentile(values, 0.99),
            "max": max(values),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl-csv", required=True, type=Path)
    ap.add_argument("--base-file", type=Path)
    ap.add_argument("--n-ctx", required=True, type=int)
    ap.add_argument("--sink", type=int, default=128)
    ap.add_argument("--tail", type=int, default=128)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--turn-window", type=int, default=4)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--boundary-min-tokens", type=int, default=None,
                    help="When provided, annotate top rows with boundary call index for captures starting at this token count.")
    ap.add_argument("--query-block", type=int, default=128,
                    help="Number of query rows per boundary dump/capture block.")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    args = ap.parse_args()

    rows = read_rows(args.kl_csv)
    base_tokens = read_base_tokens(args.base_file) if args.base_file else []
    turn_positions = {i for i, tok in enumerate(base_tokens) if tok in {105, 106}}
    for row in rows:
        row.update(classify_pos(row["target_pos"], args.n_ctx, args.sink, args.tail, args.group))
        row["token_class"] = token_class(row["target_token"])
        if base_tokens:
            gi = row["global_target_index"]
            if gi < 0 or gi >= len(base_tokens):
                row["base_token_match"] = False
                row["near_turn"] = False
            else:
                row["base_token_match"] = base_tokens[gi] == row["target_token"]
                row["near_turn"] = any((gi + delta) in turn_positions for delta in range(-args.turn_window, args.turn_window + 1))
        else:
            row["base_token_match"] = None
            row["near_turn"] = None
        annotate_capture_plan(row, boundary_min_tokens=args.boundary_min_tokens, query_block=args.query_block)

    top = sorted(rows, key=lambda r: r["kld"], reverse=True)[: args.top_n]
    region_counts = Counter(r["region"] for r in top)
    near_counts = Counter("<=4" if r["distance_to_boundary"] <= 4 else "<=16" if r["distance_to_boundary"] <= 16 else ">16" for r in top)
    record_counts = Counter(str(r["record"]) for r in top if r["record"] is not None)
    class_counts = Counter(r["token_class"] for r in top)
    turn_counts = Counter(str(r["near_turn"]) for r in top)

    summary = {
        "kl_csv": str(args.kl_csv),
        "n_ctx": args.n_ctx,
        "sink": args.sink,
        "tail": args.tail,
        "group": args.group,
        "body_range": [args.sink, max(args.sink, args.n_ctx - args.tail) - 1],
        "tail_start": max(args.sink, args.n_ctx - args.tail),
        "row_count": len(rows),
        "base_token_count": len(base_tokens),
        "turn_window": args.turn_window,
        "boundary_min_tokens": args.boundary_min_tokens,
        "query_block": args.query_block,
        "top_n": len(top),
        "top_region_counts": dict(region_counts),
        "top_token_class_counts": dict(class_counts),
        "top_near_turn_counts": dict(turn_counts),
        "top_boundary_distance_counts": dict(near_counts),
        "top_body_record_counts": dict(record_counts),
        "region_stats": split_stats(rows, "region"),
        "token_class_stats": split_stats(rows, "token_class"),
        "near_turn_stats": split_stats(rows, "near_turn") if base_tokens else {},
        "top_rows": top,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# KL Boundary Correlation",
            "",
            f"- KL CSV: `{args.kl_csv}`",
            f"- geometry: n_ctx={args.n_ctx}, sink={args.sink}, tail={args.tail}, group={args.group}",
            f"- sealed body range: {summary['body_range'][0]}..{summary['body_range'][1]}",
            f"- live tail starts: {summary['tail_start']}",
            f"- top region counts: `{dict(region_counts)}`",
            f"- top token-class counts: `{dict(class_counts)}`",
            f"- top near-turn counts: `{dict(turn_counts)}`",
            f"- top boundary-distance counts: `{dict(near_counts)}`",
            f"- top body-record counts: `{dict(record_counts)}`",
            f"- capture plan: boundary_min_tokens={args.boundary_min_tokens}, query_block={args.query_block}",
            "",
            "## Split Stats",
            "",
            "| split | bucket | count | mean KLD | p99 KLD | max KLD |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for split, stats in (
            ("region", summary["region_stats"]),
            ("token_class", summary["token_class_stats"]),
            ("near_turn", summary["near_turn_stats"]),
        ):
            for bucket, st in stats.items():
                lines.append(
                    f"| {split} | {bucket} | {int(st['count'])} | {st['mean']:.6f} | "
                    f"{st['p99']:.6f} | {st['max']:.6f} |"
                )
        lines += [
            "",
            "## Top Rows",
            "",
            "| rank | KLD | p_diff | logit | target | token | class | near turn | region | record | offset | capture tokens | capture iq | call | nearest boundary | distance |",
            "|---:|---:|---:|---:|---:|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, row in enumerate(top, 1):
            call = "" if row["capture_call_index"] is None else row["capture_call_index"]
            lines.append(
                f"| {i} | {row['kld']:.6f} | {row['p_diff']:.6f} | {row.get('logit_pos', '')} | "
                f"{row['target_pos']} | {row['target_token']} | {row['token_class']} | {row['near_turn']} | {row['region']} | "
                f"{'' if row['record'] is None else row['record']} | "
                f"{'' if row['offset'] is None else row['offset']} | {row['capture_tokens']} | "
                f"{row['capture_iq']} | {call} | {row['nearest_boundary']} | "
                f"{row['distance_to_boundary']} |"
            )
        args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.json_out and not args.md_out:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
