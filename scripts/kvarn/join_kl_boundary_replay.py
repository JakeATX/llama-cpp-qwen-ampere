#!/usr/bin/env python3
"""Join KL spike rows to exact KVarN boundary replay rows.

This diagnostic does not prove native-KV correctness. It ties the KL gate rows
to the boundary call/iq and reports the local body-store replay errors for the
same query rows, so the next native-KV capture can target concrete rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_kl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("chunk", "local_index", "logit_pos", "target_pos", "global_logit_index", "global_target_index", "target_token"):
            if key in row and row[key] != "":
                row[key] = int(row[key])
        for key in ("kld", "p_diff"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def capture_location(logit_pos: int, boundary_min_tokens: int, query_block: int) -> tuple[int, int, int]:
    capture_tokens = ((logit_pos // query_block) + 1) * query_block
    capture_tokens = max(capture_tokens, boundary_min_tokens)
    call_index = (capture_tokens - boundary_min_tokens) // query_block
    iq = logit_pos - (capture_tokens - query_block)
    return capture_tokens, call_index, iq


def read_replay_rows(root: Path, call_index: int, iq: int) -> dict[str, Any]:
    path = root / "boundary" / f"call_{call_index:06d}" / "f16_truth_full_qo_summary.csv"
    if not path.exists():
        return {
            "path": str(path),
            "present": False,
            "head_count": 0,
            "max_nmse": "",
            "max_abs": "",
            "worst_ih": "",
            "worst_d": "",
        }
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if int(row["iq"]) == iq]
    if not rows:
        return {
            "path": str(path),
            "present": True,
            "head_count": 0,
            "max_nmse": "",
            "max_abs": "",
            "worst_ih": "",
            "worst_d": "",
        }
    worst_nmse = max(rows, key=lambda row: float(row["out_nmse"]))
    worst_abs = max(rows, key=lambda row: float(row["out_max_abs"]))
    return {
        "path": str(path),
        "present": True,
        "head_count": len(rows),
        "max_nmse": float(worst_nmse["out_nmse"]),
        "max_abs": float(worst_abs["out_max_abs"]),
        "worst_ih": int(worst_nmse["ih"]),
        "worst_d": int(worst_nmse["out_worst_d"]),
    }


def format_float(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl-csv", required=True, type=Path)
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--packed-root", required=True, type=Path)
    ap.add_argument("--boundary-min-tokens", required=True, type=int)
    ap.add_argument("--query-block", type=int, default=128)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--csv-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    args = ap.parse_args()

    if args.query_block <= 0:
        raise SystemExit("--query-block must be positive")
    rows = sorted(read_kl_rows(args.kl_csv), key=lambda row: float(row["kld"]), reverse=True)[: args.top_n]

    out_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, 1):
        capture_tokens, call_index, iq = capture_location(int(row["logit_pos"]), args.boundary_min_tokens, args.query_block)
        raw = read_replay_rows(args.raw_root, call_index, iq)
        packed = read_replay_rows(args.packed_root, call_index, iq)
        out_rows.append({
            "rank": rank,
            "kld": row["kld"],
            "p_diff": row["p_diff"],
            "logit_pos": row["logit_pos"],
            "target_pos": row["target_pos"],
            "target_token": row["target_token"],
            "capture_tokens": capture_tokens,
            "call_index": call_index,
            "iq": iq,
            "raw_heads": raw["head_count"],
            "raw_max_nmse": raw["max_nmse"],
            "raw_max_abs": raw["max_abs"],
            "raw_worst_ih": raw["worst_ih"],
            "raw_worst_d": raw["worst_d"],
            "packed_heads": packed["head_count"],
            "packed_max_nmse": packed["max_nmse"],
            "packed_max_abs": packed["max_abs"],
            "packed_worst_ih": packed["worst_ih"],
            "packed_worst_d": packed["worst_d"],
        })

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
            writer.writeheader()
            writer.writerows(out_rows)

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# KL Boundary Replay Join",
            "",
            f"- KL CSV: `{args.kl_csv}`",
            f"- raw root: `{args.raw_root}`",
            f"- packed root: `{args.packed_root}`",
            f"- boundary_min_tokens={args.boundary_min_tokens}, query_block={args.query_block}",
            "",
            "| rank | KLD | p_diff | logit | target | call | iq | raw max NMSE | raw max abs | packed max NMSE | packed max abs | packed worst head |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in out_rows:
            lines.append(
                f"| {row['rank']} | {row['kld']:.6f} | {row['p_diff']:.6f} | "
                f"{row['logit_pos']} | {row['target_pos']} | {row['call_index']} | {row['iq']} | "
                f"{format_float(row['raw_max_nmse'])} | {format_float(row['raw_max_abs'])} | "
                f"{format_float(row['packed_max_nmse'])} | {format_float(row['packed_max_abs'])} | "
                f"{row['packed_worst_ih']} |"
            )
        args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.csv_out and not args.md_out:
        for row in out_rows:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
