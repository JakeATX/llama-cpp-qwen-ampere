#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

from replay_mixed_attn_boundary import read_full_mask, read_mask, reconstruct_kv


NEG_MASK_CUTOFF = -1.0e20


def nmse_rows(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    rr = ref.astype(np.float64)
    gg = got.astype(np.float64)
    diff = rr - gg
    num = np.sum(diff * diff, axis=1)
    denom = np.sum(rr * rr, axis=1)
    out = np.empty(ref.shape[0], dtype=np.float64)
    zero = denom == 0.0
    out[~zero] = num[~zero] / denom[~zero]
    out[zero] = np.where(num[zero] == 0.0, 0.0, np.inf)
    return out


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    x = scores.astype(np.float32)
    x = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x).astype(np.float32)
    return (p / np.sum(p, axis=1, keepdims=True, dtype=np.float32)).astype(np.float32)


def discover_boundaries(path: Path) -> list[Path]:
    if (path / "boundary.json").exists():
        return [path]
    candidates = sorted(p.parent for p in path.rglob("boundary.json"))
    boundaries = [p for p in candidates if (p / "full_q.bin").exists() and (p / "full_out.bin").exists()]
    if not boundaries:
        if candidates:
            raise SystemExit(f"found {len(candidates)} boundary dumps under {path}, but none have full_q.bin/full_out.bin")
        raise SystemExit(f"boundary.json not found under {path}")
    return boundaries


def load_full_qo(root: Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    head_dim = int(meta["head_dim"])
    n_queries = int(meta["n_queries"])
    n_head = int(meta["n_head"])
    expected = n_queries * n_head * head_dim

    q_path = root / "full_q.bin"
    out_path = root / "full_out.bin"
    if not q_path.exists() or not out_path.exists():
        raise SystemExit(f"{root} is not a full-QO dump; expected full_q.bin and full_out.bin")

    q = np.fromfile(q_path, dtype=np.float32)
    actual = np.fromfile(out_path, dtype=np.float32)
    if q.size != expected:
        raise SystemExit(f"{q_path} has {q.size} floats, expected {expected}")
    if actual.size != expected:
        raise SystemExit(f"{out_path} has {actual.size} floats, expected {expected}")
    return q.reshape(n_queries, n_head, head_dim), actual.reshape(n_queries, n_head, head_dim)


def infer_query_masker(root: Path, meta: dict) -> tuple[Callable[[int], np.ndarray], str]:
    n_tokens = int(meta["n_tokens"])
    n_queries = int(meta["n_queries"])
    selected_iq = int(meta["selected_iq"])
    full_mask = read_full_mask(root, meta, n_queries, n_tokens)
    if full_mask is not None:
        return lambda iq: full_mask[iq], "full-mask"

    selected = read_mask(root, meta, n_tokens)
    mask_type = int(meta.get("mask_type", 0))

    if mask_type == 0 or not (root / "mask.bin").exists():
        zero = np.zeros(n_tokens, dtype=np.float32)
        return lambda _iq: zero, "none"

    finite = selected > NEG_MASK_CUTOFF
    finite_idx = np.flatnonzero(finite)
    if finite_idx.size == 0:
        raise SystemExit(f"{root} selected mask row has no attendable tokens")

    prefix_len = int(finite_idx.size)
    is_prefix = np.array_equal(finite_idx, np.arange(prefix_len))
    finite_values = selected[:prefix_len]
    prefix_values_are_zero = bool(np.all(finite_values == 0.0))
    suffix_is_masked = bool(np.all(selected[prefix_len:] <= NEG_MASK_CUTOFF))

    if is_prefix and prefix_values_are_zero and suffix_is_masked:
        base_tokens = prefix_len - selected_iq - 1
        if base_tokens < 0:
            raise SystemExit(
                f"{root} selected mask prefix length {prefix_len} is incompatible with selected_iq={selected_iq}"
            )

        def causal_mask(iq: int) -> np.ndarray:
            allow = min(n_tokens, max(0, base_tokens + iq + 1))
            out = np.full(n_tokens, -np.inf, dtype=np.float32)
            out[:allow] = 0.0
            return out

        return causal_mask, f"causal-prefix(base_tokens={base_tokens})"

    if finite_idx.size == n_tokens and np.all(selected == 0.0):
        if selected_iq == n_queries - 1 and n_tokens >= n_queries:
            base_tokens = n_tokens - n_queries

            def last_row_causal_mask(iq: int) -> np.ndarray:
                allow = min(n_tokens, max(0, base_tokens + iq + 1))
                out = np.full(n_tokens, -np.inf, dtype=np.float32)
                out[:allow] = 0.0
                return out

            return last_row_causal_mask, f"causal-prefix-last-row(base_tokens={base_tokens})"

        zero = np.zeros(n_tokens, dtype=np.float32)
        return lambda _iq: zero, "all-visible"

    raise SystemExit(
        f"{root} mask row is not a supported full-QO pattern; only zero/all-visible and causal prefix masks are inferred"
    )


def output_paths(input_path: Path, csv_arg: str | None, json_arg: str | None) -> tuple[Path, Path]:
    if csv_arg:
        csv_path = Path(csv_arg)
    else:
        csv_path = input_path / "full_qo_replay.csv"
    if json_arg:
        json_path = Path(json_arg)
    else:
        json_path = input_path / "full_qo_replay.json"
    return csv_path, json_path


def replay_boundary(root: Path, heads_mode: str) -> tuple[list[dict], dict]:
    meta = json.loads((root / "boundary.json").read_text())
    head_dim = int(meta["head_dim"])
    n_queries = int(meta["n_queries"])
    n_head = int(meta["n_head"])
    n_gqa = int(meta["n_gqa"])
    selected_ikh = int(meta["selected_ikh"])
    selected_ih = int(meta["selected_ih"])
    scale = np.float32(float(meta["scale"]))

    full_q, full_out = load_full_qo(root, meta)
    k_all, v_all = reconstruct_kv(root, meta)
    if k_all.shape != (int(meta["n_tokens"]), head_dim):
        raise SystemExit(f"{root} reconstructed K shape {k_all.shape} does not match metadata")
    if v_all.shape != (int(meta["n_tokens"]), head_dim):
        raise SystemExit(f"{root} reconstructed V shape {v_all.shape} does not match metadata")

    if heads_mode == "selected":
        heads = [selected_ih]
    else:
        first = selected_ikh * n_gqa
        heads = list(range(first, min(n_head, first + n_gqa)))
        if not heads:
            raise SystemExit(f"{root} selected_ikh={selected_ikh} maps to no query heads")
        if heads_mode == "all" and len(heads) != n_head:
            raise SystemExit(
                f"{root} only contains KV data for selected_ikh={selected_ikh}; "
                f"replayable query heads are {heads[0]}..{heads[-1]}, not all {n_head} heads"
            )

    mask_for_iq, mask_mode = infer_query_masker(root, meta)
    rows: list[dict] = []
    worst_row: dict | None = None

    k_t = k_all.astype(np.float32).T
    v = v_all.astype(np.float32)

    for iq in range(n_queries):
        q = full_q[iq, heads, :].astype(np.float32)
        scores = (q @ k_t).astype(np.float32) * scale
        scores = (scores + mask_for_iq(iq).reshape(1, -1)).astype(np.float32)
        probs = softmax_rows(scores)
        replay_out = (probs @ v).astype(np.float32)
        actual_out = full_out[iq, heads, :].astype(np.float32)

        diff = np.abs(replay_out.astype(np.float64) - actual_out.astype(np.float64))
        worst_d = np.argmax(diff, axis=1).astype(np.int64)
        max_abs = diff[np.arange(len(heads)), worst_d]
        out_nmse = nmse_rows(replay_out, actual_out)

        for row_i, ih in enumerate(heads):
            row = {
                "boundary": str(root),
                "layer": int(meta.get("inferred_layer", -1)),
                "call_index": int(meta.get("call_index", -1)),
                "iq": iq,
                "ih": ih,
                "ikh": ih // n_gqa,
                "n_tokens": int(meta["n_tokens"]),
                "n_records": int(meta["n_records"]),
                "mask_mode": mask_mode,
                "out_nmse": float(out_nmse[row_i]),
                "out_max_abs": float(max_abs[row_i]),
                "out_worst_d": int(worst_d[row_i]),
            }
            rows.append(row)
            if worst_row is None or row["out_nmse"] > worst_row["out_nmse"]:
                worst_row = row

    summary = {
        "boundary": str(root),
        "layer": int(meta.get("inferred_layer", -1)),
        "call_index": int(meta.get("call_index", -1)),
        "n_queries": n_queries,
        "n_head": n_head,
        "n_gqa": n_gqa,
        "selected_ikh": selected_ikh,
        "replayed_heads": heads,
        "rows": len(rows),
        "mask_mode": mask_mode,
        "max_out_nmse": max((float(row["out_nmse"]) for row in rows), default=0.0),
        "max_out_abs": max((float(row["out_max_abs"]) for row in rows), default=0.0),
        "worst_row": worst_row,
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replay every full-QO query/head row supported by a KVarN boundary dump."
    )
    ap.add_argument("--dump", required=True, help="Boundary dump dir or parent containing boundary.json children")
    ap.add_argument(
        "--heads",
        choices=["selected-kv", "selected", "all"],
        default="selected-kv",
        help="Replay query heads backed by the dumped KV head, only the selected head, or require all heads",
    )
    ap.add_argument("--csv", default=None, help="Output CSV path; default is <dump>/full_qo_replay.csv")
    ap.add_argument("--json", default=None, help="Output JSON summary path; default is <dump>/full_qo_replay.json")
    ap.add_argument("--max-out-nmse", type=float, default=None)
    args = ap.parse_args()

    input_path = Path(args.dump)
    boundaries = discover_boundaries(input_path)
    csv_path, json_path = output_paths(input_path, args.csv, args.json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    boundary_summaries: list[dict] = []
    for boundary in boundaries:
        rows, summary = replay_boundary(boundary, args.heads)
        all_rows.extend(rows)
        boundary_summaries.append(summary)
        print(
            "Full-QO replay: "
            f"path={boundary} layer={summary['layer']} call={summary['call_index']} "
            f"rows={summary['rows']} heads={summary['replayed_heads']} mask={summary['mask_mode']} "
            f"max_out_nmse={summary['max_out_nmse']:.6e} max_out_abs={summary['max_out_abs']:.6e}"
        )

    fieldnames = [
        "boundary",
        "layer",
        "call_index",
        "iq",
        "ih",
        "ikh",
        "n_tokens",
        "n_records",
        "mask_mode",
        "out_nmse",
        "out_max_abs",
        "out_worst_d",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    worst_row = max(all_rows, key=lambda row: row["out_nmse"], default=None)
    summary = {
        "dump": str(input_path),
        "csv": str(csv_path),
        "json": str(json_path),
        "boundary_count": len(boundaries),
        "row_count": len(all_rows),
        "max_out_nmse": max((float(row["out_nmse"]) for row in all_rows), default=0.0),
        "max_out_abs": max((float(row["out_max_abs"]) for row in all_rows), default=0.0),
        "worst_row": worst_row,
        "boundaries": boundary_summaries,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {len(all_rows)} rows to {csv_path}")
    print(f"Wrote summary to {json_path}")

    if args.max_out_nmse is not None and summary["max_out_nmse"] > args.max_out_nmse:
        raise SystemExit(f"out NMSE {summary['max_out_nmse']:.6e} exceeds {args.max_out_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
