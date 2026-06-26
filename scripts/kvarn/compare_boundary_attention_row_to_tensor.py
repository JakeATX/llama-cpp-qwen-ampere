#!/usr/bin/env python3
"""Compare one captured KVarN attention boundary row to a dumped graph tensor row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_tensor_timeline_rows import build_timeline
from compare_tensor_timeline_rows import load_occurrences
from compare_tensor_timeline_rows import nmse
from compare_tensor_timeline_rows import select_row
from replay_f16_truth_boundary import body_records
from replay_f16_truth_boundary import load_raw_body
from replay_f16_truth_boundary import resolve_boundary
from replay_f16_truth_boundary import softmax_rows
from replay_mixed_attn_boundary import read_full_mask


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return 1.0 if denom == 0.0 else float(np.dot(af, bf) / denom)


def stats(ref: np.ndarray, got: np.ndarray) -> dict[str, float | int]:
    if ref.shape != got.shape:
        raise SystemExit(f"shape mismatch: {ref.shape} != {got.shape}")
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    worst = int(np.argmax(diff)) if diff.size else 0
    return {
        "n_values": int(ref.size),
        "nmse": float(nmse(ref, got)),
        "cosine": cosine(ref, got),
        "max_abs": float(diff[worst]) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "worst_d": worst,
    }


def format_stats(label: str, row: dict[str, float | int]) -> str:
    return (
        f"{label}: n={row['n_values']} nmse={float(row['nmse']):.6e} "
        f"cosine={float(row['cosine']):.9f} max_abs={float(row['max_abs']):.6e} "
        f"mean_abs={float(row['mean_abs']):.6e} worst_d={int(row['worst_d'])}"
    )


def full_truth_row(boundary: Path, meta: dict, body_root: Path, record_set: str, iq: int) -> np.ndarray:
    head_dim = int(meta["head_dim"])
    group_size = int(meta["group_size"])
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    n_pending = int(meta["n_pending"])
    n_tail = int(meta["n_tail"])
    n_tokens = int(meta["n_tokens"])
    n_queries = int(meta["n_queries"])
    n_head = int(meta["n_head"])
    n_gqa = int(meta["n_gqa"])
    selected_ikh = int(meta["selected_ikh"])
    layer = int(meta.get("inferred_layer", -1))
    scale = np.float32(float(meta["scale"]))

    if n_pending != 0:
        raise SystemExit("expected n_pending=0 for this offline full-QO replay")
    if n_tokens != n_sink + n_records * group_size + n_tail:
        raise SystemExit("boundary n_tokens does not match sink/body/tail layout")
    if not (0 <= iq < n_queries):
        raise SystemExit(f"iq={iq} outside boundary n_queries={n_queries}")

    sink_tail_k = np.fromfile(boundary / "sink_tail_k_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, head_dim)
    sink_tail_v = np.fromfile(boundary / "sink_tail_v_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, head_dim)
    records = body_records(body_root, layer, selected_ikh, record_set)
    body_k, body_v = load_raw_body(records, n_records, head_dim, group_size)
    k_all = np.concatenate([sink_tail_k[:n_sink], body_k, sink_tail_k[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)
    v_all = np.concatenate([sink_tail_v[:n_sink], body_v, sink_tail_v[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)

    full_mask = read_full_mask(boundary, meta, n_queries, n_tokens)
    if full_mask is None:
        raise SystemExit("boundary does not contain full_mask.bin")

    expected = n_queries * n_head * head_dim
    full_q = np.fromfile(boundary / "full_q.bin", dtype=np.float32, count=expected).reshape(n_queries, n_head, head_dim)
    full_q_body_path = boundary / "full_q_body.bin"
    full_q_body = (
        np.fromfile(full_q_body_path, dtype=np.float32, count=expected).reshape(n_queries, n_head, head_dim)
        if full_q_body_path.exists() else None
    )

    heads = list(range(selected_ikh * n_gqa, min(n_head, (selected_ikh + 1) * n_gqa)))
    q = full_q[iq, heads, :].astype(np.float32)
    q_body = q if full_q_body is None else full_q_body[iq, heads, :].astype(np.float32)

    n_body = n_records * group_size
    scores = np.zeros((len(heads), n_tokens), dtype=np.float32)
    k_t = k_all.astype(np.float32).T
    if n_sink:
        scores[:, :n_sink] = (q @ k_t[:, :n_sink]).astype(np.float32)
    if n_body:
        scores[:, n_sink:n_sink + n_body] = (q_body @ k_t[:, n_sink:n_sink + n_body]).astype(np.float32)
    tail0 = n_sink + n_body
    if tail0 < n_tokens:
        scores[:, tail0:] = (q @ k_t[:, tail0:]).astype(np.float32)
    scores = (scores * scale + full_mask[iq].reshape(1, -1)).astype(np.float32)
    return (softmax_rows(scores) @ v_all).astype(np.float32).reshape(-1)


def full_out_row(boundary: Path, meta: dict, iq: int) -> np.ndarray:
    head_dim = int(meta["head_dim"])
    n_queries = int(meta["n_queries"])
    n_head = int(meta["n_head"])
    expected = n_queries * n_head * head_dim
    return np.fromfile(boundary / "full_out.bin", dtype=np.float32, count=expected).reshape(n_queries, n_head, head_dim)[iq].reshape(-1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--body-records", required=True, type=Path)
    ap.add_argument("--record-set", choices=["earliest", "latest"], default="latest")
    ap.add_argument("--tensor-dump", required=True, type=Path)
    ap.add_argument("--tensor-name", default="kqv_out-5")
    ap.add_argument("--context-size", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--iq", type=int, required=True)
    ap.add_argument("--skip-leading-short", action="store_true")
    args = ap.parse_args()

    boundary = resolve_boundary(args.boundary)
    meta = json.loads((boundary / "boundary.json").read_text())
    tl = build_timeline(load_occurrences(args.tensor_dump, args.tensor_name, args.skip_leading_short), args.context_size)
    native, native_meta, native_local = select_row(tl, args.chunk, args.pos)
    truth = full_truth_row(boundary, meta, args.body_records, args.record_set, args.iq)
    actual = full_out_row(boundary, meta, args.iq)

    print(
        f"boundary={boundary} tensor_dump={args.tensor_dump} tensor={args.tensor_name} "
        f"chunk={args.chunk} pos={args.pos} iq={args.iq} "
        f"tensor_index={int(native_meta['index'])} tensor_local={native_local} "
        f"call_index={int(meta.get('call_index', -1))} n_tokens={int(meta['n_tokens'])}"
    )
    print(format_stats("truth_vs_boundary_full_out", stats(truth, actual)))
    print(format_stats("truth_vs_native_tensor", stats(truth, native)))
    print(format_stats("boundary_full_out_vs_native_tensor", stats(actual, native)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
