#!/usr/bin/env python3
"""Compare active KVarN sink/tail/pending cache rows to Kcur/Vcur dumps.

Body-record source comparison is handled by compare_body_records_to_kcur.py.
This script covers the uncompressed rows that are still read by mixed attention
when n_records > 0: sink rows, pending rows, and chronological tail rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_body_records_to_kcur import hadamard_rows, load_tensor_occurrences, slice_timeline, tensor_timeline


def nmse(ref: np.ndarray, got: np.ndarray) -> float:
    r = ref.astype(np.float64).ravel()
    g = got.astype(np.float64).ravel()
    d = g - r
    denom = float(np.dot(r, r))
    num = float(np.dot(d, d))
    if denom == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / denom


def max_abs(ref: np.ndarray, got: np.ndarray) -> tuple[float, int]:
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64)).ravel()
    idx = int(np.argmax(diff)) if diff.size else 0
    return (float(diff[idx]) if diff.size else 0.0, idx)


def resolve_boundary(path: Path) -> Path:
    if (path / "boundary.json").exists():
        return path
    children = sorted(p for p in path.iterdir() if (p / "boundary.json").exists())
    if len(children) != 1:
        raise SystemExit(f"expected one boundary under {path}, found {len(children)}")
    return children[0]


def load_rows(path: Path, dtype: np.dtype, rows: int, dim: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    expected = rows * dim
    if data.size != expected:
        raise SystemExit(f"{path} has {data.size} values, expected {expected}")
    return data.astype(np.float32).reshape(rows, dim)


def select_chunk(arg: str, max_chunk: int) -> int:
    if arg in ("auto", "latest"):
        return max_chunk
    if arg == "earliest":
        return 0
    return int(arg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--tensor-dump", required=True, type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, default=None, help="KV head. Defaults to boundary selected_ikh.")
    ap.add_argument("--context-size", type=int, default=4096)
    ap.add_argument("--chunk-index", default="auto", help="auto, earliest, latest, or explicit zero-based chunk")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--k-tensor-name", default=None)
    ap.add_argument("--v-tensor-name", default=None)
    ap.add_argument("--paper-frame", action="store_true")
    ap.add_argument("--skip-leading-short", action="store_true")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--max-nmse", type=float, default=None)
    args = ap.parse_args()

    boundary = resolve_boundary(args.boundary)
    meta = json.loads((boundary / "boundary.json").read_text())
    head_dim = int(meta["head_dim"])
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    n_pending = int(meta["n_pending"])
    n_tail = int(meta["n_tail"])
    group_size = int(meta["group_size"])
    if group_size != args.group_size:
        raise SystemExit(f"boundary group_size {group_size} != expected {args.group_size}")
    if int(meta["n_tokens"]) != n_sink + n_records * group_size + n_pending + n_tail:
        raise SystemExit("boundary n_tokens does not match sink/body/pending/tail counts")

    k_name = args.k_tensor_name or f"Kcur-{args.layer}"
    v_name = args.v_tensor_name or f"Vcur-{args.layer}"
    k_occurrences = load_tensor_occurrences(args.tensor_dump, k_name)
    v_occurrences = load_tensor_occurrences(args.tensor_dump, v_name)
    if args.skip_leading_short:
        if k_occurrences and int(k_occurrences[0][1].shape[2]) < args.group_size:
            k_occurrences = k_occurrences[1:]
        if v_occurrences and int(v_occurrences[0][1].shape[2]) < args.group_size:
            v_occurrences = v_occurrences[1:]
    k_timeline = tensor_timeline(k_occurrences, args.context_size)
    v_timeline = tensor_timeline(v_occurrences, args.context_size)
    if len(k_timeline) != len(v_timeline):
        raise SystemExit(f"Kcur/Vcur occurrence mismatch: {len(k_timeline)} vs {len(v_timeline)}")
    for k_item, v_item in zip(k_timeline, v_timeline):
        if k_item["chunk"] != v_item["chunk"] or k_item["pos0"] != v_item["pos0"] or k_item["pos1"] != v_item["pos1"]:
            raise SystemExit("Kcur/Vcur token timeline mismatch")
        if k_item["arr"].shape != v_item["arr"].shape:
            raise SystemExit(f"Kcur/Vcur shape mismatch: {k_item['arr'].shape} vs {v_item['arr'].shape}")

    if int(k_timeline[0]["arr"].shape[0]) != head_dim:
        raise SystemExit(f"Kcur head_dim {k_timeline[0]['arr'].shape[0]} != boundary head_dim {head_dim}")
    max_chunk = max(int(item["chunk"]) for item in k_timeline)
    chunk_index = select_chunk(args.chunk_index, max_chunk)
    head = int(meta["selected_ikh"]) if args.head is None else args.head
    n_heads = int(k_timeline[0]["arr"].shape[1])
    if head >= n_heads:
        raise SystemExit(f"head {head} out of range for Kcur/Vcur with {n_heads} heads")

    sink_tail_k = load_rows(boundary / "sink_tail_k_f16.bin", np.float16, n_sink + n_tail, head_dim)
    sink_tail_v = load_rows(boundary / "sink_tail_v_f16.bin", np.float16, n_sink + n_tail, head_dim)
    pending_k = load_rows(boundary / "pending_k.bin", np.float32, n_pending, head_dim)
    pending_v = load_rows(boundary / "pending_v.bin", np.float32, n_pending, head_dim)

    body0 = n_sink
    pending0 = body0 + n_records * group_size
    tail0 = pending0 + n_pending
    spans = [
        ("sink", 0, n_sink, sink_tail_k[:n_sink], sink_tail_v[:n_sink]),
        ("pending", pending0, tail0, pending_k, pending_v),
        ("tail", tail0, tail0 + n_tail, sink_tail_k[n_sink:n_sink + n_tail], sink_tail_v[n_sink:n_sink + n_tail]),
    ]

    rows: list[dict] = []
    worst = 0.0
    for label, pos0, pos1, got_k, got_v in spans:
        if pos1 == pos0:
            continue
        k_expected_dg, k_sources = slice_timeline(k_timeline, chunk_index, head, pos0, pos1)
        v_expected_dg, v_sources = slice_timeline(v_timeline, chunk_index, head, pos0, pos1)
        expected_k = k_expected_dg.T.astype(np.float32)
        expected_v = v_expected_dg.T.astype(np.float32)
        if args.paper_frame:
            expected_k = hadamard_rows(expected_k)
            expected_v = hadamard_rows(expected_v)

        for kv, ref, got, sources in (
            ("K", expected_k, got_k, k_sources),
            ("V", expected_v, got_v, v_sources),
        ):
            err = nmse(ref, got)
            ma, idx = max_abs(ref, got)
            worst = max(worst, err)
            row = {
                "part": label,
                "kv": kv,
                "chunk": chunk_index,
                "head": head,
                "pos0": pos0,
                "pos1": pos1,
                "nmse": err,
                "max_abs": ma,
                "token": idx // head_dim,
                "d": idx % head_dim,
                "sources": sources,
            }
            rows.append(row)
            print(
                f"Active cache {label} {kv}: nmse={err:.6e} max_abs={ma:.6e} "
                f"token={row['token']} d={row['d']} pos=[{pos0},{pos1})")

    if not rows:
        raise SystemExit("no active cache rows selected")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(
        "Active cache source oracle: "
        f"boundary={boundary} layer={args.layer} head={head} chunk={chunk_index} "
        f"n_sink={n_sink} n_records={n_records} n_pending={n_pending} n_tail={n_tail} worst_nmse={worst:.6e}"
    )
    if args.max_nmse is not None and worst > args.max_nmse:
        raise SystemExit(f"active cache source NMSE {worst:.6e} exceeds {args.max_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
