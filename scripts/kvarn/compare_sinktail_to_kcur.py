#!/usr/bin/env python3
"""Compare a KVarN mixed-attention sink/tail dump to independent Kcur/Vcur dumps.

The mixed-attention boundary replay proves that the CUDA attention kernel
consumes the sink/tail cache consistently. This script proves the earlier
boundary: the sink/tail cache must contain the expected source K/V tokens from
the model graph, in the expected paper-frame orientation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_tensor_dumps import DTYPES


def load_timeline(dump_dir: Path, name: str, context_size: int) -> list[dict]:
    out: list[dict] = []
    chunk = 0
    pos = 0
    for meta_path in sorted(dump_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("name") != name:
            continue
        ne = [int(x) for x in meta["ne"]]
        if len(ne) < 3 or ne[2] <= 1:
            continue
        typ = meta.get("type")
        if typ not in DTYPES:
            raise SystemExit(f"unsupported tensor type {typ!r} for {name}")
        data = np.fromfile(dump_dir / meta["bin"], dtype=DTYPES[typ]).astype(np.float32)
        expected = int(meta["n_bytes"]) // np.dtype(DTYPES[typ]).itemsize
        if data.size != expected:
            raise SystemExit(f"{meta['bin']} has {data.size} values, expected {expected}")
        arr = data.reshape(tuple(ne), order="F").reshape(ne[0] * ne[1], ne[2], order="F")
        n_tokens = int(arr.shape[1])
        if pos + n_tokens > context_size:
            chunk += 1
            pos = 0
        out.append({"chunk": chunk, "pos0": pos, "pos1": pos + n_tokens, "arr": arr, "meta": meta})
        pos += n_tokens
        if pos == context_size:
            chunk += 1
            pos = 0
    if not out:
        raise SystemExit(f"missing tensor dump {name!r} under {dump_dir}")
    return out


def slice_timeline(timeline: list[dict], chunk: int, pos0: int, pos1: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    cur = pos0
    for item in timeline:
        if int(item["chunk"]) != chunk:
            continue
        lo = max(pos0, int(item["pos0"]))
        hi = min(pos1, int(item["pos1"]))
        if hi <= lo:
            continue
        if lo != cur:
            raise SystemExit(f"timeline gap for chunk={chunk} span=[{pos0},{pos1}) at {cur}")
        rel0 = lo - int(item["pos0"])
        rel1 = hi - int(item["pos0"])
        pieces.append(item["arr"][:, rel0:rel1])
        cur = hi
    if cur != pos1 or not pieces:
        raise SystemExit(f"could not cover chunk={chunk} span=[{pos0},{pos1}) from tensor timeline")
    return np.concatenate(pieces, axis=1).astype(np.float32)


def hadamard_rows(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    n = y.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise SystemExit(f"Hadamard dimension must be a power of two, got {n}")
    step = 1
    while step < n:
        y2 = y.reshape(-1, n)
        for base in range(0, n, 2 * step):
            a = y2[:, base:base + step].copy()
            b = y2[:, base + step:base + 2 * step].copy()
            y2[:, base:base + step] = a + b
            y2[:, base + step:base + 2 * step] = a - b
        step <<= 1
    y *= np.float32(1.0 / np.sqrt(float(n)))
    return y


def nmse(ref: np.ndarray, got: np.ndarray) -> float:
    r = ref.astype(np.float64).ravel()
    g = got.astype(np.float64).ravel()
    d = g - r
    return float(np.dot(d, d) / (np.dot(r, r) + 1.0e-30))


def max_abs(ref: np.ndarray, got: np.ndarray) -> tuple[float, int]:
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64)).ravel()
    idx = int(np.argmax(diff)) if diff.size else 0
    return float(diff[idx]) if diff.size else 0.0, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--tensor-dump", required=True, type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--k-tensor-name", default=None)
    ap.add_argument("--v-tensor-name", default=None)
    ap.add_argument("--context-size", type=int, default=4096)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--paper-frame", action="store_true")
    ap.add_argument("--negative-control", choices=["none", "offset"], default="none")
    ap.add_argument("--max-nmse", type=float, default=None)
    args = ap.parse_args()

    boundary = args.boundary
    if not (boundary / "boundary.json").exists():
        children = sorted(p for p in boundary.iterdir() if (p / "boundary.json").exists())
        if len(children) != 1:
            raise SystemExit(f"expected one boundary under {boundary}, found {len(children)}")
        boundary = children[0]
    meta = json.loads((boundary / "boundary.json").read_text())

    head_dim = int(meta["head_dim"])
    n_sink = int(meta["n_sink"])
    n_tail = int(meta["n_tail"])
    tail_start = int(meta["tail_start"])
    n_tokens = n_sink + n_tail
    if int(meta["n_records"]) != 0 or int(meta["n_pending"]) != 0:
        raise SystemExit("sink/tail source compare currently expects a no-body, no-pending boundary")
    if int(meta["n_tokens"]) != n_tokens:
        raise SystemExit("boundary token count is not sink+tail")

    k_name = args.k_tensor_name or f"Kcur-{args.layer}"
    v_name = args.v_tensor_name or f"Vcur-{args.layer}"
    k_timeline = load_timeline(args.tensor_dump, k_name, args.context_size)
    v_timeline = load_timeline(args.tensor_dump, v_name, args.context_size)

    source_k = slice_timeline(k_timeline, args.chunk_index, 0, args.context_size)
    source_v = slice_timeline(v_timeline, args.chunk_index, 0, args.context_size)
    if source_k.shape[0] != head_dim or source_v.shape[0] != head_dim:
        raise SystemExit(f"source shape mismatch: K={source_k.shape} V={source_v.shape} head_dim={head_dim}")
    if args.paper_frame:
        source_k = hadamard_rows(source_k.T).T
        source_v = hadamard_rows(source_v.T).T

    tail_pos0 = 0 if args.negative_control == "offset" else n_sink
    expected_k = np.concatenate([source_k[:, :n_sink].T, source_k[:, tail_pos0:tail_pos0 + n_tail].T], axis=0)
    expected_v = np.concatenate([source_v[:, :n_sink].T, source_v[:, tail_pos0:tail_pos0 + n_tail].T], axis=0)

    got_k = np.fromfile(boundary / "sink_tail_k_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_tokens, head_dim)
    got_v = np.fromfile(boundary / "sink_tail_v_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_tokens, head_dim)

    rows = []
    for label, ref, got in (("K", expected_k, got_k), ("V", expected_v, got_v)):
        err = nmse(ref, got)
        ma, idx = max_abs(ref, got)
        rows.append((label, err, ma, idx // head_dim, idx % head_dim))
        print(f"Sink/tail source {label}: nmse={err:.6e} max_abs={ma:.6e} token={idx // head_dim} d={idx % head_dim}")

    worst = max(row[1] for row in rows)
    print(
        "Sink/tail source oracle: "
        f"boundary={boundary} layer={args.layer} chunk={args.chunk_index} "
        f"n_sink={n_sink} n_tail={n_tail} tail_start={tail_start} worst_nmse={worst:.6e}"
    )
    if args.max_nmse is not None and worst > args.max_nmse:
        raise SystemExit(f"sink/tail source NMSE {worst:.6e} exceeds {args.max_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
