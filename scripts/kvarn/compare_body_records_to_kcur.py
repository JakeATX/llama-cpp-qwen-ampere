#!/usr/bin/env python3
"""Compare KVarN body-record source dumps to independently dumped Kcur/Vcur.

This checks the graph/store boundary that packed-vs-split and body-local replay
cannot prove: the body store must consume the same per-token K/V values that the
normal graph produced for the corresponding token positions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_tensor_dumps import DTYPES


def load_tensor_occurrences(dump_dir: Path, name: str) -> list[tuple[dict, np.ndarray]]:
    out: list[tuple[dict, np.ndarray]] = []
    for meta_path in sorted(dump_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("name") != name:
            continue
        ne = [int(x) for x in meta["ne"]]
        # Ignore pre-reshape projection matrices such as [512, 512, 1, 1].
        # Cache source tensors are [head_dim, n_head_kv, n_tokens, 1].
        if len(ne) < 3 or ne[2] <= 1:
            continue
        typ = meta.get("type")
        if typ not in DTYPES:
            raise SystemExit(f"unsupported tensor type {typ!r} for {name}")
        data = np.fromfile(dump_dir / meta["bin"], dtype=DTYPES[typ]).astype(np.float32)
        expected = int(meta["n_bytes"]) // np.dtype(DTYPES[typ]).itemsize
        if data.size != expected:
            raise SystemExit(f"{meta['bin']} has {data.size} values, expected {expected}")
        meta["_json"] = meta_path
        out.append((meta, data.reshape(tuple(ne), order="F")))
    if not out:
        raise SystemExit(f"missing cache-source tensor dump {name!r} under {dump_dir}")
    return out


def tensor_timeline(occurrences: list[tuple[dict, np.ndarray]], context_size: int) -> list[dict]:
    timeline: list[dict] = []
    chunk = 0
    pos = 0
    for meta, arr in occurrences:
        n_tokens = int(arr.shape[2])
        if pos + n_tokens > context_size:
            chunk += 1
            pos = 0
        timeline.append({
            "chunk": chunk,
            "pos0": pos,
            "pos1": pos + n_tokens,
            "meta": meta,
            "arr": arr,
        })
        pos += n_tokens
        if pos == context_size:
            chunk += 1
            pos = 0
    return timeline


def slice_timeline(timeline: list[dict], chunk: int, head: int, pos0: int, pos1: int) -> tuple[np.ndarray, str]:
    pieces: list[np.ndarray] = []
    sources: list[str] = []
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
        piece = item["arr"][:, head, rel0:rel1]
        if piece.ndim > 2:
            piece = np.squeeze(piece, axis=tuple(range(2, piece.ndim)))
        pieces.append(piece)
        sources.append(str(item["meta"]["_json"]))
        cur = hi
    if cur != pos1 or not pieces:
        raise SystemExit(f"could not cover chunk={chunk} span=[{pos0},{pos1}) from tensor timeline")
    return np.concatenate(pieces, axis=1).astype(np.float32), ";".join(sources)


def hadamard_rows(x: np.ndarray) -> np.ndarray:
    """Apply natural-order normalized Hadamard to the last dimension."""
    y = x.astype(np.float32, copy=True)
    n = y.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise SystemExit(f"Hadamard dimension must be a power of two, got {n}")
    step = 1
    while step < n:
        y2 = y.reshape(-1, n)
        for base in range(0, n, 2*step):
            a = y2[:, base:base + step].copy()
            b = y2[:, base + step:base + 2*step].copy()
            y2[:, base:base + step] = a + b
            y2[:, base + step:base + 2*step] = a - b
        step <<= 1
    y *= np.float32(1.0 / np.sqrt(float(n)))
    return y


def nmse(ref: np.ndarray, got: np.ndarray) -> float:
    r = ref.astype(np.float64).ravel()
    g = got.astype(np.float64).ravel()
    d = g - r
    denom = float(np.dot(r, r))
    num = float(np.dot(d, d))
    if denom == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / denom


def max_abs(ref: np.ndarray, got: np.ndarray) -> float:
    if ref.size == 0:
        return 0.0
    return float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))


def canonical_record(meta: dict) -> int:
    record = int(meta["record"])
    record0 = int(meta.get("record0", record))
    src_layout = int(meta.get("src_layout", -1))
    # Direct-record batches pass a local record id inside the batch. Pending
    # paths already pass the canonical record id.
    if src_layout == 1:
        return record0 + record
    return record


def canonical_body_records(root: Path, layer: int, head: int, record_set: str) -> dict[int, Path]:
    selected: dict[int, tuple[int, Path]] = {}
    for rec_dir in sorted(root.glob("store_*")):
        meta_path = rec_dir / "body_record.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if int(meta.get("layer", -1)) != layer or int(meta.get("head", -1)) != head:
            continue
        record = canonical_record(meta)
        call_index = int(meta.get("call_index", -1))
        if record not in selected:
            selected[record] = (call_index, rec_dir)
            continue
        prev_call, _ = selected[record]
        if record_set == "latest":
            if call_index > prev_call:
                selected[record] = (call_index, rec_dir)
        else:
            if call_index < prev_call:
                selected[record] = (call_index, rec_dir)
    return {record: rec_dir for record, (_, rec_dir) in selected.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-records", required=True, type=Path)
    ap.add_argument("--tensor-dump", required=True, type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--sink-tokens", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--context-size", type=int, default=4096)
    ap.add_argument("--record-set", choices=["earliest", "latest"], default="latest")
    ap.add_argument("--min-record", type=int, default=None,
                    help="Only compare canonical records >= this value.")
    ap.add_argument("--max-record-exclusive", type=int, default=None,
                    help="Only compare canonical records < this value.")
    ap.add_argument("--k-tensor-name", default=None,
                    help="Exact dumped K source tensor name. Defaults to Kcur-<layer>.")
    ap.add_argument("--v-tensor-name", default=None,
                    help="Exact dumped V source tensor name. Defaults to Vcur-<layer>.")
    ap.add_argument("--chunk-index", default="auto", help="auto, earliest, latest, or explicit zero-based chunk")
    ap.add_argument("--skip-leading-short", action="store_true",
                    help="Skip a leading tensor occurrence shorter than --group-size. "
                         "llama-perplexity often emits a 2-token prepass before the "
                         "main fixed-size batches; body records are indexed against "
                         "the main context, not that prepass callback.")
    ap.add_argument("--negative-control", choices=["none", "offset-record", "swap-head", "wrong-chunk", "no-paper-frame"],
                    default="none")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--max-input-nmse", type=float, default=None)
    ap.add_argument("--max-rot-nmse", type=float, default=None)
    args = ap.parse_args()

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

    head_dim = int(k_timeline[0]["arr"].shape[0])
    n_heads = int(k_timeline[0]["arr"].shape[1])
    max_chunk = max(int(item["chunk"]) for item in k_timeline)
    if args.chunk_index in ("auto", "earliest"):
        chunk_index = 0 if args.record_set == "earliest" else max_chunk
    elif args.chunk_index == "latest":
        chunk_index = max_chunk
    else:
        chunk_index = int(args.chunk_index)
    if args.head >= n_heads:
        raise SystemExit(f"head {args.head} out of range for Kcur/Vcur with {n_heads} heads")
    source_head = args.head
    if args.negative_control == "swap-head":
        if n_heads < 2:
            raise SystemExit("swap-head negative control requires at least two KV heads")
        source_head = (args.head + 1) % n_heads
    if args.negative_control == "wrong-chunk":
        if max_chunk < 1:
            raise SystemExit("wrong-chunk negative control requires at least two chunks")
        chunk_index = (chunk_index + 1) % (max_chunk + 1)

    records = canonical_body_records(args.body_records, args.layer, args.head, args.record_set)
    rows = []
    worst_input = 0.0
    worst_rot = 0.0
    for _, rec_dir in sorted(records.items()):
        meta = json.loads((rec_dir / "body_record.json").read_text())
        record = canonical_record(meta)
        if args.min_record is not None and record < args.min_record:
            continue
        if args.max_record_exclusive is not None and record >= args.max_record_exclusive:
            continue
        group = int(meta["group_size"])
        if group != args.group_size:
            raise SystemExit(f"{rec_dir} group_size {group} != expected {args.group_size}")
        if int(meta["head_dim"]) != head_dim:
            raise SystemExit(f"{rec_dir} head_dim {meta['head_dim']} != Kcur head_dim {head_dim}")
        pos0 = args.sink_tokens + record*group
        pos1 = pos0 + group
        if args.negative_control == "offset-record":
            pos0 += group
            pos1 += group

        # Expected K tile is channel-major [d,g]. Expected V tile is token-major [g,d].
        k_expected, k_sources = slice_timeline(k_timeline, chunk_index, source_head, pos0, pos1)
        v_expected_dg, v_sources = slice_timeline(v_timeline, chunk_index, source_head, pos0, pos1)
        v_expected = v_expected_dg.T.astype(np.float32)
        k_input = np.fromfile(rec_dir / "k_tile_input.bin", dtype=np.float32).reshape(head_dim, group)
        v_input = np.fromfile(rec_dir / "v_tile_input.bin", dtype=np.float32).reshape(group, head_dim)
        input_already_rotated = bool(meta.get("input_already_rotated", False))

        paper_frame = bool(meta.get("paper_frame", False)) and args.negative_control != "no-paper-frame"
        paper_mixed_frame = bool(meta.get("paper_mixed_frame", False))
        if paper_frame:
            k_rot_expected = hadamard_rows(k_expected.T).T
            v_rot_expected = v_expected if paper_mixed_frame else hadamard_rows(v_expected)
        else:
            k_rot_expected = k_expected
            v_rot_expected = v_expected
        k_input_expected = k_rot_expected if input_already_rotated else k_expected
        v_input_expected = v_rot_expected if input_already_rotated else v_expected
        k_rot = np.fromfile(rec_dir / "k_rot_or_copy.bin", dtype=np.float32).reshape(head_dim, group)
        v_rot = np.fromfile(rec_dir / "v_rot_or_copy.bin", dtype=np.float32).reshape(group, head_dim)

        row = {
            "record": record,
            "dump_record": int(meta["record"]),
            "record0": int(meta.get("record0", record)),
            "head": int(meta["head"]),
            "source_head": source_head,
            "chunk": chunk_index,
            "negative_control": args.negative_control,
            "src_layout": int(meta.get("src_layout", -1)),
            "input_already_rotated": input_already_rotated,
            "pos0": pos0,
            "pos1": pos1,
            "k_input_nmse": nmse(k_input_expected, k_input),
            "k_input_max_abs": max_abs(k_input_expected, k_input),
            "v_input_nmse": nmse(v_input_expected, v_input),
            "v_input_max_abs": max_abs(v_input_expected, v_input),
            "k_rot_nmse": nmse(k_rot_expected, k_rot),
            "k_rot_max_abs": max_abs(k_rot_expected, k_rot),
            "v_rot_nmse": nmse(v_rot_expected, v_rot),
            "v_rot_max_abs": max_abs(v_rot_expected, v_rot),
            "dir": str(rec_dir),
            "kcur_json": k_sources,
            "vcur_json": v_sources,
        }
        worst_input = max(worst_input, row["k_input_nmse"], row["v_input_nmse"])
        worst_rot = max(worst_rot, row["k_rot_nmse"], row["v_rot_nmse"])
        rows.append(row)

    if not rows:
        raise SystemExit("no matching body records")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    worst = max(rows, key=lambda r: max(r["k_input_nmse"], r["v_input_nmse"], r["k_rot_nmse"], r["v_rot_nmse"]))
    print(
        "Body source oracle: "
        f"records={len(rows)} layer={args.layer} head={args.head} chunk={chunk_index} "
        f"worst_input_nmse={worst_input:.6e} worst_rot_nmse={worst_rot:.6e} "
        f"worst_record={worst['record']} src_layout={worst['src_layout']}"
    )
    failed = False
    if args.max_input_nmse is not None and worst_input > args.max_input_nmse:
        failed = True
    if args.max_rot_nmse is not None and worst_rot > args.max_rot_nmse:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
