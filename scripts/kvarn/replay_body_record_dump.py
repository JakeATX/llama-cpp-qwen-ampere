#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    denom = float(np.sum(aa * aa))
    diff = aa - bb
    if denom == 0.0:
        return 0.0 if float(np.sum(diff * diff)) == 0.0 else float("inf")
    return float(np.sum(diff * diff) / denom)


def unpack_one(buf: np.ndarray, bits: int, index: int) -> int:
    bit_pos = index * bits
    byte_pos = bit_pos >> 3
    shift = bit_pos & 7
    value = int(buf[byte_pos]) >> shift
    if shift + bits > 8:
        value |= int(buf[byte_pos + 1]) << (8 - shift)
    return value & ((1 << bits) - 1)


def hadamard_last(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    d = y.shape[-1]
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Hadamard dimension must be a positive power of two, got {d}")
    step = 1
    while step < d:
        yy = y.reshape(*y.shape[:-1], d // (2 * step), 2, step)
        a = yy[..., 0, :].copy()
        b = yy[..., 1, :].copy()
        yy[..., 0, :] = a + b
        yy[..., 1, :] = a - b
        step *= 2
    y *= np.float32(1.0 / math.sqrt(float(d)))
    return y


def dequant_k(body: np.ndarray, scales: np.ndarray, head_dim: int, group_size: int, bits: int) -> np.ndarray:
    out = np.zeros((head_dim, group_size), dtype=np.float32)
    s_col = scales[:head_dim]
    zp = scales[head_dim:2 * head_dim]
    s_row = scales[2 * head_dim:2 * head_dim + group_size]
    for d in range(head_dim):
        for g in range(group_size):
            q = unpack_one(body, bits, d * group_size + g)
            out[d, g] = np.float32((np.float32(q) * s_col[d] + zp[d]) * s_row[g])
    return out


def dequant_v(body: np.ndarray, scales: np.ndarray, head_dim: int, group_size: int, bits: int) -> np.ndarray:
    out = np.zeros((group_size, head_dim), dtype=np.float32)
    s_col = scales[:head_dim]
    s_row = scales[head_dim:head_dim + group_size]
    zp = scales[head_dim + group_size:head_dim + 2 * group_size]
    for g in range(group_size):
        for d in range(head_dim):
            q = unpack_one(body, bits, g * head_dim + d)
            out[g, d] = np.float32((np.float32(q) * s_row[g] + zp[g]) * s_col[d])
    return out


def resolve_dump(path: Path) -> Path:
    meta = path / "body_record.json"
    if meta.exists():
        return path
    children = sorted([p for p in path.iterdir() if (p / "body_record.json").exists()])
    if len(children) == 1:
        return children[0]
    raise SystemExit(f"body_record.json not found under {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a KVarN body-record store dump.")
    parser.add_argument("--dump", required=True, help="Dump directory or parent directory containing one dump")
    parser.add_argument("--max-frame-nmse", type=float, default=None)
    parser.add_argument("--max-dequant-nmse", type=float, default=None)
    args = parser.parse_args()

    root = resolve_dump(Path(args.dump))
    meta = json.loads((root / "body_record.json").read_text())
    head_dim = int(meta["head_dim"])
    group_size = int(meta["group_size"])
    key_bits = int(meta["key_bits"])
    value_bits = int(meta["value_bits"])

    k_input = np.fromfile(root / "k_tile_input.bin", dtype=np.float32).reshape(head_dim, group_size)
    v_input = np.fromfile(root / "v_tile_input.bin", dtype=np.float32).reshape(group_size, head_dim)
    k_rot = np.fromfile(root / "k_rot_or_copy.bin", dtype=np.float32).reshape(head_dim, group_size)
    v_rot = np.fromfile(root / "v_rot_or_copy.bin", dtype=np.float32).reshape(group_size, head_dim)
    k_body = np.fromfile(root / "k_body.bin", dtype=np.uint8)
    v_body = np.fromfile(root / "v_body.bin", dtype=np.uint8)
    scales_k = np.fromfile(root / "scales_k.bin", dtype=np.float32)
    scales_v = np.fromfile(root / "scales_v.bin", dtype=np.float32)

    if bool(meta["input_already_rotated"]):
        k_frame_ref = k_input
        v_frame_ref = v_input
    else:
        k_frame_ref = hadamard_last(k_input.T).T
        v_frame_ref = hadamard_last(v_input)

    k_deq = dequant_k(k_body, scales_k, head_dim, group_size, key_bits)
    v_deq = dequant_v(v_body, scales_v, head_dim, group_size, value_bits)

    k_frame_nmse = nmse(k_frame_ref, k_rot)
    v_frame_nmse = nmse(v_frame_ref, v_rot)
    k_deq_nmse = nmse(k_rot, k_deq)
    v_deq_nmse = nmse(v_rot, v_deq)

    print(
        "Body-record replay: "
        f"path={root} head_dim={head_dim} group={group_size} "
        f"k_frame_nmse={k_frame_nmse:.6e} v_frame_nmse={v_frame_nmse:.6e} "
        f"k_dequant_nmse={k_deq_nmse:.6e} v_dequant_nmse={v_deq_nmse:.6e}"
    )

    if args.max_frame_nmse is not None:
        worst_frame = max(k_frame_nmse, v_frame_nmse)
        if worst_frame > args.max_frame_nmse:
            raise SystemExit(f"frame NMSE {worst_frame:.6e} exceeds {args.max_frame_nmse:.6e}")
    if args.max_dequant_nmse is not None:
        worst_dequant = max(k_deq_nmse, v_deq_nmse)
        if worst_dequant > args.max_dequant_nmse:
            raise SystemExit(f"dequant NMSE {worst_dequant:.6e} exceeds {args.max_dequant_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
