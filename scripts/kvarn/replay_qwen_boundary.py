#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def unpack_one(buf: np.ndarray, bits: int, index: int) -> int:
    bit_pos = index * bits
    byte_pos = bit_pos >> 3
    shift = bit_pos & 7
    value = int(buf[byte_pos]) >> shift
    if shift + bits > 8:
        value |= int(buf[byte_pos + 1]) << (8 - shift)
    return value & ((1 << bits) - 1)


def f32(x) -> np.float32:
    return np.float32(x)


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    denom = np.sum(a.astype(np.float64) * a.astype(np.float64))
    if denom == 0.0:
        return 0.0 if np.sum(diff * diff) == 0.0 else float("inf")
    return float(np.sum(diff * diff) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a captured Qwen 256d KVarN boundary on CPU.")
    parser.add_argument("--dump", required=True, help="Boundary dump directory containing boundary.json")
    parser.add_argument("--write-reference", action="store_true", help="Write split_scores.bin, split_probs.bin and split_out.bin")
    parser.add_argument("--max-out-nmse", type=float, default=None)
    args = parser.parse_args()

    root = Path(args.dump)
    meta_path = root / "boundary.json"
    if not meta_path.exists():
        children = [p for p in root.iterdir() if p.is_dir()]
        if len(children) == 1 and (children[0] / "boundary.json").exists():
            root = children[0]
            meta_path = root / "boundary.json"
    if not meta_path.exists():
        raise SystemExit(f"boundary.json not found under {root}")

    meta = json.loads(meta_path.read_text())
    head_dim = int(meta["head_dim"])
    group_size = int(meta["group_size"])
    key_bits = int(meta["key_bits"])
    value_bits = int(meta["value_bits"])
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    n_pending = int(meta["n_pending"])
    n_tail = int(meta["n_tail"])
    n_tokens = int(meta["n_tokens"])
    scale = f32(meta["scale"])
    mask_type = int(meta["mask_type"])

    q = np.fromfile(root / "q.bin", dtype=np.float32, count=head_dim)
    sink_k = np.fromfile(root / "sink_tail_k_f16.bin", dtype=np.float16).reshape(n_sink + n_tail, head_dim)
    sink_v = np.fromfile(root / "sink_tail_v_f16.bin", dtype=np.float16).reshape(n_sink + n_tail, head_dim)
    body_k = np.fromfile(root / "body_k.bin", dtype=np.uint8)
    body_v = np.fromfile(root / "body_v.bin", dtype=np.uint8)
    scales_k = np.fromfile(root / "scales_k.bin", dtype=np.float32)
    scales_v = np.fromfile(root / "scales_v.bin", dtype=np.float32)
    pending_k = np.fromfile(root / "pending_k.bin", dtype=np.float32)
    pending_v = np.fromfile(root / "pending_v.bin", dtype=np.float32)
    warpqk_out = np.fromfile(root / "warpqk_out.bin", dtype=np.float32, count=head_dim)

    if scales_k.size:
        scales_k = scales_k.reshape(n_records, -1)
    else:
        scales_k = np.zeros((0, 2 * head_dim + group_size), dtype=np.float32)
    if scales_v.size:
        scales_v = scales_v.reshape(n_records, -1)
    else:
        scales_v = np.zeros((0, head_dim + 2 * group_size), dtype=np.float32)
    if pending_k.size:
        pending_k = pending_k.reshape(n_pending, head_dim)
        pending_v = pending_v.reshape(n_pending, head_dim)
    else:
        pending_k = np.zeros((0, head_dim), dtype=np.float32)
        pending_v = np.zeros((0, head_dim), dtype=np.float32)

    k_body_record_bytes = int(meta["k_body_stride_record_bytes"])
    v_body_record_bytes = int(meta["v_body_stride_record_bytes"])
    body_k = body_k.reshape(n_records, k_body_record_bytes) if n_records else np.zeros((0, k_body_record_bytes), dtype=np.uint8)
    body_v = body_v.reshape(n_records, v_body_record_bytes) if n_records else np.zeros((0, v_body_record_bytes), dtype=np.uint8)

    if mask_type == 1:
        mask = np.fromfile(root / "mask.bin", dtype=np.float32, count=n_tokens)
    elif mask_type == 2:
        mask = np.fromfile(root / "mask.bin", dtype=np.float16, count=n_tokens).astype(np.float32)
    else:
        mask = np.zeros(n_tokens, dtype=np.float32)

    scores = np.zeros(n_tokens, dtype=np.float32)
    body_tokens = n_records * group_size
    for t in range(n_tokens):
        s = np.float32(0.0)
        if t < n_sink:
            k = sink_k[t].astype(np.float32)
            for d in range(head_dim):
                s = f32(s + f32(q[d] * k[d]))
        elif t < n_sink + body_tokens:
            body_t = t - n_sink
            r = body_t // group_size
            g = body_t - r * group_size
            k_s_col = scales_k[r, :head_dim]
            k_zp = scales_k[r, head_dim:2 * head_dim]
            k_s_row = scales_k[r, 2 * head_dim:2 * head_dim + group_size]
            rec = body_k[r]
            for d in range(head_dim):
                kq = unpack_one(rec, key_bits, d * group_size + g)
                kv = f32(f32(float(kq) * k_s_col[d] + k_zp[d]) * k_s_row[g])
                s = f32(s + f32(q[d] * kv))
        elif t < n_sink + body_tokens + n_pending:
            k = pending_k[t - n_sink - body_tokens]
            for d in range(head_dim):
                s = f32(s + f32(q[d] * k[d]))
        else:
            k = sink_k[n_sink + (t - n_sink - body_tokens - n_pending)].astype(np.float32)
            for d in range(head_dim):
                s = f32(s + f32(q[d] * k[d]))
        scores[t] = f32(f32(s * scale) + mask[t])

    max_score = np.max(scores).astype(np.float32)
    probs = np.exp((scores - max_score).astype(np.float32)).astype(np.float32)
    denom = np.sum(probs.astype(np.float32), dtype=np.float32)
    probs = (probs / denom).astype(np.float32)

    out = np.zeros(head_dim, dtype=np.float32)
    for d in range(head_dim):
        s = np.float32(0.0)
        for t in range(n_sink):
            s = f32(s + f32(probs[t] * np.float32(sink_v[t, d])))
        for r in range(n_records):
            v_s_col = scales_v[r, :head_dim]
            v_s_row = scales_v[r, head_dim:head_dim + group_size]
            v_zp = scales_v[r, head_dim + group_size:head_dim + 2 * group_size]
            rec = body_v[r]
            for g in range(group_size):
                vq = unpack_one(rec, value_bits, g * head_dim + d)
                vv = f32(f32(float(vq) * v_s_row[g] + v_zp[g]) * v_s_col[d])
                s = f32(s + f32(probs[n_sink + r * group_size + g] * vv))
        for t in range(n_pending):
            s = f32(s + f32(probs[n_sink + body_tokens + t] * pending_v[t, d]))
        for t in range(n_tail):
            s = f32(s + f32(probs[n_sink + body_tokens + n_pending + t] * np.float32(sink_v[n_sink + t, d])))
        out[d] = s

    if args.write_reference:
        scores.tofile(root / "split_scores.bin")
        probs.tofile(root / "split_probs.bin")
        out.tofile(root / "split_out.bin")

    diff = np.abs(out.astype(np.float64) - warpqk_out.astype(np.float64))
    worst = int(np.argmax(diff)) if diff.size else 0
    out_nmse = nmse(out, warpqk_out)
    print(
        "Boundary CPU replay: "
        f"path={root} tokens={n_tokens} records={n_records} "
        f"out_nmse={out_nmse:.6e} out_max_abs={float(diff[worst]):.6e} worst_d={worst}"
    )
    if args.max_out_nmse is not None and out_nmse > args.max_out_nmse:
        raise SystemExit(f"out NMSE {out_nmse:.6e} exceeds {args.max_out_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
