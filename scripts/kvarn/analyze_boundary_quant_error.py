#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from kvarn_vllm_oracle import (
    PRESETS,
    rtn_quantize_dequant_per_row,
    store_dequant_k_vllm,
    store_dequant_v_vllm,
    variance_normalize_log_std,
)
from replay_f16_truth_boundary import body_records, load_raw_body, resolve_boundary, softmax_rows
from replay_mixed_attn_boundary import read_full_mask, reconstruct_kv


def nmse_rows(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    r = ref.astype(np.float64)
    g = got.astype(np.float64)
    d = r - g
    denom = np.sum(r * r, axis=1)
    num = np.sum(d * d, axis=1)
    out = np.empty(ref.shape[0], dtype=np.float64)
    zero = denom == 0.0
    out[~zero] = num[~zero] / denom[~zero]
    out[zero] = np.where(num[zero] == 0.0, 0.0, np.inf)
    return out


def replay(
        full_q: np.ndarray,
        full_q_body: np.ndarray | None,
        full_mask: np.ndarray,
        k_all: np.ndarray,
        v_all: np.ndarray,
        meta: dict,
        scale: np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    group_size = int(meta["group_size"])
    n_pending = int(meta.get("n_pending", 0))
    n_body = n_records * group_size
    q_body = full_q if full_q_body is None else full_q_body
    k_t = k_all.astype(np.float32).T
    scores = np.zeros((full_q.shape[0], k_all.shape[0]), dtype=np.float32)
    if n_sink:
        scores[:, :n_sink] = (full_q @ k_t[:, :n_sink]).astype(np.float32)
    if n_body:
        scores[:, n_sink:n_sink + n_body] = (q_body @ k_t[:, n_sink:n_sink + n_body]).astype(np.float32)
    pending0 = n_sink + n_body
    pending1 = pending0 + n_pending
    if n_pending:
        scores[:, pending0:pending1] = (full_q @ k_t[:, pending0:pending1]).astype(np.float32)
    if pending1 < k_all.shape[0]:
        scores[:, pending1:] = (full_q @ k_t[:, pending1:]).astype(np.float32)
    scores = scores * scale
    scores = (scores + full_mask).astype(np.float32)
    probs = softmax_rows(scores)
    out = (probs @ v_all.astype(np.float32)).astype(np.float32)
    return scores, probs, out


def max_abs_rows(ref: np.ndarray, got: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    idx = np.argmax(diff, axis=1).astype(np.int64)
    val = diff[np.arange(diff.shape[0]), idx]
    return val, idx


def summarize_rows(ref_out: np.ndarray, got_out: np.ndarray) -> dict:
    row_nmse = nmse_rows(ref_out, got_out)
    row_mae, row_d = max_abs_rows(ref_out, got_out)
    worst_i = int(np.argmax(row_nmse)) if row_nmse.size else 0
    return {
        "max_out_nmse": float(row_nmse[worst_i]) if row_nmse.size else 0.0,
        "mean_out_nmse": float(np.mean(row_nmse)) if row_nmse.size else 0.0,
        "worst_row_index": worst_i,
        "worst_out_max_abs": float(row_mae[worst_i]) if row_mae.size else 0.0,
        "worst_d": int(row_d[worst_i]) if row_d.size else 0,
    }


def finite_max_abs_delta(ref: np.ndarray, got: np.ndarray) -> float:
    mask = np.isfinite(ref) & np.isfinite(got)
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(got[mask].astype(np.float64) - ref[mask].astype(np.float64))))


def build_vllm_body(raw_body_k: np.ndarray, raw_body_v: np.ndarray, head_dim: int, group_size: int, key_bits: int, value_bits: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    if raw_body_k.shape != raw_body_v.shape:
        raise SystemExit(f"raw body K/V shape mismatch: {raw_body_k.shape} vs {raw_body_v.shape}")
    if raw_body_k.shape[1] != head_dim or raw_body_k.shape[0] % group_size != 0:
        raise SystemExit(f"raw body shape {raw_body_k.shape} incompatible with head_dim={head_dim} group={group_size}")

    n_records = raw_body_k.shape[0] // group_size
    k_chunks = []
    v_chunks = []
    for r in range(n_records):
        k_gd = raw_body_k[r*group_size:(r + 1)*group_size].astype(np.float32)
        v_gd = raw_body_v[r*group_size:(r + 1)*group_size].astype(np.float32)
        k_store = store_dequant_k_vllm(k_gd.T.copy(), key_bits, iters)
        v_store = store_dequant_v_vllm(v_gd.copy(), value_bits, iters)
        k_chunks.append(k_store["deq_rot"].T.copy())
        v_chunks.append(v_store["deq_rot"].copy())
    if not k_chunks:
        return np.zeros((0, head_dim), dtype=np.float32), np.zeros((0, head_dim), dtype=np.float32)
    return np.concatenate(k_chunks, axis=0).astype(np.float32), np.concatenate(v_chunks, axis=0).astype(np.float32)


def store_dequant_v_vllm_blocked_quant(v_rot_gd: np.ndarray, bits: int, iterations: int, block_size: int) -> np.ndarray:
    """Use the normal V Sinkhorn, but split the per-row RTN range by channel block."""
    if v_rot_gd.ndim != 2:
        raise ValueError(f"expected [G,D] V tile, got {v_rot_gd.shape}")
    if block_size <= 0 or v_rot_gd.shape[1] % block_size != 0:
        raise ValueError(f"block_size={block_size} does not divide head_dim={v_rot_gd.shape[1]}")

    balanced, s_col, s_row, _ = variance_normalize_log_std(v_rot_gd, iterations)
    deq_bal = np.empty_like(balanced, dtype=np.float32)
    for d0 in range(0, balanced.shape[1], block_size):
        d1 = d0 + block_size
        _, _, _, block_deq = rtn_quantize_dequant_per_row(balanced[:, d0:d1], bits)
        deq_bal[:, d0:d1] = block_deq
    return (deq_bal * s_row * s_col).astype(np.float32)


def build_blocked_v_body(raw_body_v: np.ndarray, head_dim: int, group_size: int, value_bits: int, iters: int, block_size: int) -> np.ndarray:
    if raw_body_v.shape[1] != head_dim or raw_body_v.shape[0] % group_size != 0:
        raise SystemExit(f"raw body V shape {raw_body_v.shape} incompatible with head_dim={head_dim} group={group_size}")
    chunks = []
    for r in range(raw_body_v.shape[0] // group_size):
        v_gd = raw_body_v[r*group_size:(r + 1)*group_size].astype(np.float32)
        chunks.append(store_dequant_v_vllm_blocked_quant(v_gd, value_bits, iters, block_size))
    if not chunks:
        return np.zeros((0, head_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def nmse_all(ref: np.ndarray, got: np.ndarray) -> float:
    r = ref.astype(np.float64)
    g = got.astype(np.float64)
    denom = float(np.sum(r * r))
    num = float(np.sum((r - g) * (r - g)))
    if denom == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / denom


TURBO_S1 = np.asarray([
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
], dtype=np.float32)

TURBO_S2 = np.asarray([
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
], dtype=np.float32)

TURBO_CENTROIDS = {
    2: np.asarray([-0.133462, -0.039994, 0.039994, 0.133462], dtype=np.float32),
    3: np.asarray([
        -0.190685, -0.117832, -0.065717, -0.021460,
         0.021460,  0.065717,  0.117832,  0.190685,
    ], dtype=np.float32),
    4: np.asarray([
        -0.173926, -0.117195, -0.089527, -0.068756,
        -0.051262, -0.035597, -0.020989, -0.006938,
         0.006938,  0.020989,  0.035597,  0.051262,
         0.068756,  0.089527,  0.117195,  0.173926,
    ], dtype=np.float32),
}


def turbo_fwht_blocks(x: np.ndarray, inverse: bool) -> np.ndarray:
    if x.shape[-1] % 128 != 0:
        raise ValueError(f"TurboQuant oracle requires channel dimension multiple of 128, got {x.shape}")
    out = x.astype(np.float32, copy=True).reshape(-1, 128)
    first = TURBO_S2 if inverse else TURBO_S1
    last = TURBO_S1 if inverse else TURBO_S2
    out *= first
    h = 1
    while h < 128:
        y = out.reshape(-1, 128 // (2*h), 2*h)
        a = y[:, :, :h].copy()
        b = y[:, :, h:2*h].copy()
        y[:, :, :h] = a + b
        y[:, :, h:2*h] = a - b
        h *= 2
    out *= np.float32(1.0 / np.sqrt(128.0))
    out *= last
    return out.reshape(x.shape).astype(np.float32)


def turbo_quant_dequant_rotated_rows(x: np.ndarray, bits: int) -> np.ndarray:
    centroids = TURBO_CENTROIDS[bits]
    rows = x.astype(np.float32, copy=False).reshape(-1, 128)
    out = np.empty_like(rows, dtype=np.float32)
    mids = ((centroids[:-1] + centroids[1:]) * np.float32(0.5)).astype(np.float32)
    for i, row in enumerate(rows):
        norm = np.float32(np.linalg.norm(row.astype(np.float64)))
        if norm <= np.float32(1.0e-10):
            out[i].fill(0.0)
            continue
        normalized = row / norm
        rotated = turbo_fwht_blocks(normalized.reshape(1, 128), inverse=False).reshape(128)
        idx = np.searchsorted(mids, rotated, side="left")
        recon = centroids[idx]
        recon_norm = np.float32(np.linalg.norm(recon.astype(np.float64)))
        corrected = norm / recon_norm if recon_norm > np.float32(1.0e-10) else norm
        out[i] = recon * corrected
    return out.reshape(x.shape).astype(np.float32)


def build_turbo_v_body_rot(raw_body_v: np.ndarray, bits: int) -> np.ndarray:
    return turbo_quant_dequant_rotated_rows(raw_body_v, bits)


def replay_raw_k_turbo_body_v(
        raw_probs: np.ndarray,
        sink_tail_v: np.ndarray,
        raw_body_v: np.ndarray,
        n_sink: int,
        n_records: int,
        group_size: int,
        n_tail: int,
        bits: int) -> tuple[np.ndarray, np.ndarray]:
    n_body = n_records * group_size
    exact_st_rot = turbo_fwht_blocks(sink_tail_v, inverse=False)
    body_rot = build_turbo_v_body_rot(raw_body_v, bits)
    v_rot = np.concatenate([
        exact_st_rot[:n_sink],
        body_rot,
        exact_st_rot[n_sink:n_sink + n_tail],
    ], axis=0).astype(np.float32)
    out_rot = (raw_probs @ v_rot).astype(np.float32)
    out = turbo_fwht_blocks(out_rot, inverse=True)
    body_deq_original = turbo_fwht_blocks(body_rot, inverse=True)
    if body_deq_original.shape[0] != n_body:
        raise AssertionError("TurboQuant body shape mismatch")
    return out.astype(np.float32), body_deq_original.astype(np.float32)


def replay_raw_k_turbo_f16_v(
        raw_probs: np.ndarray,
        raw_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v_rot_f16 = turbo_fwht_blocks(raw_v, inverse=False).astype(np.float16).astype(np.float32)
    out_rot = (raw_probs @ v_rot_f16).astype(np.float32)
    out = turbo_fwht_blocks(out_rot, inverse=True)
    v_restored = turbo_fwht_blocks(v_rot_f16, inverse=True)
    return out.astype(np.float32), v_restored.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare current KVarN body dequant to raw f16 and vLLM best-so-far dequant on one boundary.")
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--body-records", required=True)
    ap.add_argument("--preset", default="kvarn_k8v8_g128", choices=sorted(PRESETS.keys()))
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--record-set", choices=["earliest", "latest"], default="latest")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    boundary = resolve_boundary(Path(args.boundary))
    meta = json.loads((boundary / "boundary.json").read_text())
    head_dim = int(meta["head_dim"])
    group_size = int(meta["group_size"])
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    n_tail = int(meta["n_tail"])
    n_tokens = int(meta["n_tokens"])
    layer = int(meta.get("inferred_layer", -1))
    selected_ikh = int(meta["selected_ikh"])
    scale = np.float32(float(meta["scale"]))
    preset = PRESETS[args.preset]
    if preset.group != group_size:
        raise SystemExit(f"preset group {preset.group} != boundary group {group_size}")

    full_q_path = boundary / "full_q.bin"
    full_out_path = boundary / "full_out.bin"
    if not full_q_path.exists() or not full_out_path.exists():
        raise SystemExit("boundary must include full_q.bin and full_out.bin; capture with -DumpFullQO")
    full_mask = read_full_mask(boundary, meta, int(meta["n_queries"]), n_tokens)
    if full_mask is None:
        raise SystemExit("boundary must include full_mask.bin; capture with -DumpFullQO")

    n_queries = int(meta["n_queries"])
    n_head = int(meta["n_head"])
    n_gqa = int(meta["n_gqa"])
    full_q_all = np.fromfile(full_q_path, dtype=np.float32, count=n_queries*n_head*head_dim).reshape(n_queries, n_head, head_dim)
    full_q_body_path = boundary / "full_q_body.bin"
    full_q_body_all = (
        np.fromfile(full_q_body_path, dtype=np.float32, count=n_queries*n_head*head_dim).reshape(n_queries, n_head, head_dim)
        if full_q_body_path.exists() else None
    )
    actual_out_all = np.fromfile(full_out_path, dtype=np.float32, count=n_queries*n_head*head_dim).reshape(n_queries, n_head, head_dim)
    heads = list(range(selected_ikh*n_gqa, min(n_head, (selected_ikh + 1)*n_gqa)))
    q_rows = full_q_all[:, heads, :].reshape(n_queries*len(heads), head_dim)
    q_body_rows = None if full_q_body_all is None else full_q_body_all[:, heads, :].reshape(n_queries*len(heads), head_dim)
    actual_rows = actual_out_all[:, heads, :].reshape(n_queries*len(heads), head_dim)
    mask_rows = np.repeat(full_mask, len(heads), axis=0)
    row_iq = np.repeat(np.arange(n_queries), len(heads))
    row_ih = np.tile(np.asarray(heads), n_queries)

    sink_tail_k = np.fromfile(boundary / "sink_tail_k_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, head_dim)
    sink_tail_v = np.fromfile(boundary / "sink_tail_v_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, head_dim)
    raw_records = body_records(Path(args.body_records), layer, selected_ikh, args.record_set)
    raw_body_k, raw_body_v = load_raw_body(raw_records, n_records, head_dim, group_size)
    raw_k = np.concatenate([sink_tail_k[:n_sink], raw_body_k, sink_tail_k[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)
    raw_v = np.concatenate([sink_tail_v[:n_sink], raw_body_v, sink_tail_v[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)

    cur_k, cur_v = reconstruct_kv(boundary, meta)
    vllm_body_k, vllm_body_v = build_vllm_body(raw_body_k, raw_body_v, head_dim, group_size, preset.key_bits, preset.value_bits, args.iters)
    vllm_k = np.concatenate([sink_tail_k[:n_sink], vllm_body_k, sink_tail_k[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)
    vllm_v = np.concatenate([sink_tail_v[:n_sink], vllm_body_v, sink_tail_v[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)
    blocked_v_bodies = {}
    for block_size in (256, 128):
        if head_dim % block_size == 0 and block_size < head_dim:
            blocked_v_bodies[f"vllm_v_blocked_{block_size}"] = build_blocked_v_body(
                raw_body_v, head_dim, group_size, preset.value_bits, args.iters, block_size)
    vllm_body_v4 = None
    if preset.value_bits != 4:
        vllm_body_v4 = build_blocked_v_body(raw_body_v, head_dim, group_size, 4, args.iters, head_dim)

    raw_scores, raw_probs, raw_out = replay(q_rows, q_body_rows, mask_rows, raw_k, raw_v, meta, scale)
    cur_scores, cur_probs, cur_out = replay(q_rows, q_body_rows, mask_rows, cur_k, cur_v, meta, scale)
    vllm_scores, vllm_probs, vllm_out = replay(q_rows, q_body_rows, mask_rows, vllm_k, vllm_v, meta, scale)
    _, _, cur_k_raw_v_out = replay(q_rows, q_body_rows, mask_rows, cur_k, raw_v, meta, scale)
    _, _, raw_k_cur_v_out = replay(q_rows, q_body_rows, mask_rows, raw_k, cur_v, meta, scale)
    _, _, vllm_k_raw_v_out = replay(q_rows, q_body_rows, mask_rows, vllm_k, raw_v, meta, scale)
    _, _, raw_k_vllm_v_out = replay(q_rows, q_body_rows, mask_rows, raw_k, vllm_v, meta, scale)
    turbo_v_out = {}
    turbo_v_body = {}
    for turbo_bits in (2, 3, 4):
        turbo_v_out[turbo_bits], turbo_v_body[turbo_bits] = replay_raw_k_turbo_body_v(
            raw_probs, sink_tail_v, raw_body_v, n_sink, n_records, group_size, n_tail, turbo_bits)
    turbo_f16_v_out, turbo_f16_v_restored = replay_raw_k_turbo_f16_v(raw_probs, raw_v)

    cur_nmse = nmse_rows(raw_out, cur_out)
    vllm_nmse = nmse_rows(raw_out, vllm_out)
    actual_nmse = nmse_rows(cur_out, actual_rows)
    cur_mae, cur_d = max_abs_rows(raw_out, cur_out)
    vllm_mae, vllm_d = max_abs_rows(raw_out, vllm_out)

    rows = []
    for i in range(q_rows.shape[0]):
        raw_top = int(np.argmax(raw_probs[i]))
        cur_top = int(np.argmax(cur_probs[i]))
        vllm_top = int(np.argmax(vllm_probs[i]))
        rows.append({
            "iq": int(row_iq[i]),
            "ih": int(row_ih[i]),
            "cur_out_nmse": float(cur_nmse[i]),
            "vllm_out_nmse": float(vllm_nmse[i]),
            "actual_vs_cur_nmse": float(actual_nmse[i]),
            "cur_out_max_abs": float(cur_mae[i]),
            "vllm_out_max_abs": float(vllm_mae[i]),
            "cur_worst_d": int(cur_d[i]),
            "vllm_worst_d": int(vllm_d[i]),
            "raw_top": raw_top,
            "cur_top": cur_top,
            "vllm_top": vllm_top,
            "raw_top_prob": float(raw_probs[i, raw_top]),
            "cur_top_prob_at_raw_top": float(cur_probs[i, raw_top]),
            "vllm_top_prob_at_raw_top": float(vllm_probs[i, raw_top]),
            "cur_max_score_abs_delta": finite_max_abs_delta(raw_scores[i], cur_scores[i]),
            "vllm_max_score_abs_delta": finite_max_abs_delta(raw_scores[i], vllm_scores[i]),
        })

    csv_path = Path(args.csv) if args.csv else boundary / "quant_error_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    worst_cur = max(rows, key=lambda r: r["cur_out_nmse"])
    worst_vllm = max(rows, key=lambda r: r["vllm_out_nmse"])
    summary = {
        "boundary": str(boundary),
        "body_records": str(args.body_records),
        "layer": layer,
        "selected_ikh": selected_ikh,
        "rows": len(rows),
        "preset": args.preset,
        "iters": args.iters,
        "current": {
            "max_out_nmse": float(max(cur_nmse)),
            "mean_out_nmse": float(np.mean(cur_nmse)),
            "worst": worst_cur,
        },
        "vllm_best_so_far": {
            "max_out_nmse": float(max(vllm_nmse)),
            "mean_out_nmse": float(np.mean(vllm_nmse)),
            "worst": worst_vllm,
        },
        "ablation": {
            "current_k_raw_v": summarize_rows(raw_out, cur_k_raw_v_out),
            "raw_k_current_v": summarize_rows(raw_out, raw_k_cur_v_out),
            "vllm_k_raw_v": summarize_rows(raw_out, vllm_k_raw_v_out),
            "raw_k_vllm_v": summarize_rows(raw_out, raw_k_vllm_v_out),
            "raw_k_turbo2_body_v": summarize_rows(raw_out, turbo_v_out[2]),
            "raw_k_turbo3_body_v": summarize_rows(raw_out, turbo_v_out[3]),
            "raw_k_turbo4_body_v": summarize_rows(raw_out, turbo_v_out[4]),
            "raw_k_turbo_f16_v_no_quant": summarize_rows(raw_out, turbo_f16_v_out),
        },
        "actual_vs_current_max_nmse": float(max(actual_nmse)),
        "body_dequant_nmse": {
            "current_k": nmse_all(raw_body_k, cur_k[n_sink:n_sink + n_records*group_size]),
            "current_v": nmse_all(raw_body_v, cur_v[n_sink:n_sink + n_records*group_size]),
            "vllm_k": nmse_all(raw_body_k, vllm_body_k),
            "vllm_v": nmse_all(raw_body_v, vllm_body_v),
            "turbo2_v_body_original_domain": nmse_all(raw_body_v, turbo_v_body[2]),
            "turbo3_v_body_original_domain": nmse_all(raw_body_v, turbo_v_body[3]),
            "turbo4_v_body_original_domain": nmse_all(raw_body_v, turbo_v_body[4]),
            "turbo_f16_v_no_quant_all_original_domain": nmse_all(raw_v, turbo_f16_v_restored),
            "turbo_f16_v_no_quant_body_original_domain": nmse_all(raw_body_v, turbo_f16_v_restored[n_sink:n_sink + n_records*group_size]),
        },
        "csv": str(csv_path),
    }
    for name, body_v in blocked_v_bodies.items():
        summary["body_dequant_nmse"][name] = nmse_all(raw_body_v, body_v)
    if vllm_body_v4 is not None:
        summary["body_dequant_nmse"]["vllm_v_full_width_4bit"] = nmse_all(raw_body_v, vllm_body_v4)
    (boundary / "quant_error_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "Boundary quant-error analysis: "
        f"boundary={boundary.name} rows={len(rows)} "
        f"current_max_nmse={summary['current']['max_out_nmse']:.6e} "
        f"vllm_max_nmse={summary['vllm_best_so_far']['max_out_nmse']:.6e} "
        f"actual_vs_current_max_nmse={summary['actual_vs_current_max_nmse']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
