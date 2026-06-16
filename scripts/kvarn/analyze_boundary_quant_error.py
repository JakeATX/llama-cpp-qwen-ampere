#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from kvarn_vllm_oracle import PRESETS, store_dequant_k_vllm, store_dequant_v_vllm
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


def replay(full_q: np.ndarray, full_mask: np.ndarray, k_all: np.ndarray, v_all: np.ndarray, scale: np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = (full_q @ k_all.astype(np.float32).T).astype(np.float32) * scale
    scores = (scores + full_mask).astype(np.float32)
    probs = softmax_rows(scores)
    out = (probs @ v_all.astype(np.float32)).astype(np.float32)
    return scores, probs, out


def max_abs_rows(ref: np.ndarray, got: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    idx = np.argmax(diff, axis=1).astype(np.int64)
    val = diff[np.arange(diff.shape[0]), idx]
    return val, idx


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
    actual_out_all = np.fromfile(full_out_path, dtype=np.float32, count=n_queries*n_head*head_dim).reshape(n_queries, n_head, head_dim)
    heads = list(range(selected_ikh*n_gqa, min(n_head, (selected_ikh + 1)*n_gqa)))
    q_rows = full_q_all[:, heads, :].reshape(n_queries*len(heads), head_dim)
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

    raw_scores, raw_probs, raw_out = replay(q_rows, mask_rows, raw_k, raw_v, scale)
    cur_scores, cur_probs, cur_out = replay(q_rows, mask_rows, cur_k, cur_v, scale)
    vllm_scores, vllm_probs, vllm_out = replay(q_rows, mask_rows, vllm_k, vllm_v, scale)

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
        "actual_vs_current_max_nmse": float(max(actual_nmse)),
        "body_dequant_nmse": {
            "current_k": float(np.sum((raw_body_k.astype(np.float64) - cur_k[n_sink:n_sink + n_records*group_size].astype(np.float64))**2) /
                               max(float(np.sum(raw_body_k.astype(np.float64)**2)), 1.0e-300)),
            "current_v": float(np.sum((raw_body_v.astype(np.float64) - cur_v[n_sink:n_sink + n_records*group_size].astype(np.float64))**2) /
                               max(float(np.sum(raw_body_v.astype(np.float64)**2)), 1.0e-300)),
            "vllm_k": float(np.sum((raw_body_k.astype(np.float64) - vllm_body_k.astype(np.float64))**2) /
                            max(float(np.sum(raw_body_k.astype(np.float64)**2)), 1.0e-300)),
            "vllm_v": float(np.sum((raw_body_v.astype(np.float64) - vllm_body_v.astype(np.float64))**2) /
                            max(float(np.sum(raw_body_v.astype(np.float64)**2)), 1.0e-300)),
        },
        "csv": str(csv_path),
    }
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
