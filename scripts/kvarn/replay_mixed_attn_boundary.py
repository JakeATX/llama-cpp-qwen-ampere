#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def nmse(ref: np.ndarray, got: np.ndarray) -> float:
    rr = ref.astype(np.float64)
    gg = got.astype(np.float64)
    diff = rr - gg
    denom = float(np.sum(rr * rr))
    if denom == 0.0:
        return 0.0 if float(np.sum(diff * diff)) == 0.0 else float("inf")
    return float(np.sum(diff * diff) / denom)


def max_abs(ref: np.ndarray, got: np.ndarray) -> tuple[float, int]:
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    idx = int(np.argmax(diff)) if diff.size else 0
    return float(diff.reshape(-1)[idx]) if diff.size else 0.0, idx


def unpack_one(buf: np.ndarray, bits: int, index: int) -> int:
    bit_pos = index * bits
    byte_pos = bit_pos >> 3
    shift = bit_pos & 7
    value = int(buf[byte_pos]) >> shift
    if shift + bits > 8:
        value |= int(buf[byte_pos + 1]) << (8 - shift)
    return value & ((1 << bits) - 1)


def dequant_k(
    body: np.ndarray,
    scales: np.ndarray,
    n_records: int,
    head_dim: int,
    group_size: int,
    bits: int,
    record_stride_bytes: int,
) -> np.ndarray:
    out = np.zeros((n_records, head_dim, group_size), dtype=np.float32)
    scale_floats = 2 * head_dim + group_size
    body_bytes = (head_dim * group_size * bits + 7) // 8
    if record_stride_bytes < body_bytes:
        raise ValueError(f"K body record stride {record_stride_bytes} is smaller than packed bytes {body_bytes}")
    for r in range(n_records):
        b = body[r * record_stride_bytes:r * record_stride_bytes + body_bytes]
        s = scales[r * scale_floats:(r + 1) * scale_floats]
        s_col = s[:head_dim]
        zp = s[head_dim:2 * head_dim]
        s_row = s[2 * head_dim:2 * head_dim + group_size]
        for d in range(head_dim):
            for g in range(group_size):
                q = unpack_one(b, bits, d * group_size + g)
                out[r, d, g] = np.float32((np.float32(q) * s_col[d] + zp[d]) * s_row[g])
    return out


def dequant_v(
    body: np.ndarray,
    scales: np.ndarray,
    n_records: int,
    head_dim: int,
    group_size: int,
    bits: int,
    record_stride_bytes: int,
) -> np.ndarray:
    out = np.zeros((n_records, group_size, head_dim), dtype=np.float32)
    scale_floats = head_dim + 2 * group_size
    body_bytes = (head_dim * group_size * bits + 7) // 8
    if record_stride_bytes < body_bytes:
        raise ValueError(f"V body record stride {record_stride_bytes} is smaller than packed bytes {body_bytes}")
    for r in range(n_records):
        b = body[r * record_stride_bytes:r * record_stride_bytes + body_bytes]
        s = scales[r * scale_floats:(r + 1) * scale_floats]
        s_col = s[:head_dim]
        s_row = s[head_dim:head_dim + group_size]
        zp = s[head_dim + group_size:head_dim + 2 * group_size]
        for g in range(group_size):
            for d in range(head_dim):
                q = unpack_one(b, bits, g * head_dim + d)
                out[r, g, d] = np.float32((np.float32(q) * s_row[g] + zp[g]) * s_col[d])
    return out


def resolve_dump(path: Path) -> Path:
    if (path / "boundary.json").exists():
        return path
    children = sorted(p for p in path.iterdir() if (p / "boundary.json").exists())
    if len(children) == 1:
        return children[0]
    raise SystemExit(f"boundary.json not found under {path}; found {len(children)} candidate children")


def read_mask(root: Path, meta: dict, n_tokens: int) -> np.ndarray:
    mask_path = root / "mask.bin"
    mask_type = int(meta.get("mask_type", 0))
    if mask_type == 3:
        n_queries = int(meta.get("n_queries", 1))
        selected_iq = int(meta.get("selected_iq", max(0, n_queries - 1)))
        limit = (n_tokens - n_queries + selected_iq) if n_tokens >= n_queries else selected_iq
        mask = np.full(n_tokens, -np.inf, dtype=np.float32)
        mask[:min(n_tokens, limit + 1)] = 0.0
        return mask
    if not mask_path.exists() or mask_type == 0:
        return np.zeros(n_tokens, dtype=np.float32)
    if mask_type == 1:
        return np.fromfile(mask_path, dtype=np.float32, count=n_tokens).astype(np.float32)
    if mask_type == 2:
        return np.fromfile(mask_path, dtype=np.float16, count=n_tokens).astype(np.float32)
    raise ValueError(f"unsupported mask_type={mask_type}")


def read_full_mask(root: Path, meta: dict, n_queries: int, n_tokens: int) -> np.ndarray | None:
    mask_path = root / "full_mask.bin"
    mask_type = int(meta.get("mask_type", 0))
    if mask_type == 3:
        mask = np.full((n_queries, n_tokens), -np.inf, dtype=np.float32)
        q_base = n_tokens - n_queries if n_tokens >= n_queries else 0
        for iq in range(n_queries):
            mask[iq, :min(n_tokens, q_base + iq + 1)] = 0.0
        return mask
    if not mask_path.exists() or mask_type == 0:
        return None
    if mask_type == 1:
        data = np.fromfile(mask_path, dtype=np.float32, count=n_queries * n_tokens).astype(np.float32)
    elif mask_type == 2:
        data = np.fromfile(mask_path, dtype=np.float16, count=n_queries * n_tokens).astype(np.float32)
    else:
        raise ValueError(f"unsupported mask_type={mask_type}")
    if data.size != n_queries * n_tokens:
        raise ValueError(f"full_mask.bin contains {data.size} elements, expected {n_queries * n_tokens}")
    return data.reshape(n_queries, n_tokens)


def softmax(scores: np.ndarray) -> np.ndarray:
    x = scores.astype(np.float32)
    x = x - np.max(x)
    p = np.exp(x).astype(np.float32)
    denom = np.sum(p, dtype=np.float32)
    return (p / denom).astype(np.float32)


def reconstruct_kv(root: Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    d = int(meta["head_dim"])
    g = int(meta["group_size"])
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    n_pending = int(meta["n_pending"])
    n_tail = int(meta["n_tail"])
    k_bits = int(meta["key_bits"])
    v_bits = int(meta["value_bits"])

    sink_tail_k = np.fromfile(root / "sink_tail_k_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, d)
    sink_tail_v = np.fromfile(root / "sink_tail_v_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, d)

    body_k = np.fromfile(root / "body_k.bin", dtype=np.uint8)
    body_v = np.fromfile(root / "body_v.bin", dtype=np.uint8)
    scales_k = np.fromfile(root / "scales_k.bin", dtype=np.float32)
    scales_v = np.fromfile(root / "scales_v.bin", dtype=np.float32)

    k_stride = int(meta.get("k_body_stride_record_bytes", (d * g * k_bits + 7) // 8))
    v_stride = int(meta.get("v_body_stride_record_bytes", (d * g * v_bits + 7) // 8))
    k_body_deq = dequant_k(body_k, scales_k, n_records, d, g, k_bits, k_stride)
    v_body_deq = dequant_v(body_v, scales_v, n_records, d, g, v_bits, v_stride)

    if n_pending:
        pending_k = np.fromfile(root / "pending_k.bin", dtype=np.float32).reshape(n_pending, d)
        pending_v = np.fromfile(root / "pending_v.bin", dtype=np.float32).reshape(n_pending, d)
    else:
        pending_k = np.zeros((0, d), dtype=np.float32)
        pending_v = np.zeros((0, d), dtype=np.float32)

    k_rows = [sink_tail_k[:n_sink]]
    v_rows = [sink_tail_v[:n_sink]]

    for r in range(n_records):
        k_rows.append(k_body_deq[r].T.copy())
        v_rows.append(v_body_deq[r].copy())

    k_rows.append(pending_k)
    v_rows.append(pending_v)
    k_rows.append(sink_tail_k[n_sink:n_sink + n_tail])
    v_rows.append(sink_tail_v[n_sink:n_sink + n_tail])

    return np.concatenate(k_rows, axis=0).astype(np.float32), np.concatenate(v_rows, axis=0).astype(np.float32)


def replay_row(
    q: np.ndarray,
    q_body: np.ndarray | None,
    mask: np.ndarray,
    k_all: np.ndarray,
    v_all: np.ndarray,
    meta: dict,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    group_size = int(meta["group_size"])
    n_pending = int(meta["n_pending"])
    n_body = n_records * group_size
    scores = np.zeros(k_all.shape[0], dtype=np.float32)
    if n_sink:
        scores[:n_sink] = (k_all[:n_sink] @ q).astype(np.float32)
    if n_body:
        body_q = q if q_body is None else q_body
        scores[n_sink:n_sink + n_body] = (k_all[n_sink:n_sink + n_body] @ body_q).astype(np.float32)
    pending0 = n_sink + n_body
    pending1 = pending0 + n_pending
    if n_pending:
        scores[pending0:pending1] = (k_all[pending0:pending1] @ q).astype(np.float32)
    if pending1 < k_all.shape[0]:
        scores[pending1:] = (k_all[pending1:] @ q).astype(np.float32)
    scores = scores * np.float32(scale) + mask
    probs = softmax(scores)
    replay_out = (probs.astype(np.float32) @ v_all.astype(np.float32)).astype(np.float32)
    return scores, probs, replay_out


def replay_full_qo(root: Path, meta: dict, k_all: np.ndarray, v_all: np.ndarray, scale: float) -> None:
    full_q_path = root / "full_q.bin"
    full_out_path = root / "full_out.bin"
    if not full_q_path.exists() or not full_out_path.exists():
        return

    d = int(meta["head_dim"])
    n_queries = int(meta["n_queries"])
    n_head = int(meta["n_head"])
    n_gqa = int(meta["n_gqa"])
    selected_ikh = int(meta["selected_ikh"])
    n_tokens = int(meta["n_tokens"])
    selected_iq = int(meta["selected_iq"])

    full_q = np.fromfile(full_q_path, dtype=np.float32, count=n_queries * n_head * d).reshape(n_queries, n_head, d)
    full_q_body_path = root / "full_q_body.bin"
    full_q_body = (
        np.fromfile(full_q_body_path, dtype=np.float32, count=n_queries * n_head * d).reshape(n_queries, n_head, d)
        if full_q_body_path.exists() else None
    )
    full_out = np.fromfile(full_out_path, dtype=np.float32, count=n_queries * n_head * d).reshape(n_queries, n_head, d)
    full_mask = read_full_mask(root, meta, n_queries, n_tokens)
    selected_mask = None if full_mask is not None else read_mask(root, meta, n_tokens)

    covered_heads = [ih for ih in range(n_head) if ih // n_gqa == selected_ikh]
    covered_queries = range(n_queries) if full_mask is not None else [selected_iq]

    rows: list[dict[str, int | float]] = []
    worst: dict[str, int | float] | None = None
    for iq in covered_queries:
        mask = full_mask[iq] if full_mask is not None else selected_mask
        for ih in covered_heads:
            row_q_body = None if full_q_body is None else full_q_body[iq, ih]
            _, _, replay_out = replay_row(full_q[iq, ih], row_q_body, mask, k_all, v_all, meta, scale)
            actual_out = full_out[iq, ih]
            row_nmse = nmse(replay_out, actual_out)
            row_mae, row_idx = max_abs(replay_out, actual_out)
            row = {
                "iq": int(iq),
                "ih": int(ih),
                "ikh": int(ih // n_gqa),
                "out_nmse": row_nmse,
                "out_max_abs": row_mae,
                "out_worst_d": int(row_idx),
            }
            rows.append(row)
            if worst is None or float(row["out_nmse"]) > float(worst["out_nmse"]):
                worst = row

    csv_path = root / "full_qo_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iq", "ih", "ikh", "out_nmse", "out_max_abs", "out_worst_d"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dump": str(root),
        "layer": meta.get("inferred_layer", -1),
        "selected_ikh": selected_ikh,
        "covered_heads": covered_heads,
        "covered_queries": list(covered_queries),
        "has_full_mask": full_mask is not None,
        "coverage_note": (
            "all queries for query heads mapped to selected KV head"
            if full_mask is not None else
            "selected query only because full_mask.bin is not present"
        ),
        "rows": len(rows),
        "worst": worst,
    }
    (root / "full_qo_summary.json").write_text(json.dumps(summary, indent=2))

    if worst is None:
        print("  full_qo: no replayable rows")
    else:
        print(
            "  full_qo: "
            f"rows={len(rows)} full_mask={full_mask is not None} selected_ikh={selected_ikh} "
            f"worst_iq={worst['iq']} worst_ih={worst['ih']} "
            f"out_nmse={float(worst['out_nmse']):.6e} out_max_abs={float(worst['out_max_abs']):.6e}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a KVarN mixed-attention boundary dump.")
    ap.add_argument("--dump", required=True, help="Dump directory or parent containing exactly one boundary dump")
    ap.add_argument("--max-score-nmse", type=float, default=None)
    ap.add_argument("--max-prob-nmse", type=float, default=None)
    ap.add_argument("--max-out-nmse", type=float, default=None)
    ap.add_argument("--write-replay", action="store_true", help="Write replay_scores/probs/out bin files into the dump")
    args = ap.parse_args()

    root = resolve_dump(Path(args.dump))
    meta = json.loads((root / "boundary.json").read_text())
    d = int(meta["head_dim"])
    n_tokens = int(meta["n_tokens"])
    scale = float(meta["scale"])

    q = np.fromfile(root / "q.bin", dtype=np.float32, count=d).astype(np.float32)
    q_body_path = root / "q_body.bin"
    q_body = np.fromfile(q_body_path, dtype=np.float32, count=d).astype(np.float32) if q_body_path.exists() else None
    out_file = root / "mixed_out.bin"
    if not out_file.exists():
        out_file = root / "warpqk_out.bin"
    actual_out = np.fromfile(out_file, dtype=np.float32, count=d).astype(np.float32)

    k_all, v_all = reconstruct_kv(root, meta)
    if k_all.shape != (n_tokens, d):
        raise SystemExit(f"reconstructed K shape {k_all.shape} != {(n_tokens, d)}")
    if v_all.shape != (n_tokens, d):
        raise SystemExit(f"reconstructed V shape {v_all.shape} != {(n_tokens, d)}")

    mask = read_mask(root, meta, n_tokens)
    scores, probs, replay_out = replay_row(q, q_body, mask, k_all, v_all, meta, scale)

    out_nmse = nmse(replay_out, actual_out)
    out_mae, out_idx = max_abs(replay_out, actual_out)

    print(
        "Mixed-attn replay: "
        f"path={root} layer={meta.get('inferred_layer', -1)} iq={meta.get('selected_iq')} ih={meta.get('selected_ih')} "
        f"tokens={n_tokens} records={meta.get('n_records')} mode={meta.get('cuda_trace_mode')} "
        f"out_nmse={out_nmse:.6e} out_max_abs={out_mae:.6e} out_worst_d={out_idx}"
    )

    if args.write_replay:
        scores.astype(np.float32).tofile(root / "replay_scores.bin")
        probs.astype(np.float32).tofile(root / "replay_probs.bin")
        replay_out.astype(np.float32).tofile(root / "replay_out.bin")

    replay_full_qo(root, meta, k_all, v_all, scale)

    score_path = root / "scores.bin"
    if score_path.exists():
        actual_scores = np.fromfile(score_path, dtype=np.float32, count=n_tokens).astype(np.float32)
        score_nmse = nmse(scores, actual_scores)
        score_mae, score_idx = max_abs(scores, actual_scores)
        print(f"  scores: nmse={score_nmse:.6e} max_abs={score_mae:.6e} worst_t={score_idx}")
        if args.max_score_nmse is not None and score_nmse > args.max_score_nmse:
            raise SystemExit(f"score NMSE {score_nmse:.6e} exceeds {args.max_score_nmse:.6e}")

    prob_path = root / "probs.bin"
    if prob_path.exists():
        actual_probs = np.fromfile(prob_path, dtype=np.float32, count=n_tokens).astype(np.float32)
        prob_nmse = nmse(probs, actual_probs)
        prob_mae, prob_idx = max_abs(probs, actual_probs)
        print(f"  probs: nmse={prob_nmse:.6e} max_abs={prob_mae:.6e} worst_t={prob_idx}")
        if args.max_prob_nmse is not None and prob_nmse > args.max_prob_nmse:
            raise SystemExit(f"prob NMSE {prob_nmse:.6e} exceeds {args.max_prob_nmse:.6e}")

    if args.max_out_nmse is not None and out_nmse > args.max_out_nmse:
        raise SystemExit(f"out NMSE {out_nmse:.6e} exceeds {args.max_out_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
