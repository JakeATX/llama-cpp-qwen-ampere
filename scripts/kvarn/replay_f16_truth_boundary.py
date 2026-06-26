#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from replay_mixed_attn_boundary import read_full_mask


def nmse(ref: np.ndarray, got: np.ndarray) -> float:
    r = ref.astype(np.float64)
    g = got.astype(np.float64)
    d = r - g
    denom = float(np.sum(r * r))
    return 0.0 if denom == 0.0 and float(np.sum(d * d)) == 0.0 else float(np.sum(d * d) / denom)


def max_abs(ref: np.ndarray, got: np.ndarray) -> tuple[float, int]:
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    if d.size == 0:
        return 0.0, 0
    i = int(np.argmax(d))
    return float(d.reshape(-1)[i]), i


def softmax(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32) - np.max(x.astype(np.float32))
    p = np.exp(y).astype(np.float32)
    return (p / np.sum(p, dtype=np.float32)).astype(np.float32)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32) - np.max(x.astype(np.float32), axis=1, keepdims=True)
    p = np.exp(y).astype(np.float32)
    return (p / np.sum(p, axis=1, keepdims=True, dtype=np.float32)).astype(np.float32)


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


def resolve_boundary(path: Path) -> Path:
    if (path / "boundary.json").exists():
        return path
    children = sorted(p for p in path.iterdir() if (p / "boundary.json").exists())
    if len(children) != 1:
        raise SystemExit(f"expected exactly one boundary dump under {path}, found {len(children)}")
    return children[0]


def body_records(root: Path, layer: int, head: int, record_set: str) -> dict[int, Path]:
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for meta_path in root.rglob("body_record.json"):
        meta = json.loads(meta_path.read_text())
        if int(meta.get("layer", -1)) != layer:
            continue
        if int(meta.get("head", -1)) != head:
            continue
        record = int(meta["record"])
        if int(meta.get("src_layout", -1)) == 1:
            record += int(meta.get("record0", 0))
        call_index = int(meta.get("call_index", 0))
        grouped.setdefault(record, []).append((call_index, meta_path.parent))
    out: dict[int, Path] = {}
    for record, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        out[record] = entries[-1 if record_set == "latest" else 0][1]
    return out


def read_mask(root: Path, meta: dict, n_tokens: int) -> np.ndarray:
    path = root / "mask.bin"
    mask_type = int(meta.get("mask_type", 0))
    if mask_type == 3:
        n_queries = int(meta.get("n_queries", 1))
        selected_iq = int(meta.get("selected_iq", max(0, n_queries - 1)))
        limit = (n_tokens - n_queries + selected_iq) if n_tokens >= n_queries else selected_iq
        mask = np.full(n_tokens, -np.inf, dtype=np.float32)
        mask[:min(n_tokens, limit + 1)] = 0.0
        return mask
    if not path.exists() or mask_type == 0:
        return np.zeros(n_tokens, dtype=np.float32)
    if mask_type == 1:
        return np.fromfile(path, dtype=np.float32, count=n_tokens).astype(np.float32)
    if mask_type == 2:
        return np.fromfile(path, dtype=np.float16, count=n_tokens).astype(np.float32)
    raise SystemExit(f"unsupported mask_type={mask_type}")


def load_raw_body(records: dict[int, Path], n_records: int, head_dim: int, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    k_chunks = []
    v_chunks = []
    missing = []
    for record in range(n_records):
        root = records.get(record)
        if root is None:
            missing.append(record)
            continue
        meta = json.loads((root / "body_record.json").read_text())
        if int(meta["head_dim"]) != head_dim or int(meta["group_size"]) != group_size:
            raise SystemExit(f"record {record} shape mismatch in {root}")
        # K is channel-major [head_dim, group_size]; V is token-major [group_size, head_dim].
        k = np.fromfile(root / "k_rot_or_copy.bin", dtype=np.float32, count=head_dim * group_size)
        v = np.fromfile(root / "v_rot_or_copy.bin", dtype=np.float32, count=head_dim * group_size)
        if k.size != head_dim * group_size or v.size != head_dim * group_size:
            raise SystemExit(f"record {record} has truncated raw dump in {root}")
        k_chunks.append(k.reshape(head_dim, group_size).T.copy())
        v_chunks.append(v.reshape(group_size, head_dim).copy())
    if missing:
        raise SystemExit(f"missing body record dumps for records: {missing[:16]}{'...' if len(missing) > 16 else ''}")
    if not k_chunks:
        return np.zeros((0, head_dim), dtype=np.float32), np.zeros((0, head_dim), dtype=np.float32)
    return np.concatenate(k_chunks, axis=0).astype(np.float32), np.concatenate(v_chunks, axis=0).astype(np.float32)


def mixed_frame_scores(
        q: np.ndarray,
        q_body: np.ndarray | None,
        k_all: np.ndarray,
        meta: dict,
        scale: np.float32,
        mask: np.ndarray) -> np.ndarray:
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
    return (scores * scale + mask).astype(np.float32)


def replay_full_qo_truth(
        boundary: Path,
        meta: dict,
        k_all: np.ndarray,
        v_all: np.ndarray,
        scale: np.float32) -> None:
    full_q_path = boundary / "full_q.bin"
    full_out_path = boundary / "full_out.bin"
    if not full_q_path.exists() or not full_out_path.exists():
        return

    head_dim = int(meta["head_dim"])
    n_queries = int(meta["n_queries"])
    n_head = int(meta["n_head"])
    n_gqa = int(meta["n_gqa"])
    n_tokens = int(meta["n_tokens"])
    selected_ikh = int(meta["selected_ikh"])

    full_mask = read_full_mask(boundary, meta, n_queries, n_tokens)
    if full_mask is None:
        print("  f16_truth_full_qo: skipped because full_mask.bin is not present")
        return

    expected = n_queries * n_head * head_dim
    full_q = np.fromfile(full_q_path, dtype=np.float32, count=expected).reshape(n_queries, n_head, head_dim)
    full_q_body_path = boundary / "full_q_body.bin"
    full_q_body = (
        np.fromfile(full_q_body_path, dtype=np.float32, count=expected).reshape(n_queries, n_head, head_dim)
        if full_q_body_path.exists() else None
    )
    full_out = np.fromfile(full_out_path, dtype=np.float32, count=expected).reshape(n_queries, n_head, head_dim)

    heads = list(range(selected_ikh * n_gqa, min(n_head, (selected_ikh + 1) * n_gqa)))
    k_t = k_all.astype(np.float32).T
    v = v_all.astype(np.float32)

    rows: list[dict[str, int | float]] = []
    worst: dict[str, int | float] | None = None
    for iq in range(n_queries):
        q = full_q[iq, heads, :].astype(np.float32)
        q_body = q if full_q_body is None else full_q_body[iq, heads, :].astype(np.float32)
        n_sink = int(meta["n_sink"])
        n_records = int(meta["n_records"])
        group_size = int(meta["group_size"])
        n_body = n_records * group_size
        scores = np.zeros((len(heads), k_all.shape[0]), dtype=np.float32)
        if n_sink:
            scores[:, :n_sink] = (q @ k_t[:, :n_sink]).astype(np.float32)
        if n_body:
            scores[:, n_sink:n_sink + n_body] = (q_body @ k_t[:, n_sink:n_sink + n_body]).astype(np.float32)
        tail0 = n_sink + n_body
        if tail0 < k_all.shape[0]:
            scores[:, tail0:] = (q @ k_t[:, tail0:]).astype(np.float32)
        scores = scores * scale
        scores = (scores + full_mask[iq].reshape(1, -1)).astype(np.float32)
        probs = softmax_rows(scores)
        truth_out = (probs @ v).astype(np.float32)
        actual_out = full_out[iq, heads, :].astype(np.float32)

        diff = np.abs(truth_out.astype(np.float64) - actual_out.astype(np.float64))
        worst_d = np.argmax(diff, axis=1).astype(np.int64)
        max_abs = diff[np.arange(len(heads)), worst_d]
        out_nmse = nmse_rows(truth_out, actual_out)

        for row_i, ih in enumerate(heads):
            row = {
                "iq": int(iq),
                "ih": int(ih),
                "ikh": int(ih // n_gqa),
                "out_nmse": float(out_nmse[row_i]),
                "out_max_abs": float(max_abs[row_i]),
                "out_worst_d": int(worst_d[row_i]),
            }
            rows.append(row)
            if worst is None or float(row["out_nmse"]) > float(worst["out_nmse"]):
                worst = row

    csv_path = boundary / "f16_truth_full_qo_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iq", "ih", "ikh", "out_nmse", "out_max_abs", "out_worst_d"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "boundary": str(boundary),
        "layer": int(meta.get("inferred_layer", -1)),
        "selected_ikh": selected_ikh,
        "covered_heads": heads,
        "rows": len(rows),
        "worst": worst,
        "max_out_nmse": max((float(row["out_nmse"]) for row in rows), default=0.0),
        "max_out_abs": max((float(row["out_max_abs"]) for row in rows), default=0.0),
    }
    (boundary / "f16_truth_full_qo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if worst is None:
        print("  f16_truth_full_qo: no replayable rows")
    else:
        print(
            "  f16_truth_full_qo: "
            f"rows={len(rows)} selected_ikh={selected_ikh} "
            f"worst_iq={worst['iq']} worst_ih={worst['ih']} "
            f"out_nmse={float(worst['out_nmse']):.6e} out_max_abs={float(worst['out_max_abs']):.6e}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare a KVarN boundary dump against raw f16/f32 body-store truth.")
    ap.add_argument("--boundary", required=True, help="Boundary dump dir or parent containing one boundary.json")
    ap.add_argument("--body-records", required=True, help="Body-record dump parent containing body_record.json children")
    ap.add_argument("--record-set", choices=["earliest", "latest"], default="latest",
                    help="Which duplicate body-record dump to use when records were captured more than once")
    ap.add_argument("--max-out-nmse", type=float, default=None)
    ap.add_argument("--write-truth", action="store_true")
    args = ap.parse_args()

    boundary = resolve_boundary(Path(args.boundary))
    meta = json.loads((boundary / "boundary.json").read_text())
    head_dim = int(meta["head_dim"])
    group_size = int(meta["group_size"])
    n_sink = int(meta["n_sink"])
    n_records = int(meta["n_records"])
    n_pending = int(meta["n_pending"])
    n_tail = int(meta["n_tail"])
    n_tokens = int(meta["n_tokens"])
    layer = int(meta.get("inferred_layer", -1))
    head = int(meta["selected_ikh"])
    scale = np.float32(float(meta["scale"]))

    if n_pending != 0:
        raise SystemExit("f16 truth replay currently expects n_pending=0; capture a no-pending ctx4096 boundary first")
    if n_tokens != n_sink + n_records * group_size + n_tail:
        raise SystemExit("boundary n_tokens does not match sink/body/tail layout")

    q = np.fromfile(boundary / "q.bin", dtype=np.float32, count=head_dim).astype(np.float32)
    q_body_path = boundary / "q_body.bin"
    q_body = np.fromfile(q_body_path, dtype=np.float32, count=head_dim).astype(np.float32) if q_body_path.exists() else None
    actual_out_path = boundary / "mixed_out.bin"
    if not actual_out_path.exists():
        actual_out_path = boundary / "warpqk_out.bin"
    actual_out = np.fromfile(actual_out_path, dtype=np.float32, count=head_dim).astype(np.float32)
    mask = read_mask(boundary, meta, n_tokens)

    sink_tail_k = np.fromfile(boundary / "sink_tail_k_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, head_dim)
    sink_tail_v = np.fromfile(boundary / "sink_tail_v_f16.bin", dtype=np.float16).astype(np.float32).reshape(n_sink + n_tail, head_dim)
    raw_records = body_records(Path(args.body_records), layer, head, args.record_set)
    body_k, body_v = load_raw_body(raw_records, n_records, head_dim, group_size)

    k_all = np.concatenate([sink_tail_k[:n_sink], body_k, sink_tail_k[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)
    v_all = np.concatenate([sink_tail_v[:n_sink], body_v, sink_tail_v[n_sink:n_sink + n_tail]], axis=0).astype(np.float32)
    if k_all.shape != (n_tokens, head_dim) or v_all.shape != (n_tokens, head_dim):
        raise SystemExit(f"truth K/V shape mismatch: K={k_all.shape} V={v_all.shape} expected={(n_tokens, head_dim)}")

    scores = mixed_frame_scores(q, q_body, k_all, meta, scale, mask)
    probs = softmax(scores)
    truth_out = (probs.astype(np.float32) @ v_all.astype(np.float32)).astype(np.float32)

    out_nmse = nmse(truth_out, actual_out)
    out_mae, out_idx = max_abs(truth_out, actual_out)
    top_truth = int(np.argmax(probs))
    mask_allowed = int(np.sum(mask > -1.0e20))

    print(
        "F16-truth boundary replay: "
        f"boundary={boundary} body_records={args.body_records} layer={layer} head={head} "
        f"tokens={n_tokens} records={n_records} mask_allowed={mask_allowed} "
        f"top_truth_t={top_truth} out_nmse={out_nmse:.6e} out_max_abs={out_mae:.6e} out_worst_d={out_idx}"
    )

    if args.write_truth:
        scores.astype(np.float32).tofile(boundary / "f16_truth_scores.bin")
        probs.astype(np.float32).tofile(boundary / "f16_truth_probs.bin")
        truth_out.astype(np.float32).tofile(boundary / "f16_truth_out.bin")

    replay_full_qo_truth(boundary, meta, k_all, v_all, scale)

    if args.max_out_nmse is not None and out_nmse > args.max_out_nmse:
        raise SystemExit(f"truth/KVarN out NMSE {out_nmse:.6e} exceeds {args.max_out_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
