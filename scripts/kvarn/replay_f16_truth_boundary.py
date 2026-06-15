#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare a KVarN boundary dump against raw f16/f32 body-store truth.")
    ap.add_argument("--boundary", required=True, help="Boundary dump dir or parent containing one boundary.json")
    ap.add_argument("--body-records", required=True, help="Body-record dump parent containing body_record.json children")
    ap.add_argument("--record-set", choices=["earliest", "latest"], default="earliest",
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

    scores = (k_all @ q).astype(np.float32) * scale + mask
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

    if args.max_out_nmse is not None and out_nmse > args.max_out_nmse:
        raise SystemExit(f"truth/KVarN out NMSE {out_nmse:.6e} exceeds {args.max_out_nmse:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
