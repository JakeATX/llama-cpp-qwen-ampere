#!/usr/bin/env python3
"""Validate a llama-perplexity KL base file against its writer's PPL.

The historical `_logits_` KL-base format is lossy enough to corrupt Gemma KL
baselines. This checker decodes the target-token NLLs from a base file and
compares the resulting PPL to the PPL printed by the baseline run.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def decode_base(path: Path) -> dict:
    with path.open("rb") as f:
        magic = f.read(8)
        if len(magic) != 8:
            raise SystemExit(f"truncated KL base header: got {len(magic)} magic bytes, expected 8")
        if magic not in (b"_logits_", b"_logp16_", b"_logp16n"):
            raise SystemExit(f"unsupported KL base magic {magic!r}")
        header = f.read(12)
        if len(header) != 12:
            raise SystemExit(f"truncated KL base header: got {len(header)} metadata bytes, expected 12")
        n_ctx, n_vocab, n_chunk = struct.unpack("<iii", header)
        if n_ctx <= 1 or n_vocab <= 0 or n_chunk <= 0:
            raise SystemExit(f"invalid header n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk}")
        token_count = n_ctx * n_chunk
        tokens = np.fromfile(f, dtype=np.int32, count=token_count)
        if tokens.size != token_count:
            raise SystemExit(f"truncated token table: got {tokens.size}, expected {token_count}")
        data_offset = f.tell()

    first = n_ctx // 2
    n_eval = n_ctx - 1 - first
    if magic == b"_logits_":
        nv = 2 * ((n_vocab + 1) // 2) + 4
    elif magic == b"_logp16_":
        nv = n_vocab
    else:
        nv = n_vocab + 2
    expected_size = data_offset + n_chunk * n_eval * nv * np.dtype(np.uint16).itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(f"file size mismatch: got {actual_size}, expected {expected_size}")

    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=data_offset, shape=(n_chunk, n_eval, nv))
    sum_nll = 0.0
    count = 0
    for chunk in range(n_chunk):
        target_tokens = tokens[chunk * n_ctx + first + 1: chunk * n_ctx + first + 1 + n_eval].astype(np.int64)
        if np.any(target_tokens < 0) or np.any(target_tokens >= n_vocab):
            bad = target_tokens[(target_tokens < 0) | (target_tokens >= n_vocab)][0]
            raise SystemExit(f"target token {int(bad)} is outside vocab size {n_vocab}")
        block = rows[chunk]
        if magic == b"_logits_":
            scale_min = block[:, :4].copy().reshape(-1, 4).view(np.float32)
            scales = scale_min[:, 0].astype(np.float64)
            mins = scale_min[:, 1].astype(np.float64)
            qs = block[np.arange(n_eval), 4 + target_tokens].astype(np.float64)
            nll = -(scales * qs + mins)
        elif magic == b"_logp16_":
            vals = block[np.arange(n_eval), target_tokens].astype(np.uint16).view(np.float16).astype(np.float64)
            nll = -vals
        else:
            nll = block[:, :2].copy().reshape(-1, 2).view(np.float32).astype(np.float64).reshape(-1)
        sum_nll += float(np.sum(nll, dtype=np.float64))
        count += int(nll.size)

    log_ppl = sum_nll / count
    return {
        "path": str(path),
        "magic": magic.decode("ascii"),
        "n_ctx": n_ctx,
        "n_vocab": n_vocab,
        "n_chunk": n_chunk,
        "count": count,
        "log_ppl": log_ppl,
        "ppl": math.exp(log_ppl),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--expected-ppl", type=float, default=None)
    ap.add_argument("--max-log-ppl-diff", type=float, default=0.005)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    result = decode_base(args.base)
    print(
        "KL base file: "
        f"magic={result['magic']} n_ctx={result['n_ctx']} n_vocab={result['n_vocab']} "
        f"n_chunk={result['n_chunk']} count={result['count']} ppl={result['ppl']:.6f}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.expected_ppl is not None:
        if args.expected_ppl <= 0:
            raise SystemExit("--expected-ppl must be positive")
        diff = abs(math.log(result["ppl"]) - math.log(args.expected_ppl))
        print(f"KL base PPL check: expected={args.expected_ppl:.6f} log_diff={diff:.6e}")
        if diff > args.max_log_ppl_diff:
            raise SystemExit(
                f"KL base PPL log-diff {diff:.6e} exceeds {args.max_log_ppl_diff:.6e}; "
                "the KL baseline file is not faithful to the baseline run"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
