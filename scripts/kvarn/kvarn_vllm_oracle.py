#!/usr/bin/env python3
"""
Independent KVarN oracle matching the Huawei/vLLM reference contracts.

This script is intentionally independent from llama.cpp CUDA/body-store code.  It
exists to catch cases where packed-vs-split/scratch all agree with each other
while sharing the same wrong layout or normalization recipe.

Reference contracts:
  * K body tile layout: [D, G]  (channels x tokens)
  * V body tile layout: [G, D]  (tokens x channels)
  * Hadamard rotation is along head_dim.
  * Log-domain std Sinkhorn is the vLLM reference recipe.
  * K scales: [s_col_K' D][zp_K' D][s_row_K G]
  * V scales: [s_col_V D][s_row_V' G][zp_V' G]
  * Attention uses rotated Q/K/V and unrotates the output.

Run:
  python scripts/kvarn/kvarn_vllm_oracle.py --self-test
  python scripts/kvarn/kvarn_vllm_oracle.py --self-test --head-dims 128,256,512 --presets k4v2,k4v4,k8v8
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from typing import Iterable

import numpy as np


CLIP_STD_MIN = 1.0e-3
CLIP_STD_MAX = 1.0e3
LOG_S_MIN = -0.3
LOG_S_MAX = 10.0


@dataclasses.dataclass(frozen=True)
class Preset:
    name: str
    key_bits: int
    value_bits: int
    group: int


PRESETS = {
    "kvarn_k4v2_g128": Preset("kvarn_k4v2_g128", 4, 2, 128),
    "kvarn_k4v4_g128": Preset("kvarn_k4v4_g128", 4, 4, 128),
    "kvarn_k8v8_g128": Preset("kvarn_k8v8_g128", 8, 8, 128),
    "k4v2": Preset("kvarn_k4v2_g128", 4, 2, 128),
    "k4v4": Preset("kvarn_k4v4_g128", 4, 4, 128),
    "k8v8": Preset("kvarn_k8v8_g128", 8, 8, 128),
}


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def hadamard_last(x: np.ndarray) -> np.ndarray:
    """Apply normalized Sylvester-Hadamard transform along the last axis."""
    y = np.asarray(x, dtype=np.float32).copy()
    d = y.shape[-1]
    if not _is_power_of_two(d):
        raise ValueError(f"Hadamard dim must be power-of-two, got {d}")

    step = 1
    while step < d:
        yy = y.reshape(*y.shape[:-1], d // (2 * step), 2, step)
        a = yy[..., 0, :].copy()
        b = yy[..., 1, :].copy()
        yy[..., 0, :] = a + b
        yy[..., 1, :] = a - b
        step *= 2

    y *= 1.0 / math.sqrt(float(d))
    return y


def _std(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    # PyTorch torch.std(dim=...) defaults to correction=1. Match that.
    return np.std(x, axis=axis, ddof=1, keepdims=keepdims).astype(np.float32)


def _imbalance(tile: np.ndarray) -> float:
    sc = _std(tile, axis=-2, keepdims=False)
    sr = _std(tile, axis=-1, keepdims=False)
    return float(sc.max() / max(float(sc.min()), 1.0e-8) +
                 sr.max() / max(float(sr.min()), 1.0e-8))


def variance_normalize_log_std(tile: np.ndarray, iterations: int = 8):
    """vLLM-style log-domain std Sinkhorn for one [R,C] tile."""
    m = np.asarray(tile, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"expected [R,C] tile, got shape {m.shape}")

    r, c = m.shape
    log_s_col = np.zeros((1, c), dtype=np.float32)
    log_s_row = np.zeros((r, 1), dtype=np.float32)

    cur = m / np.exp(log_s_col) / np.exp(log_s_row)
    imb_best = _imbalance(cur)
    sc_best = np.exp(log_s_col).astype(np.float32).copy()
    sr_best = np.exp(log_s_row).astype(np.float32).copy()

    for _ in range(iterations):
        col_std = np.clip(_std(cur, axis=0, keepdims=True), CLIP_STD_MIN, CLIP_STD_MAX)
        log_s_col = np.clip(log_s_col + np.log(col_std), LOG_S_MIN, LOG_S_MAX).astype(np.float32)
        cur = m / np.exp(log_s_col) / np.exp(log_s_row)

        row_std = np.clip(_std(cur, axis=1, keepdims=True), CLIP_STD_MIN, CLIP_STD_MAX)
        log_s_row = np.clip(log_s_row + np.log(row_std), LOG_S_MIN, LOG_S_MAX).astype(np.float32)
        cur = m / np.exp(log_s_col) / np.exp(log_s_row)

        imb = _imbalance(cur)
        if imb <= imb_best:
            imb_best = imb
            sc_best = np.exp(log_s_col).astype(np.float32).copy()
            sr_best = np.exp(log_s_row).astype(np.float32).copy()

    balanced = m / sc_best / sr_best
    return balanced.astype(np.float32), sc_best.astype(np.float32), sr_best.astype(np.float32), imb_best


def variance_normalize_rms_last_iter(tile: np.ndarray, iterations: int = 4):
    """Current llama.cpp-style linear RMS Sinkhorn approximation for comparison."""
    data = np.asarray(tile, dtype=np.float32).copy()
    rows, cols = data.shape
    row_scale = np.ones((rows, 1), dtype=np.float32)
    col_scale = np.ones((1, cols), dtype=np.float32)
    eps = 1.0e-6

    for _ in range(iterations):
        row_rms = np.sqrt(np.mean(data * data, axis=1, keepdims=True) + eps).astype(np.float32)
        row_scale *= row_rms
        data /= row_rms

        col_rms = np.sqrt(np.mean(data * data, axis=0, keepdims=True) + eps).astype(np.float32)
        col_scale *= col_rms
        data /= col_rms

    return data, col_scale, row_scale, _imbalance(data)


def rtn_quantize_dequant_per_row(tile: np.ndarray, bits: int):
    """Asymmetric per-row RTN. Returns q, scale, zp, dequantized tile."""
    qmax = float((1 << bits) - 1)
    lo = tile.min(axis=1, keepdims=True).astype(np.float32)
    hi = tile.max(axis=1, keepdims=True).astype(np.float32)
    scale = np.maximum((hi - lo) / qmax, 1.0e-10).astype(np.float32)
    q = np.rint((tile - lo) / scale).clip(0, qmax).astype(np.uint8)
    deq = (q.astype(np.float32) * scale + lo).astype(np.float32)
    return q, scale, lo, deq


def store_dequant_k_vllm(k_rot_dg: np.ndarray, bits: int, iterations: int):
    balanced, s_col, s_row, imb = variance_normalize_log_std(k_rot_dg, iterations)
    q, rtn_scale, rtn_zp, deq_bal = rtn_quantize_dequant_per_row(balanced, bits)
    # K: s_col is per-token [1,G]; s_row is per-channel [D,1].
    s_col_K = (s_row * rtn_scale).squeeze(1).astype(np.float32)
    zp_K = (s_row * rtn_zp).squeeze(1).astype(np.float32)
    s_row_K = s_col.squeeze(0).astype(np.float32)
    deq_rot = (deq_bal * s_row * s_col).astype(np.float32)
    return {
        "q": q,
        "s_col_K": s_col_K,
        "zp_K": zp_K,
        "s_row_K": s_row_K,
        "deq_rot": deq_rot,
        "imbalance": imb,
    }


def store_dequant_v_vllm(v_rot_gd: np.ndarray, bits: int, iterations: int):
    balanced, s_col, s_row, imb = variance_normalize_log_std(v_rot_gd, iterations)
    q, rtn_scale, rtn_zp, deq_bal = rtn_quantize_dequant_per_row(balanced, bits)
    # V: s_col is per-channel [1,D]; s_row is per-token [G,1].
    s_col_V = s_col.squeeze(0).astype(np.float32)
    s_row_V = (s_row * rtn_scale).squeeze(1).astype(np.float32)
    zp_V = (s_row * rtn_zp).squeeze(1).astype(np.float32)
    deq_rot = (deq_bal * s_row * s_col).astype(np.float32)
    return {
        "q": q,
        "s_col_V": s_col_V,
        "s_row_V": s_row_V,
        "zp_V": zp_V,
        "deq_rot": deq_rot,
        "imbalance": imb,
    }


def softmax(x: np.ndarray) -> np.ndarray:
    xx = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(xx).astype(np.float32)
    return exp / exp.sum(axis=-1, keepdims=True)


def attention_oracle(q_raw_qd: np.ndarray, k_raw_gd: np.ndarray, v_raw_gd: np.ndarray,
                     preset: Preset, iterations: int):
    """Reference KVarN attention for one tile."""
    d = q_raw_qd.shape[-1]
    if k_raw_gd.shape != (preset.group, d):
        raise ValueError(f"k shape {k_raw_gd.shape} != {(preset.group, d)}")
    if v_raw_gd.shape != (preset.group, d):
        raise ValueError(f"v shape {v_raw_gd.shape} != {(preset.group, d)}")

    q_rot = hadamard_last(q_raw_qd)
    k_rot_dg = hadamard_last(k_raw_gd).T.copy()
    v_rot_gd = hadamard_last(v_raw_gd)

    k_store = store_dequant_k_vllm(k_rot_dg, preset.key_bits, iterations)
    v_store = store_dequant_v_vllm(v_rot_gd, preset.value_bits, iterations)

    scores = (q_rot @ k_store["deq_rot"]) / math.sqrt(float(d))
    probs = softmax(scores)
    out_rot = probs @ v_store["deq_rot"]
    out = hadamard_last(out_rot)
    return out, {
        "q_rot": q_rot,
        "k_rot_dg": k_rot_dg,
        "v_rot_gd": v_rot_gd,
        "k_deq_rot_dg": k_store["deq_rot"],
        "v_deq_rot_gd": v_store["deq_rot"],
        "scores": scores,
        "probs": probs,
        "k_imbalance": k_store["imbalance"],
        "v_imbalance": v_store["imbalance"],
    }


def fp32_attention(q_raw_qd: np.ndarray, k_raw_gd: np.ndarray, v_raw_gd: np.ndarray) -> np.ndarray:
    scores = (q_raw_qd @ k_raw_gd.T) / math.sqrt(float(q_raw_qd.shape[-1]))
    return softmax(scores) @ v_raw_gd


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.sum(a.astype(np.float64) ** 2))
    if den == 0.0:
        return float(np.sum((a - b).astype(np.float64) ** 2))
    return float(np.sum((a - b).astype(np.float64) ** 2) / den)


def run_self_test(head_dims: Iterable[int], presets: Iterable[Preset], seed: int,
                  iterations: int, queries: int) -> int:
    rng = np.random.default_rng(seed)
    failures = 0

    for d in head_dims:
        if not _is_power_of_two(d):
            raise ValueError(f"head_dim must be power-of-two: {d}")
        for preset in presets:
            g = preset.group
            q = rng.standard_normal((queries, d), dtype=np.float32) * 0.11
            k = rng.standard_normal((g, d), dtype=np.float32) * 0.09
            v = rng.standard_normal((g, d), dtype=np.float32) * 0.07

            # Add structured token/channel outliers to exercise token-scale error.
            k[g // 3, :] *= 4.0
            v[:, d // 5] *= 3.0

            ref = fp32_attention(q, k, v)
            out, dbg = attention_oracle(q, k, v, preset, iterations)
            e = nmse(ref, out)

            # Internal shape/layout invariants.
            ok = True
            ok &= dbg["k_rot_dg"].shape == (d, g)
            ok &= dbg["v_rot_gd"].shape == (g, d)
            ok &= dbg["k_deq_rot_dg"].shape == (d, g)
            ok &= dbg["v_deq_rot_gd"].shape == (g, d)
            ok &= np.all(np.isfinite(out))
            ok &= np.all(np.isfinite(dbg["scores"]))
            ok &= np.allclose(hadamard_last(hadamard_last(q)), q, rtol=2e-5, atol=2e-5)

            status = "PASS" if ok else "FAIL"
            print(
                f"{status} head_dim={d} preset={preset.name} iters={iterations} "
                f"nmse_vs_fp32={e:.6e} k_imb={dbg['k_imbalance']:.4f} v_imb={dbg['v_imbalance']:.4f}"
            )
            failures += 0 if ok else 1

    return failures


def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_presets(s: str) -> list[Preset]:
    result = []
    for name in [x.strip() for x in s.split(",") if x.strip()]:
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name}; known: {', '.join(sorted(PRESETS))}")
        result.append(PRESETS[name])
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--head-dims", default="128,256,512")
    ap.add_argument("--presets", default="k4v2,k4v4,k8v8")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--queries", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    if not args.self_test:
        ap.error("only --self-test is implemented in the initial oracle patch")

    failures = run_self_test(
        parse_csv_ints(args.head_dims),
        parse_presets(args.presets),
        args.seed,
        args.iters,
        args.queries,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
