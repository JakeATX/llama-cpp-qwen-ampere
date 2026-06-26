#!/usr/bin/env python3
"""Compare row-bound tensor dumps by explicit KVarN dump metadata.

This intentionally does not do similarity matching, nearest-row matching, or
occurrence arithmetic. Rows are matched only by:

  canonical tensor name + inferred_full_row

or:

  canonical tensor name + inferred_scored_row

The script is meant for Gemma/Qwen KVarN root-cause work where repeated prompt
patterns make cosine/similarity matching unsafe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Dump:
    directory: Path
    json_path: Path
    bin_path: Path
    name: str
    canonical_name: str
    dtype: str
    row_key: int
    inferred_full_row: int
    inferred_scored_row: int
    source_row: int
    name_occurrence: int
    source_ne: List[int]
    source_nb: List[int]
    n_values: int


def canonicalize_name(name: str, mapping: Dict[str, str]) -> str:
    return mapping.get(name, name)


def load_meta(path: Path, mapping: Dict[str, str], row_mode: str) -> Optional[Dump]:
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    name = str(meta.get("name", ""))
    canonical = canonicalize_name(name, mapping)
    full = int(meta.get("inferred_full_row", -1))
    scored = int(meta.get("inferred_scored_row", -1))
    row_key = full if row_mode == "full" else scored
    if row_key < 0:
        return None

    ne = [int(x) for x in meta.get("ne", [])]
    if not ne:
        return None
    n_values = 1
    for v in ne:
        n_values *= max(1, int(v))

    bin_name = str(meta.get("bin", ""))
    if not bin_name:
        return None

    return Dump(
        directory=path.parent,
        json_path=path,
        bin_path=path.parent / bin_name,
        name=name,
        canonical_name=canonical,
        dtype=str(meta.get("type", "")),
        row_key=row_key,
        inferred_full_row=full,
        inferred_scored_row=scored,
        source_row=int(meta.get("source_row", -1)),
        name_occurrence=int(meta.get("name_occurrence", -1)),
        source_ne=[int(x) for x in meta.get("source_ne", [])],
        source_nb=[int(x) for x in meta.get("source_nb", [])],
        n_values=n_values,
    )


def load_dir(directory: Path, mapping: Dict[str, str], row_mode: str) -> Dict[Tuple[str, int], List[Dump]]:
    out: Dict[Tuple[str, int], List[Dump]] = {}
    for path in sorted(directory.glob("*.json")):
        d = load_meta(path, mapping, row_mode)
        if d is None:
            continue
        out.setdefault((d.canonical_name, d.row_key), []).append(d)
    return out


def fp16_to_float_list(data: bytes, n: int) -> List[float]:
    vals = []
    for i in range(n):
        h = struct.unpack_from("<H", data, 2 * i)[0]
        vals.append(half_to_float(h))
    return vals


def half_to_float(h: int) -> float:
    s = (h >> 15) & 0x0001
    e = (h >> 10) & 0x001f
    f = h & 0x03ff
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        return ((-1.0) ** s) * 2.0 ** (-14) * (f / 1024.0)
    if e == 31:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    return ((-1.0) ** s) * 2.0 ** (e - 15) * (1.0 + f / 1024.0)


def read_values(d: Dump) -> List[float]:
    data = d.bin_path.read_bytes()
    if d.dtype == "f32":
        n = len(data) // 4
        return list(struct.unpack("<" + "f" * n, data))
    if d.dtype == "f16":
        return fp16_to_float_list(data, len(data) // 2)
    if d.dtype in {"i32", "u32"}:
        n = len(data) // 4
        fmt = "i" if d.dtype == "i32" else "I"
        return [float(x) for x in struct.unpack("<" + fmt * n, data)]
    if d.dtype in {"i64", "u64"}:
        n = len(data) // 8
        fmt = "q" if d.dtype == "i64" else "Q"
        return [float(x) for x in struct.unpack("<" + fmt * n, data)]
    raise ValueError(f"unsupported dtype {d.dtype} for {d.json_path}")


def metrics(a: List[float], b: List[float]) -> Dict[str, float]:
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "nmse": math.nan, "rmse": math.nan, "mae": math.nan, "max_abs": math.nan}
    se = 0.0
    ref = 0.0
    ae = 0.0
    max_abs = 0.0
    for i in range(n):
        da = b[i] - a[i]
        se += da * da
        ref += a[i] * a[i]
        ad = abs(da)
        ae += ad
        if ad > max_abs:
            max_abs = ad
    return {
        "n": float(n),
        "nmse": se / max(ref, 1.0e-30),
        "rmse": math.sqrt(se / n),
        "mae": ae / n,
        "max_abs": max_abs,
    }


def parse_map(values: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--map value must be FROM=TO, got {item!r}")
        src, dst = item.split("=", 1)
        mapping[src] = dst
    return mapping


def choose_one(dumps: List[Dump]) -> Dump:
    # If multiple dumps have the same canonical name and row key, keep the last
    # occurrence. In practice this usually means the final graph occurrence for
    # that row/tensor name. The CSV still records occurrence/file names.
    return sorted(dumps, key=lambda d: (d.name_occurrence, d.json_path.name))[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--kvarn-dir", required=True, type=Path)
    ap.add_argument("--row", required=True, type=int)
    ap.add_argument("--row-mode", choices=["full", "scored"], default="full")
    ap.add_argument("--map", action="append", default=[], help="KVarN_NAME=BASE_NAME canonical name mapping")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--first-bad-nmse", type=float, default=1e-4)
    args = ap.parse_args()

    mapping = parse_map(args.map)
    base = load_dir(args.base_dir, {}, args.row_mode)
    kvarn = load_dir(args.kvarn_dir, mapping, args.row_mode)

    rows: List[Dict[str, object]] = []
    for key in sorted(set(base) & set(kvarn)):
        name, row = key
        if row != args.row:
            continue
        bd = choose_one(base[key])
        kd = choose_one(kvarn[key])
        try:
            bv = read_values(bd)
            kv = read_values(kd)
            m = metrics(bv, kv)
            err = ""
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            m = {"n": 0, "nmse": math.nan, "rmse": math.nan, "mae": math.nan, "max_abs": math.nan}
            err = str(exc)

        rows.append({
            "name": name,
            "row": row,
            "base_name": bd.name,
            "kvarn_name": kd.name,
            "base_source_row": bd.source_row,
            "kvarn_source_row": kd.source_row,
            "base_occurrence": bd.name_occurrence,
            "kvarn_occurrence": kd.name_occurrence,
            "dtype_base": bd.dtype,
            "dtype_kvarn": kd.dtype,
            "n": int(m["n"]),
            "nmse": m["nmse"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "max_abs": m["max_abs"],
            "base_file": str(bd.json_path),
            "kvarn_file": str(kd.json_path),
            "error": err,
        })

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "name", "row", "base_name", "kvarn_name",
                "base_source_row", "kvarn_source_row",
                "base_occurrence", "kvarn_occurrence",
                "dtype_base", "dtype_kvarn", "n", "nmse", "rmse", "mae", "max_abs",
                "base_file", "kvarn_file", "error",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"matched rows: {len(rows)}")
    first_bad = None
    for r in rows:
        nmse = r["nmse"]
        if isinstance(nmse, float) and not math.isnan(nmse) and nmse > args.first_bad_nmse:
            first_bad = r
            break
    for r in rows:
        print(
            f"{r['name']:45s} row={r['row']} nmse={r['nmse']:.6e} "
            f"rmse={r['rmse']:.6e} max_abs={r['max_abs']:.6e} "
            f"base_src={r['base_source_row']} kvarn_src={r['kvarn_source_row']}"
        )
    if first_bad is not None:
        print(
            "FIRST_BAD "
            f"name={first_bad['name']} row={first_bad['row']} "
            f"nmse={first_bad['nmse']:.6e} base={first_bad['base_file']} kvarn={first_bad['kvarn_file']}"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
