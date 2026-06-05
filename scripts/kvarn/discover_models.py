#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = REPO_ROOT / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFReader  # noqa: E402


def field_value(reader: GGUFReader, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return None
    value = field.contents()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        if len(value) > 6:
            return "[" + ", ".join(scalar(v) for v in value[:6]) + ", ...]"
        return "[" + ", ".join(scalar(v) for v in value) + "]"
    return str(value)


def first_value(reader: GGUFReader, keys: list[str]) -> Any:
    for key in keys:
        value = field_value(reader, key)
        if value is not None:
            return value
    return None


def metadata_for(path: Path) -> dict[str, str]:
    reader = GGUFReader(path)
    arch = scalar(field_value(reader, "general.architecture"))
    prefix = arch if arch else "<unknown>"

    fields = reader.fields.keys()
    ssm_keys = [key for key in fields if ".ssm." in key or key.endswith(".ssm_conv_kernel")]
    sliding_keys = [key for key in fields if "sliding" in key or key.endswith(".attention.window_size")]
    mla_keys = [key for key in fields if "lora_rank" in key or "kv_lora" in key or "mla" in key]

    expert_count = first_value(reader, [
        f"{prefix}.expert_count",
        f"{prefix}.feed_forward.expert_count",
        "general.expert_count",
    ])
    expert_used = first_value(reader, [
        f"{prefix}.expert_used_count",
        f"{prefix}.feed_forward.expert_used_count",
        "general.expert_used_count",
    ])

    key_len = first_value(reader, [
        f"{prefix}.attention.key_length",
        f"{prefix}.attention.key_length.full",
    ])
    value_len = first_value(reader, [
        f"{prefix}.attention.value_length",
        f"{prefix}.attention.value_length.full",
    ])
    inferred_dim = False
    if key_len is None and value_len is None:
        embedding_len = first_value(reader, [f"{prefix}.embedding_length", "general.embedding_length"])
        head_count = first_value(reader, [f"{prefix}.attention.head_count"])
        try:
            if embedding_len is not None and head_count is not None and int(head_count) > 0:
                inferred = int(embedding_len) // int(head_count)
                if inferred * int(head_count) == int(embedding_len):
                    key_len = inferred
                    value_len = inferred
                    inferred_dim = True
        except (TypeError, ValueError):
            pass

    notes: list[str] = []
    try:
        key_len_int = int(key_len) if key_len is not None else None
        value_len_int = int(value_len) if value_len is not None else None
    except (TypeError, ValueError):
        key_len_int = None
        value_len_int = None

    if key_len_int is not None or value_len_int is not None:
        if key_len_int != value_len_int:
            notes.append("k/v-dim-mismatch")
        elif key_len_int == 256:
            notes.append("primary-256")
        elif key_len_int == 128:
            notes.append("regression-128")
        else:
            notes.append("unsupported-kv-dim")
        if inferred_dim:
            notes.append("dim-inferred")

    if arch in {"gemma2", "gemma3", "gemma3n", "gemma4", "exaone4", "openai_moe", "mimo2"} or sliding_keys:
        notes.append("swa/iswa-likely")
    if ssm_keys:
        notes.append("hybrid-ssm")
    if mla_keys:
        notes.append("mla-likely")
    if expert_count not in (None, 0, "0"):
        notes.append("moe")

    return {
        "path": str(path),
        "size_gib": f"{path.stat().st_size / (1024 ** 3):.2f}",
        "arch": arch,
        "layers": scalar(first_value(reader, [f"{prefix}.block_count", "general.block_count"])),
        "ctx": scalar(first_value(reader, [f"{prefix}.context_length", "general.context_length"])),
        "heads": scalar(first_value(reader, [f"{prefix}.attention.head_count"])),
        "kv_heads": scalar(first_value(reader, [f"{prefix}.attention.head_count_kv"])),
        "key_dim": scalar(key_len),
        "value_dim": scalar(value_len),
        "experts": scalar(expert_count),
        "experts_used": scalar(expert_used),
        "ssm_keys": str(len(ssm_keys)),
        "sliding_keys": str(len(sliding_keys)),
        "notes": ",".join(notes),
    }


def iter_models(paths: list[Path]) -> list[Path]:
    models: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".gguf":
            models.append(path)
        elif path.is_dir():
            models.extend(sorted(path.rglob("*.gguf")))
    return sorted(set(models))


def print_table(rows: list[dict[str, str]], columns: list[str]) -> None:
    widths = {col: max(len(col), *(len(row[col]) for row in rows)) for col in columns}
    print(" | ".join(col.ljust(widths[col]) for col in columns))
    print("-|-".join("-" * widths[col] for col in columns))
    for row in rows:
        print(" | ".join(row[col].ljust(widths[col]) for col in columns))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize local GGUF metadata relevant to KVarN validation.")
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF files or directories to scan")
    parser.add_argument("--contains", default="", help="case-insensitive substring filter for path")
    args = parser.parse_args()

    models = iter_models(args.paths)
    if args.contains:
        needle = args.contains.lower()
        models = [path for path in models if needle in str(path).lower()]

    rows: list[dict[str, str]] = []
    for path in models:
        try:
            rows.append(metadata_for(path))
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "path": str(path),
                "size_gib": "",
                "arch": "",
                "layers": "",
                "ctx": "",
                "heads": "",
                "kv_heads": "",
                "key_dim": "",
                "value_dim": "",
                "experts": "",
                "experts_used": "",
                "ssm_keys": "",
                "sliding_keys": "",
                "notes": f"error:{exc}",
            })

    if not rows:
        return 1

    print_table(rows, [
        "size_gib",
        "arch",
        "layers",
        "ctx",
        "heads",
        "kv_heads",
        "key_dim",
        "value_dim",
        "experts",
        "experts_used",
        "ssm_keys",
        "sliding_keys",
        "notes",
        "path",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
