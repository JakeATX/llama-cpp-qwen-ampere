#!/usr/bin/env python3
"""Summarize worst KL-divergence rows with tokenizer-aligned context.

This is a diagnostic helper for KVarN accuracy gates. It does not prove a
causal layer/head mechanism by itself; it makes the worst failure tokens
concrete enough to decide the next targeted dump.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any


TOKEN_START_RE = re.compile(r"^\s*(-?\d+)\s*->\s*'(.*)$")


def tokenizer_argv(tokenizer: Path, model: Path, dataset: Path, parse_special: bool) -> list[str]:
    argv = [
        str(tokenizer),
        "-m", str(model),
        "-f", str(dataset),
        "--log-disable",
    ]
    # llama-tokenize parses special tokens by default and only exposes the
    # negative form on current builds.
    if not parse_special:
        argv.append("--no-parse-special")
    return argv


def parse_piece_output(text: str) -> list[str]:
    pieces: list[str] = []
    cur_id: int | None = None
    cur_parts: list[str] = []
    for line in text.splitlines():
        if cur_id is not None:
            cur_parts.append(line)
            if line.endswith("'"):
                piece = "\n".join(cur_parts)
                pieces.append(piece[:-1])
                cur_id = None
                cur_parts = []
            continue

        m = TOKEN_START_RE.match(line)
        if not m:
            continue
        rest = m.group(2)
        if rest.endswith("'"):
            pieces.append(rest[:-1])
        else:
            cur_id = int(m.group(1))
            cur_parts = [rest]
    return pieces


def run_tokenizer(tokenizer: Path, model: Path, dataset: Path, parse_special: bool) -> list[dict[str, Any]]:
    ids_argv = tokenizer_argv(tokenizer, model, dataset, parse_special) + ["--ids"]
    ids_proc = subprocess.run(
        ids_argv,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ids_proc.returncode != 0:
        raise SystemExit(
            f"tokenizer --ids failed with exit code {ids_proc.returncode}\n"
            f"argv={ids_argv!r}\nstdout={ids_proc.stdout[-2000:]}\nstderr={ids_proc.stderr[-2000:]}"
        )
    try:
        ids = ast.literal_eval(ids_proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise SystemExit(f"failed to parse tokenizer --ids output: {exc}\n{ids_proc.stdout[-2000:]}") from exc
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise SystemExit("tokenizer --ids output was not a list of integers")

    piece_argv = tokenizer_argv(tokenizer, model, dataset, parse_special)
    piece_proc = subprocess.run(
        piece_argv,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pieces: list[str] = []
    if piece_proc.returncode == 0:
        pieces = parse_piece_output(piece_proc.stdout)
    if len(pieces) != len(ids):
        pieces = [""] * len(ids)

    tokens = [{"id": int(tok), "piece": pieces[i]} for i, tok in enumerate(ids)]
    if not tokens:
        raise SystemExit("tokenizer emitted no token ids")
    return tokens


def read_base_tokens(path: Path) -> tuple[list[int], dict[str, Any]]:
    with path.open("rb") as f:
        magic = f.read(8)
        if magic not in (b"_logits_", b"_logp16_", b"_logp16n"):
            raise SystemExit(f"unsupported KL base magic {magic!r}")
        n_ctx = struct.unpack("<i", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        if n_ctx <= 1 or n_vocab <= 0 or n_chunk <= 0:
            raise SystemExit(f"invalid KL base header n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk}")
        raw = f.read(n_ctx * n_chunk * 4)
        if len(raw) != n_ctx * n_chunk * 4:
            raise SystemExit(f"truncated KL base token table in {path}")
    tokens = list(struct.unpack("<" + "i" * (n_ctx * n_chunk), raw))
    return tokens, {"magic": magic.decode("ascii"), "n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk}


def fallback_piece(tok: int) -> str:
    known = {
        105: "<|turn>",
        106: "<turn|>",
        107: "\\n",
        251: "\\n",
    }
    return known.get(tok, "")


def read_kl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no KL rows in {path}")
    for row in rows:
        for key in ("chunk", "local_index", "logit_pos", "target_pos", "global_logit_index", "global_target_index", "target_token"):
            if key in row and row[key] != "":
                row[key] = int(row[key])
        for key in ("kld", "p_diff"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def token_context(tokens: list[dict[str, Any]], center: int, radius: int) -> dict[str, Any]:
    lo = max(0, center - radius)
    hi = min(len(tokens), center + radius + 1)
    window = []
    for idx in range(lo, hi):
        item = dict(tokens[idx])
        item["index"] = idx
        item["is_target"] = idx == center
        window.append(item)
    text = "".join(item["piece"] for item in window)
    return {"start": lo, "end": hi, "tokens": window, "piece_text": text}


def classify_piece(piece: str) -> list[str]:
    tags: list[str] = []
    if piece == "":
        return ["unknown-piece"]
    if "\n" in piece or piece in {"<0x0A>", "\\n"}:
        tags.append("newline")
    if "<" in piece or ">" in piece or "turn" in piece or piece in {"user", "model"}:
        tags.append("chat/control-ish")
    if piece.strip() == "":
        tags.append("whitespace")
    if piece and all(ch in ".,:;!?-_=*/()[]{}<>|`'\"" for ch in piece.strip()):
        tags.append("punctuation")
    if piece.strip().isdigit():
        tags.append("numeric")
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl-csv", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokenizer-exe", type=Path, required=True)
    ap.add_argument("--base-file", type=Path, default=None,
                    help="Optional llama-perplexity KL base file. When present, its token table is authoritative.")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--context-tokens", type=int, default=12)
    ap.add_argument("--parse-special", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    args = ap.parse_args()

    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if args.context_tokens < 0:
        raise SystemExit("--context-tokens must be non-negative")

    rows = read_kl_rows(args.kl_csv)
    tokenizer_tokens = run_tokenizer(args.tokenizer_exe, args.model, args.dataset, args.parse_special)
    base_info: dict[str, Any] | None = None
    if args.base_file is not None:
        base_ids, base_info = read_base_tokens(args.base_file)
        tokens = []
        for i, tok in enumerate(base_ids):
            piece = ""
            if i < len(tokenizer_tokens) and tokenizer_tokens[i]["id"] == tok:
                piece = tokenizer_tokens[i]["piece"]
            if piece == "":
                piece = fallback_piece(tok)
            tokens.append({"id": int(tok), "piece": piece})
    else:
        tokens = tokenizer_tokens
    worst = sorted(rows, key=lambda r: float(r.get("kld", 0.0)), reverse=True)[:args.top_n]

    out_rows = []
    for rank, row in enumerate(worst, start=1):
        target_index = int(row.get("global_target_index", row.get("target_pos", -1)))
        token_entry = tokens[target_index] if 0 <= target_index < len(tokens) else None
        token_id = token_entry["id"] if token_entry else None
        piece = token_entry["piece"] if token_entry else ""
        csv_target = int(row.get("target_token", -1))
        out_rows.append({
            "rank": rank,
            "kld": float(row.get("kld", 0.0)),
            "p_diff": float(row.get("p_diff", 0.0)),
            "chunk": row.get("chunk"),
            "local_index": row.get("local_index"),
            "logit_pos": row.get("logit_pos"),
            "target_pos": row.get("target_pos"),
            "global_logit_index": row.get("global_logit_index"),
            "global_target_index": target_index,
            "csv_target_token": csv_target,
            "tokenizer_token": token_id,
            "token_match": token_id == csv_target,
            "piece": piece,
            "tags": classify_piece(piece),
            "context": token_context(tokens, target_index, args.context_tokens) if token_entry else None,
        })

    result = {
        "kl_csv": str(args.kl_csv),
        "dataset": str(args.dataset),
        "model": str(args.model),
        "base_file": str(args.base_file) if args.base_file else "",
        "base_info": base_info,
        "token_count": len(tokens),
        "tokenizer_token_count": len(tokenizer_tokens),
        "row_count": len(rows),
        "top_n": args.top_n,
        "context_tokens": args.context_tokens,
        "rows": out_rows,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# KL Spike Drilldown", ""]
        lines.append(f"- KL CSV: `{args.kl_csv}`")
        lines.append(f"- dataset: `{args.dataset}`")
        if args.base_file:
            lines.append(f"- KL base file: `{args.base_file}`")
        if base_info:
            lines.append(f"- KL base: `{base_info}`")
        lines.append(f"- token count: {len(tokens)}")
        lines.append(f"- tokenizer token count: {len(tokenizer_tokens)}")
        lines.append(f"- KL rows: {len(rows)}")
        lines.append("")
        lines.append("| rank | KLD | p_diff | token index | token id | match | piece | tags |")
        lines.append("|---:|---:|---:|---:|---:|:---:|---|---|")
        for item in out_rows:
            piece = str(item["piece"]).replace("|", "\\|").replace("\n", "\\n")
            tags = ",".join(item["tags"])
            lines.append(
                f"| {item['rank']} | {item['kld']:.6f} | {item['p_diff']:.6f} | "
                f"{item['global_target_index']} | {item['tokenizer_token']} | "
                f"{'yes' if item['token_match'] else 'NO'} | `{piece}` | {tags} |"
            )
        lines.append("")
        lines.append("## Contexts")
        for item in out_rows:
            ctx = item["context"]
            if ctx is None:
                continue
            lines.append("")
            lines.append(f"### Rank {item['rank']} token {item['global_target_index']} KLD {item['kld']:.6f}")
            lines.append("")
            lines.append("```text")
            lines.append(ctx["piece_text"])
            lines.append("```")
            lines.append("")
            lines.append("```text")
            lines.append(" ".join(
                f"[{t['index']}:{t['id']}:{'*' if t['is_target'] else ''}{t['piece']!r}]"
                for t in ctx["tokens"]
            ))
            lines.append("```")
        args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.json_out and not args.md_out:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
