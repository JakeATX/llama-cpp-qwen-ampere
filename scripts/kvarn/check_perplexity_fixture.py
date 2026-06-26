#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


CHAT_MARKER_RE = re.compile(r"<\|[^>\s|]+\|>|<\|[^>\s|]+>|<[^>\s|]+\|>|<[^>\s|]+_of_[^>\s]+>")


def read_gguf_fields(model: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "gguf-py"))
    try:
        from gguf import GGUFReader
    except Exception as exc:  # pragma: no cover - environment failure path
        raise SystemExit(f"failed to import gguf reader from {repo_root / 'gguf-py'}: {exc}") from exc

    reader = GGUFReader(str(model))
    fields: dict[str, object] = {}
    for field in reader.fields.values():
        if field.name in {
            "tokenizer.chat_template",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.suppress_tokens",
        }:
            fields[field.name] = field.contents()
    return fields


def field_as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if value is None:
        return ""
    return str(value)


def field_as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def field_as_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (bytes, str)):
        return []
    try:
        return [int(v) for v in value]  # numpy arrays and GGUF array fields
    except TypeError:
        return []


def tokenize_ids(tokenizer: Path, model: Path, dataset: Path, parse_special: bool) -> list[int]:
    argv = [
        str(tokenizer),
        "-m", str(model),
        "-f", str(dataset),
        "--ids",
        "--show-count",
        "--log-disable",
    ]
    if not parse_special:
        argv.append("--no-parse-special")

    try:
        proc = subprocess.run(argv, check=False, text=True, capture_output=True)
    except OSError as exc:
        raise SystemExit(f"failed to run tokenizer fixture check: {exc}") from exc

    if proc.returncode != 0:
        raise SystemExit(
            "tokenizer fixture check failed with exit code "
            f"{proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    match = re.search(r"\[[\s\S]*?\]", proc.stdout)
    if match is None:
        raise SystemExit("tokenizer fixture check did not emit a token id list")
    try:
        ids = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError) as exc:
        raise SystemExit(f"failed to parse tokenizer id list: {exc}") from exc
    if not isinstance(ids, list) or not all(isinstance(token_id, int) for token_id in ids):
        raise SystemExit("tokenizer id output was not a list of integers")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity check for KVarN perplexity fixtures.")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None, help="Optional GGUF model; enables chat-template marker compatibility checks.")
    ap.add_argument("--tokenizer-exe", type=Path, default=None,
                    help="Optional llama-tokenize executable; enables model-tokenizer unknown-token checks.")
    ap.add_argument("--max-unk-rate", type=float, default=0.001,
                    help="Maximum allowed literal <unk> occurrences per whitespace-delimited word; set negative to disable.")
    ap.add_argument("--max-token-unk-rate", type=float, default=0.001,
                    help="Maximum allowed tokenizer unknown-token ID rate; set negative to disable.")
    ap.add_argument("--max-suppressed-token-rate", type=float, default=0.0,
                    help="Maximum allowed model-suppressed token ID rate; set negative to disable.")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="Minimum tokenized length required by the planned perplexity run; set 0 to disable.")
    ap.add_argument("--parse-special", action="store_true",
                    help="Parse special tokens during tokenizer validation, matching llama-perplexity --parse-special.")
    ap.add_argument("--fail-on-template-mismatch", action="store_true",
                    help="Fail when the dataset contains chat markers that do not appear in the model chat template.")
    ap.add_argument("--allow-chat-markers", action="store_true",
                    help="Allow literal chat/control markers in the fixture. By default they fail because perplexity "
                         "tools may apply -inf logit bias to EOG/control tokens, making absolute PPL unusable.")
    args = ap.parse_args()

    text = args.dataset.read_text(encoding="utf-8", errors="replace")
    words = max(1, len(text.split()))
    literal_unk = text.count("<unk>")
    unk_rate = literal_unk / words

    failures: list[str] = []
    warnings: list[str] = []
    if args.max_unk_rate >= 0.0 and unk_rate > args.max_unk_rate:
        failures.append(f"literal <unk> rate {unk_rate:.6f} exceeds --max-unk-rate {args.max_unk_rate:.6f}")

    markers = sorted(set(CHAT_MARKER_RE.findall(text)))
    fields: dict[str, object] = {}
    template = ""
    missing_markers: list[str] = []
    if markers and not args.allow_chat_markers:
        failures.append(
            "dataset contains literal chat/control markers; use a plain-text fixture for PPL/KL gates or pass "
            "--allow-chat-markers only when you have separately proven these tokens are not logit-biased labels: "
            + ", ".join(markers[:16])
        )
    if args.model is not None:
        fields = read_gguf_fields(args.model)
        template = field_as_text(fields.get("tokenizer.chat_template"))
        missing_markers = [marker for marker in markers if marker not in template]
        if missing_markers:
            msg = "dataset chat markers absent from model chat template: " + ", ".join(missing_markers[:16])
            if args.fail_on_template_mismatch:
                failures.append(msg)
            else:
                warnings.append(msg)

    token_count = 0
    token_unk_count = 0
    token_unk_rate = 0.0
    token_unk_id = None
    suppress_ids: list[int] = []
    suppressed_count = 0
    suppressed_rate = 0.0
    if args.tokenizer_exe is not None:
        if args.model is None:
            failures.append("--tokenizer-exe requires --model")
        elif not args.tokenizer_exe.exists():
            failures.append(f"tokenizer executable not found: {args.tokenizer_exe}")
        else:
            token_unk_id = field_as_int(fields.get("tokenizer.ggml.unknown_token_id"))
            if token_unk_id is None:
                warnings.append("model has no tokenizer.ggml.unknown_token_id field; tokenizer unknown-token rate not checked")
            else:
                ids = tokenize_ids(args.tokenizer_exe, args.model, args.dataset, args.parse_special)
                token_count = len(ids)
                if args.min_tokens > 0 and token_count < args.min_tokens:
                    failures.append(
                        f"tokenized fixture length {token_count} is less than --min-tokens {args.min_tokens}; "
                        "llama-perplexity will not produce a valid PPL/KL baseline for this context/chunk setting"
                    )
                token_unk_count = sum(1 for token_id in ids if token_id == token_unk_id)
                token_unk_rate = token_unk_count / max(1, token_count)
                if args.max_token_unk_rate >= 0.0 and token_unk_rate > args.max_token_unk_rate:
                    failures.append(
                        f"tokenizer unknown-token rate {token_unk_rate:.6f} exceeds "
                        f"--max-token-unk-rate {args.max_token_unk_rate:.6f}"
                    )
                suppress_ids = field_as_int_list(fields.get("tokenizer.ggml.suppress_tokens"))
                if suppress_ids:
                    suppress_set = set(suppress_ids)
                    suppressed_count = sum(1 for token_id in ids if token_id in suppress_set)
                    suppressed_rate = suppressed_count / max(1, token_count)
                    if args.max_suppressed_token_rate >= 0.0 and suppressed_rate > args.max_suppressed_token_rate:
                        failures.append(
                            f"tokenizer suppressed-token rate {suppressed_rate:.6f} exceeds "
                            f"--max-suppressed-token-rate {args.max_suppressed_token_rate:.6f}; "
                            "llama-perplexity applies -inf logit bias to suppressed tokens, so this fixture cannot "
                            "produce a trustworthy absolute PPL"
                        )

    print(f"fixture={args.dataset}")
    if args.model is not None:
        print(f"model={args.model}")
    print(f"words={words} literal_unk={literal_unk} unk_rate={unk_rate:.6f}")
    print("chat_markers=" + (", ".join(markers) if markers else "(none)"))
    if args.model is not None:
        print(f"template_present={bool(template)}")
        print("missing_template_markers=" + (", ".join(missing_markers) if missing_markers else "(none)"))
    if args.tokenizer_exe is not None:
        print(f"tokenizer={args.tokenizer_exe}")
        print(f"parse_special={args.parse_special}")
        print(f"unk_token_id={token_unk_id if token_unk_id is not None else '(none)'}")
        print(f"tokens={token_count} tokenizer_unk={token_unk_count} tokenizer_unk_rate={token_unk_rate:.6f}")
        print(f"suppress_token_ids=" + (", ".join(str(x) for x in suppress_ids) if suppress_ids else "(none)"))
        print(f"tokenizer_suppressed={suppressed_count} tokenizer_suppressed_rate={suppressed_rate:.6f}")
    for warning in warnings:
        print("WARNING: " + warning)
    for failure in failures:
        print("FAIL: " + failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
