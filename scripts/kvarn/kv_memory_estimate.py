#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass


def fmt_bytes(n: int) -> str:
    return f"{n} ({n / (1024 ** 2):.2f} MiB)"


@dataclass(frozen=True)
class Estimate:
    fp16_sink_tail: int
    body_packed: int
    scales: int

    @property
    def total(self) -> int:
        return self.fp16_sink_tail + self.body_packed + self.scales


def packed_nbytes(n_values: int, bits: int) -> int:
    return (n_values * bits + 7) // 8


def estimate(
    layers: int,
    kv_heads: int,
    head_dim: int,
    ctx: int,
    group: int,
    key_bits: int,
    value_bits: int,
    sink_tokens: int,
    tail_tokens: int,
) -> Estimate:
    n_sink_tail = min(ctx, sink_tokens + tail_tokens)
    n_body = max(0, ctx - n_sink_tail)
    records = (n_body + group - 1) // group

    k_body = packed_nbytes(head_dim * group, key_bits)
    v_body = packed_nbytes(head_dim * group, value_bits)
    k_scale_floats = 2 * head_dim + group
    v_scale_floats = head_dim + 2 * group

    fp16_sink_tail = layers * kv_heads * n_sink_tail * (head_dim + head_dim) * 2
    body_packed = layers * kv_heads * records * (k_body + v_body)
    scales = layers * kv_heads * records * (k_scale_floats + v_scale_floats) * 4

    return Estimate(fp16_sink_tail=fp16_sink_tail, body_packed=body_packed, scales=scales)


def run_self_test() -> None:
    est128 = estimate(
        layers=2,
        kv_heads=4,
        head_dim=128,
        ctx=512,
        group=128,
        key_bits=4,
        value_bits=2,
        sink_tokens=128,
        tail_tokens=128,
    )
    assert est128.fp16_sink_tail == 1048576
    assert est128.body_packed == 196608
    assert est128.scales == 49152
    assert est128.total == 1294336

    est256 = estimate(
        layers=2,
        kv_heads=4,
        head_dim=256,
        ctx=512,
        group=128,
        key_bits=4,
        value_bits=2,
        sink_tokens=128,
        tail_tokens=128,
    )
    assert est256.fp16_sink_tail == 2097152
    assert est256.body_packed == 393216
    assert est256.scales == 73728
    assert est256.total == 2564096


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate logical KVarN KV cache memory using the runtime formula.")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--ctx", type=int, help="allocated KV cache cells, matching llama.cpp -c/metadata cache size")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--sink-tokens", type=int, default=128)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--self-test", action="store_true", help="verify against C++ test-kvarn-kv reference totals")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        if args.layers is None and args.kv_heads is None and args.head_dim is None and args.ctx is None:
            print("KVarN memory estimator self-test: PASS")
            return

    if args.layers is None or args.kv_heads is None or args.head_dim is None or args.ctx is None:
        raise SystemExit("layers, kv-heads, head-dim, and ctx are required unless only --self-test is requested")
    if args.layers <= 0 or args.kv_heads <= 0 or args.head_dim <= 0 or args.ctx <= 0:
        raise SystemExit("layers, kv-heads, head-dim, and ctx must be positive")
    if args.group <= 0 or args.key_bits <= 0 or args.value_bits <= 0:
        raise SystemExit("group, key-bits, and value-bits must be positive")
    if args.key_bits > 8 or args.value_bits > 8:
        raise SystemExit("key-bits and value-bits must be <= 8")
    if args.sink_tokens < 0 or args.tail_tokens < 0:
        raise SystemExit("sink-tokens and tail-tokens must be non-negative")

    records = math.ceil(max(0, args.ctx - min(args.ctx, args.sink_tokens + args.tail_tokens)) / args.group)
    fp16_total = args.layers * args.kv_heads * args.ctx * args.head_dim * 2 * 2
    lowbit_total = args.layers * args.kv_heads * args.ctx * args.head_dim * (args.key_bits + args.value_bits) // 8
    est = estimate(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        ctx=args.ctx,
        group=args.group,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        sink_tokens=args.sink_tokens,
        tail_tokens=args.tail_tokens,
    )

    print(f"FP16 full-context total: {fmt_bytes(fp16_total)}")
    print(f"approximate {args.key_bits}-bit K / {args.value_bits}-bit V full-context total: {fmt_bytes(lowbit_total)}")
    print(f"KVarN FP16 sink/tail: {fmt_bytes(est.fp16_sink_tail)}")
    print(f"KVarN packed body: {fmt_bytes(est.body_packed)}")
    print(f"KVarN scales: {fmt_bytes(est.scales)}")
    print(f"KVarN total: {fmt_bytes(est.total)}")
    print(f"KVarN body records per layer/head: {records}")
    print(f"KVarN amortized bytes/token/layer/head: {est.total / args.ctx / args.layers / args.kv_heads:.6f}")


if __name__ == "__main__":
    main()
