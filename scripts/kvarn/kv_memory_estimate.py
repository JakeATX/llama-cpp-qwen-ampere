#!/usr/bin/env python3
import argparse
import math


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate FP16, simple low-bit, and KVarN KV cache memory.")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=2)
    args = parser.parse_args()

    records = math.ceil(args.ctx / args.group)
    k_body = math.ceil(args.head_dim * args.group * args.key_bits / 8)
    v_body = math.ceil(args.head_dim * args.group * args.value_bits / 8)
    scale = ((2 * args.head_dim + args.group) + (args.head_dim + 2 * args.group)) * 4
    kvarn_record = k_body + v_body + scale

    fp16_total = args.layers * args.kv_heads * args.ctx * args.head_dim * 2 * 2
    lowbit_total = args.layers * args.kv_heads * args.ctx * args.head_dim * (args.key_bits + args.value_bits) // 8
    kvarn_total = args.layers * args.kv_heads * records * kvarn_record

    print(f"FP16 total: {fp16_total}")
    print(f"approximate {args.key_bits}-bit K / {args.value_bits}-bit V body-only total: {lowbit_total}")
    print(f"KVarN total: {kvarn_total}")
    print(f"KVarN amortized bytes/token/layer/head: {kvarn_total / args.ctx / args.layers / args.kv_heads:.6f}")


if __name__ == "__main__":
    main()
