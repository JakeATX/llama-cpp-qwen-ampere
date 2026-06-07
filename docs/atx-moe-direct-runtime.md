# ATX MoE Direct Runtime

This fork exposes a mixed MoE residency path for CUDA builds. The stable surface is:

- `--moe-residency-mode off|layer|exact-v1|direct|hybrid|auto`
- `--moe-residency-policy FILE`
- `--moe-residency-stats FILE`
- `--moe-direct-require`
- `--moe-direct-strict-hot-no-stage`
- `--moe-prewarm-experts off|lazy|eager`
- `--moe-pin-cpu-experts auto|on|off`
- `--moe-cold-copy-mode cpu-id-readback|device-bitset|device-ranges|auto`
- `--moe-policy-output FILE`
- `--moe-residency-trace FILE`
- `--moe-residency-debug`

`exact-v1` preserves the old staged-resident path for comparison. `direct`, `hybrid`, and `auto` enable the CUDA direct hot-cache for selected hot layer-expert cells. Under `--moe-direct-strict-hot-no-stage`, the scheduler aborts if a hot expert is included in a host staging range. Under `--moe-direct-require`, unsupported direct tensors abort instead of silently falling back.

Current CUDA direct support is focused on Unsloth Qwen3.6 MoE K-quants: UD-Q4_K_M, UD-Q4_K_XL, UD-Q4, UD-Q5_K_XL, UD-Q6_K_M, and UD-Q6_K_XL. TurboQuant is intentionally outside the completion target.

Use `scripts/atx_moe_bench.py` for focused mode comparisons and `scripts/atx_moe_cuda_acceptance.py` for acceptance matrix output under `runs/atx_moe_direct/cuda_acceptance`.
