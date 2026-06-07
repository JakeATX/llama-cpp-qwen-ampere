# ATX llama.cpp Fork

This local fork is based on the QuinsZouls TurboQuant llama.cpp fork at commit `1e5a46dccb8dd9b8d52817030cf4e334f76a8725`.

## Added Controls

- Exact arbitrary MoE layer residency:
  - `--moe-cpu-layers`
  - `--moe-gpu-layers`
  - `--moe-keep-layers`
- Packed-GGUF compatibility helper for heat-map layer-expert recommendations:
  - `--moe-promote-layer-experts`
- Exact expert and layer-expert residency through the ATX scheduler expert-slice cache:
  - `--moe-keep-experts`
  - `--moe-keep-layer-experts`
  - `--moe-residency-policy`
  - `--moe-residency-stats`

## Preserved

- Quins Qwen3.6/Qwen3.5 MoE trunk support.
- Quins MTP draft override path via `--spec-type mtp`.

## Validation

Validated locally on Qwen3.6-35B-A3B GGUFs:

- `UD-Q4_K_XL` deterministic no-policy vs `--moe-keep-experts 0-31` smoke matched output text.
- `UD-Q4_K_XL` layer-expert smoke produced nonzero resident cache hits.
- `UD-Q4_K_M` MTP smoke passed with `--parallel 1 --spec-type mtp`; logs included `MTP draft head registered`.
- `runs/atx_expert_residency/policy_matrix.json` was regenerated and validated (parquet optional via local pyarrow shim).

Notes:

- Expert residency intentionally disables CPU weight repacking so packed MoE source tensors remain host-visible to the selective scheduler copy path.
- MTP server runs require `--parallel 1`.

Metal direct decode (mul_mv_id + expert_map) and prompt/prefill matrix path (mul_mm_id) are wired on Apple Silicon when `--moe-residency-mode direct|hybrid|auto` is used with `--moe-direct-require`.

Autonomous Metal iteration:

- Server harness (source of truth): `scripts/atx_moe_metal_server_acceptance.py`
- Orchestrator: `scripts/atx_moe_autonomous_loop.py` via `scripts/atx_moe_session.sh`
- Layer profiler: `--layer-profile` (see `docs/atx-layer-bottleneck-profiler.md`)
- Iteration log: `docs/atx-moe-metal-iteration-log.md`
- Baseline artifacts: `runs/atx_moe_metal/autonomous/iter_000_baseline/`
- CPU-MoE/offload checkpoint: `runs/atx_moe_metal/autonomous/iter_012/`
- Corrected full-Metal parity checkpoint: `runs/atx_moe_metal/clean_bench/qwen35_a3b_q4km_true_metal_baseline_promptx16_1024tok/`

Current M4 Max production-shape result:

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Server flags: `--spec-type mtp --parallel 1 --ctx-size 64000 -ctk q8_0 -ctv q8_0`
- Policy: whole-layer MoE residency `keep_layers: 0-40`
- Decode: **89.22 tok/s** against a true full-Metal reference of **87.22 tok/s** in the corrected long run.
- Prefill: **929.69 tok/s** against a true full-Metal reference of **921.40 tok/s** in the corrected long run.
- MTP acceptance: `629/788 = 0.79822`, matching reference for the same prompt/run length.
- Hybrid direct proof: `direct_kernel_dispatch_mmvq=1122`, `direct_kernel_dispatch_mmq=66`, `resident_staging_copy_calls=0`

Apple Metal note: the large `iter_012` speedup was versus a `-ncmoe 34` CPU-MoE/offload reference, not versus a true full-Metal baseline. With true full Metal, the ATX/Kvarn policy is roughly parity on Apple unified memory. CUDA/discrete-memory systems need a separate validation matrix.

See:

- `docs/atx-moe-residency.md`
- `docs/atx-exact-expert-residency-blocker.md`
- `docs/atx-runs.md`
- `scripts/atx_moe_metal_acceptance.py` (CLI smoke)
- `scripts/atx_moe_metal_server_acceptance.py` (server MTP 64K)
- `scripts/atx_moe_session.sh`
