# ATX Metal MoE iteration log

Production target: `llama-server` on M4 Max with Qwen3.6-35B-A3B MTP Q4_K_M, `--parallel 1 --ctx-size 64000`, hybrid/layer+bottleneck policies.

## iter_000 — measurement harness + baseline matrix

- Added `scripts/atx_moe_metal_server_acceptance.py` (llama-server MTP 64K, `/shutdown` stats flush).
- Fixed `scripts/atx_moe_bench.py` to use non-interactive `llama-completion --simple-io`.
- Added `scripts/atx_moe_autonomous_loop.py` and upgraded `scripts/atx_moe_session.sh`.
- Metal `kernel_mul_mm_id` direct path wired (`direct_kernel_dispatch_mmq`); removed prompt-phase staging fallback on Metal direct hits.
- Eager expert prewarm (`--moe-prewarm-experts eager`) hydrates resident slices at cache creation; GPU-weight hydration uses `ggml_view_1d` + `ggml_backend_tensor_copy`.
- Merged layer bottleneck profiler (`common/layer-profile.*`, `--layer-profile*` flags).
- Baseline matrix artifacts: `runs/atx_moe_metal/autonomous/iter_000_baseline/acceptance_summary.json`.

### iter_000 baseline (MTP Q4_K_M, ctx 64000, max_tokens 32)

| Scenario | Decode tok/s | MTP accept | direct_mmvq | Notes |
|----------|-------------:|-----------:|------------:|-------|
| reference | 46.1 | 0.61 | 0 | no policy |
| known_fast_tail | 56.1 | 0.61 | 0 | best layer-only |
| bottleneck_auto | 55.0 | 0.61 | 0 | swap policy |
| exact_v1_10pct | 14.3 | 0.61 | 0 | ~56 GB host_bytes |
| hybrid_top16 | 15.6 | 0.00 | 0 | runs after strict-hot split fix; direct dispatch still pending |

Artifacts: `runs/atx_moe_metal/autonomous/iter_000_baseline/acceptance_summary.json`

Gates (≥80 tok/s stretch, ≥72 minimum, zero hot staging, direct dispatch counters, MTP acceptance) tracked by orchestrator triage in `checkpoint.json`. **Not yet passing** — next iterations focus on nonzero `direct_kernel_dispatch_*` on hybrid and policy search toward 72+ tok/s.

## iter_001–012 — Metal direct unblocked and CPU-MoE/offload gate passed

The direct-cache blocker was the Metal buffer type name check: runtime code matched `Metal`, while the shared Metal backend names are `MTL0`/`MTL*`. After accepting `MTL` buffer types and keeping direct cache allocation aligned with the selected backend, `hybrid_top16` produced direct dispatch proof (`direct_kernel_dispatch_mmvq=1122`, `direct_kernel_dispatch_mmq=66`) with `resident_staging_copy_calls=0`, `hot_staging_bytes=0`, and MTP restored to reference (`17/28 = 0.60714`). Hybrid policies still copied 33-57 GB of cold experts and stayed below 26 tok/s, so policy search pivoted to whole-layer residency. The `keep_layers: 0-40` policy reached **76.69 tok/s** at 64K in `runs/atx_moe_metal/autonomous/iter_012/`; that checkpoint compared against a `-ncmoe 34` CPU-MoE/offload reference, so it should not be read as a true full-Metal speedup.

## corrected Metal baseline — full-Metal parity

The corrected long benchmark in `runs/atx_moe_metal/clean_bench/qwen35_a3b_q4km_true_metal_baseline_promptx16_1024tok/` disables `-ncmoe` for the reference scenario. On the same Qwen3.6-35B-A3B MTP Q4_K_M model, true full-Metal reference reached **87.22 tok/s** decode and **921.40 tok/s** prefill; `keep_layers: 0-40` reached **89.22 tok/s** decode and **929.69 tok/s** prefill, with identical MTP acceptance (`629/788`). The honest Apple Silicon result is therefore parity with full Metal, not a major speedup. CUDA remains the next target for validating whether layer/expert residency helps on discrete-memory systems.
