# ATX Exact Expert Residency Status

## Status

This document is retained as the design record for the original packed-GGUF blocker.

The blocker has been addressed for V1 by adding an ATX scheduler expert-slice cache. The following flags now parse and run:

- `--moe-keep-experts`
- `--moe-keep-layer-experts`
- `--moe-residency-policy`
- `--moe-residency-stats`

The implementation preserves packed GGUF files and copies routed slices through the existing `ggml_mul_mat_id` scheduler path. Selected expert slices are hydrated into a durable Metal/shared-buffer cache on first use and reused on later routed hits.

Performance update:

- V1 exact expert residency remains a memory-compatibility path, not the final fast path. Resident cache hits are still copied into the packed `ggml_mul_mat_id` staging tensor on each routed use.
- The scheduler now coalesces cold misses into contiguous expert ranges and reports staging, host-copy, resident-copy, sync, and per-layer counters. This reduces tiny host copy calls and makes the transfer bottleneck measurable.
- Whole-layer promotion remains the near-term throughput path because it keeps packed tensors on GPU and lets the existing `ggml_mul_mat_id` kernel consume them directly.
- Exact expert residency should not be called fast until resident hits avoid per-token staging copies or a compact/mixed-source matmul path is implemented.

Validated so far:

- Qwen3.6-35B-A3B `UD-Q4_K_XL` server smoke with `--moe-keep-experts 0-31`
- deterministic text equality versus no-policy baseline under fixed sampling
- nonzero resident cache hits and tensor mappings in `runs/atx_expert_residency/smoke/xl_policy_stats.json`
- layer-expert smoke with `--moe-keep-layer-experts 0:0-31`

Previously blocked, now resolved:

- MTP preservation smoke. The configured local MTP GGUF symlink was broken, but the model path was repaired from the verified Hugging Face cache copy and the smoke now passes with `--parallel 1 --spec-type mtp`. See `runs/atx_expert_residency/checkpoint_06_mtp_BLOCKER.RESOLVED.md`.

## Why This Blocks Exact Experts

Qwen3.6 GGUF stores all routed experts for a layer in three packed tensors:

- `blk.N.ffn_gate_exps.weight`
- `blk.N.ffn_up_exps.weight`
- `blk.N.ffn_down_exps.weight`

Each packed tensor has one backend buffer. llama.cpp tensor buffer overrides can place that whole tensor on Metal/GPU or CPU/RAM, but cannot place individual slices of the tensor on different backends.

The original issue was that the existing CPU-MoE path copied only routed expert slices from CPU/RAM into a GPU-side `mul_mat_id` input copy at runtime. That destination is scheduler scratch memory and cannot itself be treated as persistent residency.

The V1 fix adds a separate cache that is durable across graph executions. Treating the scratch copy as "resident experts" would still be incorrect because it would either:

- be overwritten/reused by later graph allocations; or
- require pinning full packed tensor copies on GPU, which removes the memory benefit of expert-level residency.

## Follow-Up Work

The scheduler-cache V1 is enough to measure exact expert and layer-expert policies on the target Metal machine. Longer-term designs that may improve performance further:

1. Use saliency to promote hot whole layers first, preserving packed tensor execution while selecting layers more intelligently than the fixed attention-spaced baseline.
2. Add compact per-layer hot expert tensors with ID remapping so selected experts can be consumed directly by a backend kernel and only cold misses use staging.
3. Add a split-expert tensor layout or converter so each layer expert can be loaded as its own tensor or as compact resident/nonresident groups.
4. Extend `ggml_mul_mat_id` and the CUDA/Metal backend to accept mixed expert sources with dynamic ID remapping.
5. Add a public backend range-copy API so resident cache slices can be blitted GPU-to-GPU into the temporary `mul_mat_id` input copy without host-mediated `set_tensor` calls.

Any of those paths must preserve:

- Qwen3.6 `qwen35moe` trunk execution;
- Quins `qwen35moe_mtp` draft execution;
- deterministic token equivalence versus the unmodified packed tensor path;
- speed and memory measurements for first-N layers, selected layers, and eventual exact expert policies.
