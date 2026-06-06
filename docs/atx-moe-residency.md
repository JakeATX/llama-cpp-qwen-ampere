# ATX MoE Residency Controls

This ATX fork is based on the QuinsZouls TurboQuant llama.cpp fork and keeps the existing Qwen3.5/Qwen3.6 MTP path intact.

## Implemented

The following layer flags are routed through llama.cpp tensor buffer overrides and work for packed Qwen MoE GGUF tensors:

- `--moe-cpu-layers 0-9,14`
  - Keep the packed MoE expert tensors for the listed layers in CPU/RAM.
- `--moe-gpu-layers 30-39`
  - Keep the packed MoE expert tensors for the listed layers in the first GPU/VRAM buffer.
- `--moe-keep-layers 30-39`
  - Exact MoE layer residency policy: listed MoE layers stay GPU/VRAM-resident; all other packed MoE expert tensors go to CPU/RAM.
- `--moe-promote-layer-experts 37:103,38:33`
  - Packed-GGUF compatibility helper: promote each listed layer-expert cell to its containing layer, then apply the same exact layer policy as `--moe-keep-layers`.

For Qwen3.6 35B-A3B with one MTP block, trunk MoE layers are usually `0-39`; the MTP block is loaded through the `qwen35moe_mtp` override path and is typically layer `40`. Include `40` in `--moe-keep-layers` if you want the MTP block's MoE expert tensors GPU-resident too.

## Exact Expert Residency

The following flags are implemented through an ATX scheduler expert-slice cache:

- `--moe-keep-experts 0,3-5`
  - Keep the listed expert IDs in a durable Metal/shared-buffer cache across packed MoE layers.
- `--moe-keep-layer-experts 37:103,40:7`
  - Keep exact layer-expert cells in the expert cache.
- `--moe-residency-policy policy.json`
  - Load `keep_layers`, `keep_experts`, and/or `keep_layer_experts` from JSON. `keep_layers` uses the fast packed whole-layer path; exact expert keys use the scheduler cache.
- `--moe-residency-stats stats.json`
  - Write cache hit/miss, host-copy, hydration, and tensor mapping counters.

The implementation keeps the packed GGUF layout unchanged. Expert policy flags force packed MoE tensors into host-visible CPU buffers, then the scheduler routes Qwen `ggml_mul_mat_id` ops to Metal and copies only routed expert slices into the GPU-side temporary input. Selected resident slices are hydrated once into the ATX cache and reused on subsequent routed hits.

The stats file is the source of truth for whether a policy actually hit resident experts. At minimum, a useful policy should show:

- `resident_cache_hit_slices > 0`
- `resident_cache_hydrate_slices > 0`
- `moe_named_node_misses == 0`
- deterministic output equality versus a no-policy baseline under fixed sampling

Current V1 caveats:

- Expert residency disables CPU weight repacking for the model run because the selective MoE copy path requires host-visible source buffers.
- The cache is durable and GPU-visible. On CUDA and Metal decode (`mul_mv_id`), `--moe-residency-mode direct|hybrid` can consume hot experts from the compact cache without staging into `input_cpy`. Prompt-phase `mul_mm_id` on Metal still stages from the resident GPU cache until the matrix path is wired. `exact-v1` still stages hot experts into `input_cpy`.
- Cold misses in the exact-cache path are coalesced into contiguous expert ranges before staging. Older V1 builds copied every cold expert one at a time.
- MTP validation requires `--parallel 1`; the repaired local `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` path passed the smoke with `--spec-type mtp`; see `runs/atx_expert_residency/checkpoint_06_mtp_BLOCKER.RESOLVED.md`.

Additional telemetry fields distinguish memory-saving residency from a fast direct-use path:

- `resident_staging_copy_calls`: resident cache hits copied into `input_cpy`; nonzero means exact experts are still not consumed directly by the matmul kernel.
- `cold_expert_miss_slices` and `cold_expert_miss_range_calls`: cold routed experts that fell back to host staging, with range coalescing.
- `host_expert_range_copy_calls` and `host_expert_single_copy_calls`: copy call shape after coalescing.
- `input_cpy_staging_bytes`: total bytes staged into the packed temporary input from both host and resident cache sources.
- `per_layer`: per-layer used slices, resident hits, cold misses, bytes, and copy call counts.

## Packed Tensor Background

Qwen GGUF stores routed experts as packed per-layer tensors such as:

- `blk.37.ffn_gate_exps.weight`
- `blk.37.ffn_up_exps.weight`
- `blk.37.ffn_down_exps.weight`

Each tensor contains all experts for the layer in one buffer. llama.cpp tensor overrides can still only move the whole packed tensor, but ATX now adds a runtime cache on top of the CPU-MoE routed-slice path so exact expert and layer-expert policies can be measured without converting the GGUF.

## Example

Keep the behaviorally important late MoE layers resident and offload the rest:

```bash
./build-atx-metal/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --spec-type mtp \
  --moe-keep-layers 30-40 \
  --ctx-size 4096 \
  --parallel 1
```

Compatibility helper for heat-map selected layer-expert cells:

```bash
./build-atx-metal/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --moe-promote-layer-experts 37:103,38:33 \
  --ctx-size 4096 \
  --parallel 1
```

Exact expert cache:

```bash
./build-atx-metal/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --moe-keep-experts 0-31 \
  --moe-residency-stats runs/atx_expert_residency/smoke/xl_policy_stats.json \
  --ctx-size 2048 \
  --parallel 1
```

Saliency-promoted whole-layer policy:

```json
{
  "config_id": "hybrid_promote_5",
  "kind": "hybrid_promote_layers",
  "keep_layers": [25, 31, 32, 33, 34, 35, 36, 37, 38, 39]
}
```

```bash
./build-atx-cuda/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --moe-residency-policy hybrid_promote_5.json \
  --ctx-size 4096 \
  --parallel 1
```

Exact layer-expert cache:

```bash
./build-atx-metal/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --moe-keep-layer-experts 0:0-31,37:103 \
  --moe-residency-stats runs/atx_expert_residency/smoke/layer_expert_stats.json \
  --ctx-size 2048 \
  --parallel 1
```

The MTP support is inherited from the Quins path:

- trunk `qwen35moe`
- draft override `qwen35moe_mtp`
- `llama_set_mtp: MTP draft head registered`
