# ATX MoE Bottleneck-First Runtime

## Status

This branch now has runtime telemetry and tooling for bottleneck-first MoE residency experiments. The current implementation does not yet meet the 80 tok/s gate. The latest short server smoke shows why: reducing host bytes alone can lower throughput when it also worsens MTP acceptance or increases cold range fragmentation.

## Runtime Additions

- `--moe-cold-coalesce-gap N`
  - Allows cold expert copy ranges to bridge up to `N` unused non-hot experts.
  - Default should remain `0`. A gap of `2` reduced range count in a smoke run but copied far more bytes and slowed decode.
- Residency stats now include:
  - `host_copy_submit_ns`
  - `max_host_range_slices`
  - `coalesced_unused_gap_slices`
  - prompt/decode split counters for used, cold-miss, and direct-hit slices
  - ranked `bottleneck_layers`
- Policy JSON may use `bottleneck_keep_layers` as an alias for whole-layer MoE promotion.

## Compiler Additions

`scripts/atx_moe_policy_compile.py` can now consume residency stats:

```powershell
python scripts\atx_moe_policy_compile.py `
  --policy-source "C:\Users\sjake\OneDrive\Documents\New project\results\hf_policies_check" `
  --out-dir runs\atx_moe_bottleneck\policies `
  --mode auto `
  --stats-json runs\atx_moe_bottleneck\known_fast_b13_stats_probe\stats\known_fast_tail_layers.residency.json `
  --base-keep-layers 25-28,31-39 `
  --attention-baseline-layers 3,7,11,15,19,23,27,31,35,39
```

It emits:

- `bottleneck_top_N_layers.atx.json`
- `bottleneck_base_preserve_N_layers.atx.json`
- `bottleneck_swap_in_X_drop_Y.atx.json` when `--swap-candidates` is set
- `*.bottleneck_first.atx.json`
- `attention_layer_baseline.atx.json`
- `known_fast_base_layers.atx.json`

## Server Acceptance Harness

`scripts/atx_moe_bottleneck_acceptance.ps1` runs server/MTP-shaped tests matching the historical ATX benchmark family:

- `llama-server`
- Qwen3.6-35B-A3B MTP GGUF
- 64K context by default
- Q8 KV
- `--spec-type mtp --spec-draft-n-max 2`
- whole-layer and bottleneck policies
- graceful `/shutdown` so residency stats are written

## Latest Short Smoke

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\atx_moe_bottleneck_acceptance.ps1 `
  -Scenarios bottleneck-auto `
  -BottleneckPolicyPattern bottleneck_top_13_layers.atx.json `
  -MaxTokens 32 `
  -Context 64000 `
  -OutDir runs\atx_moe_bottleneck\bottleneck_top13_probe2
```

Results:

| scenario | decode tok/s | host bytes | ranges | single ranges | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| known_fast_tail_layers | 51.32 | 5.57 GB | 4899 | 2772 | 0.692 |
| bottleneck_auto top13 | 47.20 | 4.91 GB | 4851 | 3039 | 0.607 |

Conclusion: top-by-host-byte bottleneck ranking reduced host traffic by about 12%, but it selected a layer set with worse MTP acceptance and more single-copy fragmentation, so decode slowed. The next policy scorer should include measured tok/s/MTP acceptance and penalize fragmentation, not just host bytes.

A one-layer swap probe (`bottleneck_swap_in_40_drop_25`) also regressed: 53.17 tok/s for the base tail set vs 46.11 tok/s for the swap, with MTP acceptance falling from 0.692 to 0.607. This makes the MTP acceptance penalty concrete.

## 20-Iteration Search

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\atx_moe_bottleneck_iterate.ps1 `
  -MaxIterations 20 `
  -MaxHours 120 `
  -MaxTokens 64 `
  -Context 64000 `
  -OutDir runs\atx_moe_bottleneck\iterate_20_live
```

Best candidate:

- Policy: `policies/atx_bottleneck_swap_in_0_drop_36_q4kxl_64k.atx.json`
- Layers: `0,25,26,27,28,31,32,33,34,35,37,38,39`
- Decode: `62.68 tok/s`
- Baseline in same run: `55.80 tok/s`
- Improvement: `1.12x`
- Host bytes: `5.43 GB` vs baseline `5.57 GB`
- MTP acceptance: `0.76`, same as baseline

This confirms that one measured bottleneck layer, layer `0`, helps when it replaces some tail layers, with the best tested replacement being layer `36`. It still does not reach the 80 tok/s gate.

## Next Patch Direction

The most promising path is a throughput-aware layer search:

- keep the proven 13-layer tail set as the production baseline candidate
- test one-layer swaps from the ranked bottleneck list into that fixed 13-layer budget
- score by median decode tok/s first, then host bytes/token and MTP acceptance
- reject swaps that reduce MTP acceptance or increase single-range ratio
- only enable exact-hot direct residency inside layers that already have low cold traffic
