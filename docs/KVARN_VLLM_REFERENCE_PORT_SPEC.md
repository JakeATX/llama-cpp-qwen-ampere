# KVarN vLLM Reference Port And Correctness Patch Spec

This document is a handoff for an external architecture/code agent. It should be
read together with `docs/KVARN_PRODUCTION_PATCH_HANDOFF.md`,
`docs/KVARN_PAPER_FIDELITY_AUDIT.md`, and
`docs/KVARN_NEXT_PATCHES_HANDOVER.md`.

## Current Situation

The latest KVarN branch has fixed a major body-store K layout bug:
`kvarn_transpose_pending_k_head_kernel()` now writes K pending tiles in
channel-major order, `k_tile[d * group_size + g]`. This matches the Huawei/vLLM
KVarN reference and repaired a large part of the ctx4096 quality failure.

However, long-context quality still fails:

| Model / mode | Baseline PPL | KVarN PPL | Increase | Status |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6 MTP, ctx4096, paper-frame, k4v2 | 4.5843 | 5.1155 | 11.59% | FAIL |
| Qwen3.6 MTP, ctx4096, paper-frame, k8v8 | 4.5843 | 5.1320 | 11.95% | FAIL |
| Gemma 4 12B true KVarN+ISWA, ctx4096, k4v2 | 418.2027 | 944.8251 | 125.93% | FAIL |
| Gemma 4 12B true KVarN+ISWA, ctx4096, k8v8 | 418.2027 | 520.7935 | 24.53% | FAIL |

Mainline upstream normal-KV Gemma PPL on the same model/dataset/context is also
`418.2027`, so the Gemma baseline is not branch-specific. The failure is in the
KVarN cache path.

The `k8v8` result is diagnostic only. It is not the intended production
compression target. If 8-bit K/V still fails, the remaining issue is unlikely to
be only low-bit RTN resolution. It points to reference-math mismatch, topology,
ISWA windowing, record mapping, or dequant/attention differences.

## vLLM Reference Sources Reviewed

Reference repository:

`https://github.com/huawei-csl/KVarN`

Key files:

- `vllm/model_executor/layers/quantization/kvarn/config.py`
- `vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`
- `vllm/v1/attention/ops/kvarn_store.py`
- `vllm/v1/attention/ops/kvarn_decode.py`
- `vllm/v1/attention/ops/triton_kvarn_sinkhorn.py`
- `vllm/v1/attention/ops/triton_kvarn_decode.py`
- `kvarn_mla_tile_validate.py`
- `kvarn_mla_decode_ref.py`
- `kvarn_mla_tilepack.py`

Important conclusion: KVarN is not conceptually 128d-only. The reference has a
dense tested 128d path and an MLA/tilepack path with `R = 512`, `GROUP = 128`.
The implementation assumes power-of-two transform dimensions, not a fixed 128d
limit. Qwen3.6 256d and Gemma 512d are valid targets.

## What To Port From vLLM

Do not port the Triton kernels directly. They are coupled to vLLM paged-cache
layout and are not a clean fit for ggml/llama.cpp's sink/body/tail/pending
cache topology.

Port these concepts instead:

1. Independent reference oracle.
2. vLLM-style variance-normalization recipe.
3. Optional group-size presets after the oracle is in place.
4. Any reference-proven layout contracts that are not yet covered by tests.

## Reference Contracts To Preserve

The current llama.cpp branch should continue to follow these contracts:

- Rotate Q before KVarN mixed attention.
- Store body K in channel-major tile layout: `[head_dim, group_size]`.
- Store body V in token-major tile layout: `[group_size, head_dim]`.
- Store both KVarN scale factors.
- Apply dequant scales inside the logical dequant/attention path.
- Unrotate the V-side attention output with the Hadamard transpose.
- Pending-to-body seal path must not double-rotate records already copied from
  the rotated tail under paper-frame mode.
- Direct body-store path must rotate raw K/V input before normalization.

The recently fixed pending-K transpose should not be reverted.

## Patch A - Add An Independent vLLM-Style Oracle

### Problem

Existing correctness gates were structurally blind to the pending-K layout bug
because they compared two KVarN paths that shared the same broken store layout.
Packed-vs-split and packed-vs-scratch NMSE can be zero while both sides are
wrong versus the intended algorithm.

### Required Change

Add an independent KVarN oracle that does not call the production CUDA store,
production CUDA dequant, or production mixed-attention implementation.

Recommended implementation options:

- Preferred: C++/CUDA unit test with CPU reference math in
  `tests/test-kvarn-cuda-dequant.cpp` or a new `tests/test-kvarn-reference.cpp`.
- Acceptable: Python oracle script under `scripts/kvarn/` using PyTorch or
  NumPy, called by a smoke wrapper.

The oracle should implement from first principles:

1. Hadamard rotation, natural-order butterfly, normalized by `1 / sqrt(d)`.
2. K tile canonical layout `[D, G]`.
3. V tile canonical layout `[G, D]`.
4. Variance normalization scale computation.
5. RTN quantization and dequantization.
6. Query rotation.
7. Attention score computation over dequantized rotated K.
8. AV accumulation over dequantized rotated V.
9. Output unrotation.

### Required Coverage

Run deterministic synthetic tests for:

- `head_dim = 128`, `group_size = 128`
- `head_dim = 256`, `group_size = 128`
- `head_dim = 512`, `group_size = 128`
- K/V bit widths: `k4v2`, `k4v4`, `k8v8`
- paper-frame off/on where applicable
- pending seal and direct store paths

The oracle must compare against production output using an independent reference
layout. It must not reuse `kvarn_transpose_pending_k_head_kernel()` or any other
production gather/transpose code to build the expected tile.

### Stop Condition

Do not benchmark new speed patches until this oracle passes for 128d, 256d, and
512d. If the oracle fails, fix the correctness mismatch first.

## Patch B - Add vLLM-Style Log/Std Sinkhorn Behind A Flag

### Problem

The current llama.cpp KVarN normalization is not the vLLM reference recipe.

Current branch behavior:

- Alternating linear RMS normalization.
- Uses final iteration.
- Scale accumulation is linear.
- No best-iteration selection.
- CUDA tests mirror this branch-local RMS math.

vLLM reference behavior:

- Uses standard deviation, not RMS.
- Accumulates scales in log space.
- Clamps measured std.
- Clamps log scales.
- Tracks the best iteration by imbalance metric and returns the best state.
- Uses higher iteration counts in reference paths; config comments indicate
  8 iterations can be lossless versus 16 for tested Qwen paths.

This mismatch is now a high-priority suspect because Qwen ctx4096 does not
recover with k8v8, and Gemma k8v8 still fails by 24.53%.

### Required Change

Implement an opt-in normalization mode:

`LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN=1`

or an equivalent internal experimental setting.

Default behavior must remain unchanged until gates prove the new recipe.

The new mode should:

1. Compute per-row/per-column standard deviation, not RMS.
2. Clamp std values to a reference-compatible finite range.
3. Accumulate row and column scales in log space.
4. Clamp log-scale values.
5. Track imbalance after each iteration.
6. Return the best iteration, not necessarily the last.
7. Preserve invertibility in dequant by applying the exact accumulated scales.

Start with CPU/reference implementation first, then CUDA store kernels.

### Acceptance Tests

Add tests comparing CPU and CUDA log/std Sinkhorn scale/output behavior for:

- `D=128,G=128`
- `D=256,G=128`
- `D=512,G=128`
- random normal data
- outlier-heavy data
- near-constant rows/columns

For synthetic tests, dequantized reconstruction error should be no worse than
the current RMS path at the same bit width unless the test explicitly stresses
RMS-only behavior.

### Accuracy Gate

After implementation, run:

Qwen3.6 MTP:

```powershell
$env:LLAMA_KVARN_ENABLE_PAPER_FRAME="1"
$env:LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN="1"
scripts/kvarn/run_accuracy_gate.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" `
  -Dataset "C:\Users\sjake\OneDrive\Documents\New project\external\terminal-bench\tasks\word2vec-from-scratch\wikitext-data\validation.txt" `
  -ContextSize 4096 -BatchSize 4096 -Chunks 2 `
  -KvarnPreset kvarn_k4v2_g128 -KvarnIters 8 `
  -ExpectedKvarnLayers "3-39:4" `
  -ExtraArgs "-ncmoe","34"
```

Gemma 4 true KVarN+ISWA:

```powershell
$env:LLAMA_KVARN_ENABLE_PAPER_FRAME="1"
$env:LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN="1"
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA="1"
scripts/kvarn/run_accuracy_gate.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf" `
  -Dataset "C:\Users\sjake\OneDrive\Documents\New project\external\terminal-bench\tasks\word2vec-from-scratch\wikitext-data\validation.txt" `
  -ContextSize 4096 -BatchSize 4096 -Chunks 2 `
  -KvarnPreset kvarn_k4v2_g128 -KvarnIters 8 `
  -ExpectedKvarnLayers "5-47:6"
```

If k4v2 still fails, repeat with `k4v4` and `k8v8` to separate low-bit quality
from topology bugs.

## Patch C - Add Boundary-Level Store/Dequant Diagnostics

### Problem

Perplexity tells us quality failed, but not where. The next failure could be:

- direct store versus pending seal divergence
- ISWA window eviction/reuse corruption
- record index/span mapping
- scale application mismatch
- attention order/masking mismatch

### Required Change

Add diagnostic modes that dump or compare one sealed body record at a boundary.

Recommended flag:

`LLAMA_KVARN_DEBUG_BODY_RECORD=<layer>:<record>:<head>`

When set, record enough data to compare:

- raw pending/direct K/V source tile
- rotated tile before VarN
- normalized tile
- quantized packed bytes
- stored scale tensors
- dequantized rotated tile
- final unrotated V contribution for a small query batch

The dump can be binary or CSV under an artifact directory. Keep it bounded and
off by default.

### Required Bisects

Add flags or test harness controls to force:

- all records through direct store
- all records through pending seal where supported
- direct record batching off
- direct record batching on
- ISWA eviction/recycle stress

The purpose is to make ctx4096 failures reproducible on a single body record,
not only through end-to-end PPL.

## Patch D - ISWA-Specific Correctness Audit

### Problem

Gemma true KVarN+ISWA is much worse than Qwen after the pending-K fix. k8v8
improves Gemma strongly but still fails. That suggests Gemma has both a general
KVarN quality issue and an ISWA-specific topology/windowing issue.

### Required Audit Points

Review line-by-line:

- sliding-window eviction from tail/pending into body records
- record reuse and epoch invalidation
- layer-local versus body-global epochs
- K/V rotation state during eviction
- record span mapping under wrapped windows
- sink/tail capacity clamping
- attention mask order for `sink | body | pending | tail`
- Gemma heterogeneous K/V dimensions under FA off

### Required Tests

Add an ISWA unit or smoke test that simulates:

1. Fill sink/tail.
2. Evict tail into pending.
3. Seal pending into body.
4. Advance window until record reuse occurs.
5. Compare dequantized K/V records against an independent CPU reference using
   the expected logical token positions.

Run for Gemma-like dimensions:

- K head dim 512 where applicable
- V padded/cache dim behavior with FA off
- expected layers `5-47:6`

## Patch E - Optional Group-Size Presets

### Problem

The vLLM reference exposes `g64` presets as well as `g128`. Smaller groups may
improve quality because each scale covers fewer tokens, at the cost of more
records/scales and potentially more launch overhead.

### Required Change

Do not implement this before Patch A and Patch B. If quality still fails after
the reference oracle and log/std Sinkhorn, then prototype:

- `kvarn_k4v2_g64`
- `kvarn_k4v4_g64`
- `kvarn_k8v8_g64` only as a diagnostic

This touches graph/cache record sizing and must be tested carefully.

### Acceptance

Use g64 only if it improves ctx4096 quality enough to pass without unacceptable
long-context speed regression.

## Validation Sequence

Before each patch:

```powershell
ctest --test-dir build-kvarn-cuda-static-vs -C Release `
  -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" `
  --output-on-failure

python scripts/kvarn/kv_memory_estimate.py --self-test
```

After correctness-affecting patches:

```powershell
scripts/kvarn/compare_cuda_logits_ref.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" `
  -Context 512 -Batch 512 `
  -ExpectedKvarnLayers "3-39:4" `
  -ExtraArgs "-ncmoe","34" `
  -CheckPackedRepeat -CheckPackedSplit

scripts/kvarn/compare_cuda_logits_ref.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf" `
  -Context 512 -Batch 512 `
  -ExpectedKvarnLayers "5-47:6" `
  -CheckPackedRepeat -CheckPackedSplit
```

Then run ctx4096 accuracy gates before any long-context benchmark claim.

## Success Criteria

Correctness must pass before speed work is accepted:

- Qwen3.6 MTP ctx4096 PPL increase <= 5% versus f16 normal KV.
- Gemma 4 true KVarN+ISWA ctx4096 PPL increase <= 5% versus f16 normal KV.
- Packed-repeat and packed-vs-split NMSE remain zero where expected.
- Independent oracle passes for 128d, 256d, and 512d.
- Expected KVarN layers are observed.

Only after those gates pass should long-context speed patches be evaluated:

- Gemma pp4096/tg4096 against upstream mainline.
- Qwen pp4096/tg4096 against upstream mainline.

## Guidance For 5.5 Pro

The highest-value next patch is not another benchmark optimization. It is:

1. Add the independent vLLM-style oracle.
2. Implement opt-in log/std Sinkhorn.
3. Use the oracle and ctx4096 gates to identify whether the remaining failure is
   normalization quality or ISWA topology.

Treat `k8v8` as a diagnostic. A failing `k8v8` run means the remaining problem is
not merely the intended production bit width.
