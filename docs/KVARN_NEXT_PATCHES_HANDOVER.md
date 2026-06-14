# KVarN next-patches handover (for codex)

Reviewer pass over commit `aafdb00f1` ("kvarn: add long-context diagnostics
and accuracy gate"). The short gate and decode already pass; the open problem
is long context (pp4096 39.7%/61.9%, tg4096 51.7%/65.7%). This note ships one
ready patch and specs the next one.

## Confirmed-good in aafdb00f1

- `mark_body_store` now keys the store epoch on the tensor view-root
  (`ggml-cuda.cu` ~3445) - real cache-invalidation fix.
- Dequant-cache key widened to `(k_body, epoch, head_dim, group_size,
  n_head_kv, format)` via `kvarn_dequant_cache_refill_from` - real stale-hit fix.
- Direct prefill body-store generalized from "first chunk at pos 0" to any
  contiguous chunk with per-record in-bounds gating (`llama-graph.cpp`) - the
  main long-context prefill enabler.

## Patch 1 (included): `scripts/kvarn/run_iters_sweep.ps1`

Highest-probability, zero-code-risk speedup for pp4096. The body store runs
`--kvarn-iters` Sinkhorn passes per (record, head) tile, each pass = 2 kernel
launches; at 4096 the store dominates prefill and iters multiply it linearly.
Sinkhorn converges fast, so iters=2/3 usually matches iters=4 accuracy.

The script drives the existing `run_accuracy_gate.ps1` once per candidate iters
(smallest first) and recommends the smallest value that still passes f16
parity. It is additive and delegates all gate semantics.

Run:

```powershell
scripts/kvarn/run_iters_sweep.ps1 `
    -Model  "C:\...\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" `
    -Dataset "C:\...\wiki.test.raw" `
    -BuildDir "build-kvarn-cuda-static-vs" `
    -ContextSize 4096 -BatchSize 4096 `
    -IterValues 1,2,3,4 -ExtraArgs @('-ncmoe','34') -ExpectedKvarnLayers '3-39:4'
```

Then set production `--kvarn-iters <recommended>` and re-run the parity matrix.
Use `-UseKLDivergence -MaxMeanKL 0.02` for a stricter accuracy bar.
Do this BEFORE Patch 2 - it tells you how much store cost is even on the table.

## Patch 2 (to implement): a genuinely batched body-store kernel

**Problem.** `ggml_cuda_kvarn_store_body_direct_records_minmax` (`kvarn.cu`
~1261) only batches at the graph-node level; on-device it still loops
`for record { for head { transpose + sinkhorn x iters + quantize + finalize } }`,
all reusing one shared scratch (`k_tile`/`v_tile`/`pipeline`), so the per-tile
launch count is unchanged from the per-record path. This is why
`LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH` did not help and stays off. At 4096
this is thousands of tiny serial launches per layer.

**Fix.** Process all `n_records * n_heads` tiles in one grid: one CTA (or small
CTA cluster) per tile, with per-tile scratch, so the whole batch is one launch
per phase instead of per tile.

Concrete steps:

1. **Per-tile scratch.** The current scratch holds one tile's working set. Size
   it to `n_tiles` working sets (the op already receives `n_records`, `n_heads`)
   and have CTA `blockIdx.x = tile` index its own slice
   (`scratch + tile * per_tile_floats`). Update the scratch-floats computation
   in `ggml.c` (`ggml_kvarn_store_kv_body_direct_records`) and the
   `supports_op` assertion in `ggml-cuda.cu` to `n_tiles * per_tile_floats`, and
   the graph-side scratch allocation in `llama-kv-cache-kvarn.cpp` /
   `build_attn` accordingly. Keep the `src_layout == 1` gate.

2. **Grid-strided phase kernels.** The existing phase kernels
   (`kvarn_hadamard_cols/rows_parallel_kernel`, `kvarn_sinkhorn_*_parallel`,
   `kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel`,
   `kvarn_store_*_finalize_scales_kernel`, plus the transpose/gather) index by
   `blockIdx.x` over a single tile. Add a tile dimension: launch with
   `gridDim` covering `n_tiles` (e.g. `blockIdx.y = tile`, or fold tile into
   `blockIdx.x`) and have each block compute its tile's base pointers from the
   record/head strides already passed in. The per-tile math is unchanged, so it
   stays logits-equivalent to the per-record path - verify with the existing
   `test-kvarn-cuda-scratch-ref` / dequant tests at NMSE tolerance, then the
   accuracy gate end-to-end.

3. **Keep correctness invariants.** Call `ggml_cuda_kvarn_mark_body_store` once
   on the view-root (as the current code does post-fix). The Sinkhorn iteration
   count and pack order must match the scalar/per-record path exactly so the
   stored body is bit-compatible with the existing dequant/attention readers.

**Validation for Patch 2:**
- `ctest -R "test-kvarn-cuda|test-kvarn-kv"` green.
- `run_accuracy_gate.ps1` on the production models within threshold.
- Parity matrix pp4096 improves vs aafdb00f1 with
  `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH=1`; flip default on only after both.

## Still open (do not skip)

- **Run the accuracy gate on the production Qwen3.6 MTP and Gemma models**, not
  only the 0.8B smoke. The 0.03% smoke is encouraging evidence the
  Hadamard/rotation handling is actually correct, but it is not yet confirmed
  at production scale/context. This is the cheapest way to retire the rotation
  question for good.
- tg4096 (decode, ~65.7%) is bottlenecked by the generic fused kernel running
  `grid = n_head` CTAs that each serially unpack ~30x128 body tokens. Raising
  decode occupancy (split body records across CTAs per head + reduction) is the
  follow-up after the store path.

## Round 26 Codex update

The `--kvarn-iters` sweep is now diagnostic-only, not an optimization path.

Changes made after applying this handover:

- `scripts/kvarn/run_iters_sweep.ps1` now launches each accuracy-gate candidate
  in a child PowerShell process, stops after the first passing candidate by
  default, and validates both the gate summary status and the numeric metric
  threshold before marking a candidate PASS.
- `scripts/kvarn/run_accuracy_gate.ps1` and the sweep now accept `-Chunks`,
  forwarding to `llama-perplexity --chunks`, so production-model checks can be
  bounded.
- `docs/KVARN_PAPER_FIDELITY_AUDIT.md` records the paper/reference-code audit
  and redirects the next patches toward correctness, true batched VarN, and
  fused dual-scale dequant.

New accuracy evidence:

- Qwen3.6 MTP, context 4096, batch 4096, `--chunks 2`, `-ncmoe 34`, expected
  layers `3-39:4`, artifact
  `artifacts/kvarn-iters-sweep/round26-qwen36-ctx4096-chunks2-fixed`:
  - `iters=1`: PPL increase `45.23%`.
  - `iters=2`: PPL increase `47.21%`.
  - `iters=3`: PPL increase `45.28%`.
  - `iters=4`: PPL increase `46.71%`.
  - No candidate passed `MaxPplIncrease=5%`.
- Gemma 4 12B true KVarN+ISWA, context 4096, batch 4096, `--chunks 2`,
  expected layers `5-47:6`, artifact
  `artifacts/kvarn-accuracy/round26-gemma-ctx4096-chunks2-iters4`:
  - f16 PPL `418.2027`.
  - KVarN PPL `137079490.9999`.
  - Result: FAIL.

Updated priority:

1. First fix long-context f16-vs-KVarN accuracy/fidelity.
2. Then implement true batched records * heads VarN/body-store.
3. Then implement fused dual-scale dequant/attention and long-decode occupancy.
4. Only then revisit `--kvarn-iters`.

## Round 27 Codex update

The next paper-steered implementation slice is now in-tree as an opt-in diagnostic path.

What changed:

- `tests/test-kvarn-cuda-dequant.cpp` now has executable paper-frame checks for
  the Hadamard placement/consumption contract:
  - K attention is checked as `H(q) . K_rot_deq == q . H(K_rot_deq)`.
  - V attention is checked as `H(sum p*V_rot_deq) == sum p*H(V_rot_deq)`.
  - The checks cover 128d, Qwen-shaped 256d, and Gemma-shaped 512d.
- `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH_PHASES=1` adds a true
  records-by-heads batched direct-record body-store phase path for the default
  production format (`k4/v2`, `rtn_quantile=1.0`, power-of-two head dims).
- The existing `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH=1` graph path is still
  required. The new phase path is intentionally not enabled by default.
- Anchored dequant-cache invalidation is now append-aware for one store epoch:
  append-only body seals can refill from the first dirty record instead of
  always rebuilding cached records `0..n-1`.
  `ggml-cuda.cu` now marks the root body pointer with the dispatch branch's
  record range, instead of pre-marking every store as dirty from record 0.

Validation completed:

- Build: `llama-cli`, `llama-bench`, and `test-kvarn-cuda-scratch-ref`.
- Focused CTest: `test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure`.
- `scripts/kvarn/kv_memory_estimate.py --self-test`.
- Small Qwen3.5 CUDA smoke with opt-in batched phases.
- Logits gates with opt-in batched phases:
  - small Qwen3.5: repeat/split/scratch NMSE `0`.
  - Qwen3.6 MTP: repeat/split/scratch NMSE `0`, expected layers `3-39:4`, `-ncmoe 34`.
  - Gemma 4 12B true KVarN+ISWA: repeat/split/scratch NMSE `0`, expected layers `5-47:6`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`.

Performance result:

- Small Qwen3.5 pp4096 improved only slightly after K/V stream overlap:
  `4046.65 +/- 62.53 t/s` default vs `4073.22 +/- 41.14 t/s` batched phases.
- Qwen3.6 MTP pp4096 regressed:
  `189.77 t/s` default vs `185.21 t/s` batched phases.
- Qwen3.6 MTP pp4096 with `LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE=1` after the
  root mark fix measured `189.97 t/s` and traced `dirty_from=0`, so the current
  direct-record prefill topology still does not get append-only dequant reuse.

Conclusion:

- The branch now follows the paper steer at the test-contract and first
  systems-shape level, but this batched body-store prototype is not the
  production performance fix.
- The append-aware dequant-cache update is a production-shape infrastructure
  fix, but Qwen3.6 pp4096 still needs a graph/store topology or attention-kernel
  change before that reuse matters for the failing long cell.
- Do not enable `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH_PHASES` by default.
- Do not lower production `--kvarn-iters`; the bounded production accuracy
  evidence is still bad for that direction.
- The next likely production patch is fused/occupancy-improved body-record
  attention: split active body records across CTAs per head/query, reduce
  score max/sum/value, and keep packed-vs-split/scratch as the oracle.

## Round 28 Codex update

The Opus paper-frame critique was implemented as a default-off scaffold:

- Env gate: `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`.
- Regular KVarN and true KVarN+ISWA mixed attention now can rotate `q`, store
  rotated sink/tail, and unrotate the output using an explicit Hadamard matrix.
- Pending stores are treated as already rotated under the flag, while direct raw
  prefill body stores are forced through the direct-record path so they still get
  exactly one body Hadamard.

Results:

- Default-off focused tests still pass.
- Small Qwen3.5 paper-frame logits pass repeat/split/scratch with NMSE `0`.
- Qwen3.6 ctx512/chunks2 accuracy passes both default and paper-frame at `0.28%`
  PPL increase.
- Qwen3.6 ctx4096/chunks2 improves but still fails:
  - old failure: `46.71%` PPL increase.
  - paper-frame after direct/pending corrections: `37.24%` PPL increase.
- Gemma 4 12B true KVarN+ISWA ctx4096/chunks2 still fails catastrophically.
- The Round 28 decode-parallel patch from Downloads is not directly applyable
  (`git apply --check` reports a corrupt patch at line 246) and should remain a
  speed follow-up, not the next production patch, until the 4096 accuracy gate
  passes.

Updated priority:

1. Do boundary-level f16-vs-KVarN diagnosis at ctx4096 under
   `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`.
2. Identify whether the remaining error is body quantization/scales, mask/window
   indexing, pending/direct layout, or output-frame handling.
3. Only after Qwen3.6 and Gemma ctx4096 accuracy pass, implement the
   decode-parallel body-attention path as an opt-in diagnostic and then benchmark
   `tg4096`.

## Round 29 update

Implemented the high-information 8-bit body ablation requested by Opus:

- New diagnostic preset: `kvarn_k8v8_g128`.
- Production preset remains `kvarn_k4v2_g128`.
- Common CLI, `llama-bench`, model validation, and graph validation accept KVarN group size 128 with 1-8 bit K/V.

Results with `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`:

- Qwen3.6 MTP ctx4096/chunks2, `kvarn_k8v8_g128`, `-ncmoe 34`: still FAIL, `35.74%` PPL increase.
- Qwen3.6 MTP ctx4096/chunks2, `kvarn_k8v8_g128`, `LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN=1`: still FAIL, `32.06%` PPL increase.
- Gemma 4 12B true KVarN+ISWA ctx4096/chunks2, `kvarn_k8v8_g128`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`: still catastrophic, `53780619.13%` PPL increase.

Conclusion:

- The Qwen ctx4096 failure is not primarily 2-bit V quality; 8-bit K/V still fails.
- The direct prefill body-store path is not the main Qwen culprit; pending-only improves slightly but still fails.
- The next patch should add an f16-vs-KVarN boundary dump at a selected long-context body-active attention call, not another speed kernel.
- Gemma true KVarN+ISWA likely has an additional ISWA window/eviction/recycling bug on top of the shared long-body issue.
