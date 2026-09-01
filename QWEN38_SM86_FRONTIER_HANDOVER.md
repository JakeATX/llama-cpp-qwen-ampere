# Qwen3.8-27B SM86 optimization handover

Updated: 2026-09-01

## Read this first

The production target is Qwen3.8-27B on one RTX 3090/3090 Ti (`sm_86`), full
GPU offload, `--parallel 1`, Flash Attention, Q8_0 K cache, Turbo3 V cache, and
native MTP3/MTP4. Q3_K_XL is the only currently qualified quant that preserves
the required 200K populated-context tier at `-ub 1024`.

The accepted product is stable. The next high-priority research task is a
**single fused CUDA chunked Gated DeltaNet (GDN) prefill kernel using the
existing WY/triangular algorithm**, followed by the larger Q3_K_XL replacement-
layout/narrow-integer-MMA decode project.

Do not conflate the two objectives. Chunked WY/GDN is a prompt-ingestion/TTFT
project: multi-token chunking is unavailable during ordinary singleton decode,
where GDN has measured only about 1--2% of GPU time. It will not by itself fix
the smaller 200K decode gain. Long-context generated-token throughput still
requires Q3-specific MMVQ work and/or a new attention dataflow that reduces the
ever-growing KV scan.

Do not repeat the graph-level chunked-GDN screen, generic large-N MMQ routing,
or the exact local Q3 unpack-reuse family. They have already been measured.

## GitHub and branch map

Fork remote:

`https://github.com/JakeATX/llama.cpp`

The important branches are:

- `perf/qwen38-sm86-decode-product`: accepted decode product, `c387d38b23b952823f408002d63a5a229b8ba4e3`;
- `perf/qwen38-sm86-prefill`: accepted prefill/runtime product, `dd66db0ccaec07cd9d325992e9040a53d6271eb6`;
- `research/qwen38-sm86-100k-decode-frontier`: decode experiments and this handover; source tip before the handover commit was `2795680c47ce02f83b225ef60c658994d6495c28`;
- `research/qwen38-sm86-ub512-1024-prefill-frontier`: prefill frontier experiments, tip `685d66faefe1dd754561776042c4c0ead08e0107`;
- `archive/qwen38-sm86-fast-mmvq`: rejected optional non-exact MMVQ path, `aef2a435527ce4761bbde3218bae4093899b9c81`.

All experiment commits are isolated and bisectable. Rejected candidates retain
default-off flags or a subsequent revert.

## Local machine map

Source worktrees:

- accepted/main development tree: `/home/jake-k/TheTom-llama-cpp-turboquant`;
- accepted decode product: use branch `perf/qwen38-sm86-decode-product`;
- decode frontier: `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-frontier-decode`;
- prefill frontier: `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-frontier-prefill`;
- archived FAST diagnostic: `/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-fast-diag`.

Models:

- Q4_K_M: `/home/jake-k/qwen38-bench/models/unsloth_qwen38_27b_ud_q4_k_m/Qwen3.8-27B-UD-Q4_K_M.gguf`;
- Q3_K_XL: `/home/jake-k/qwen38-bench/models/unsloth_qwen38_27b_ud_q3_k_xl/Qwen3.8-27B-UD-Q3_K_XL.gguf`.

Benchmark and audit root:

`/home/jake-k/qwen38-bench/kernel_cache_investigation`

Start with these local documents:

- `frontier/QWEN38_SM86_FRONTIER_FINAL.md`: consolidated frontier explanation and next-work ordering;
- `frontier/FRONTIER_LEDGER.csv`: every frontier experiment, commit, flag, result, and artifact directory;
- `frontier/FRONTIER_EXECUTION_PLAN.md`: lifecycle, correctness rules, and paper-trail requirements;
- `frontier/P3_gdn_chunk/`: chunked-GDN hypothesis, raw results, analysis, decision, and commands;
- `frontier/D2_q3_narrow_mmvq/reopen/INITIAL_DEV_REPORT.md`: most recent Q3 initial-development result;
- `frontier/DF_decode_portfolio/ANALYSIS.md`: final combined Q3 portfolio;
- `Q3_K_XL_MMVQ_FOLLOWUP.md`: Q3-specific follow-up specification;
- `QWEN38_SM86_DECODE_FINAL.md`: accepted decode qualification;
- `QWEN38_SM86_PREFILL_FINAL.md`: accepted prefill qualification;
- `prefill/CONTEXT_CAPACITY_UB1024.md`: direct maximum-context results;
- `prefill/runtime_200k_100tok_ab/RUNTIME_200K_100TOK_AB.md`: existing 200K runtime canary.

Raw Nsight reports and JSON results are intentionally not copied into Git.
They remain under the paths above on this machine.

## Reproduce the build

From either frontier worktree:

```bash
cmake -S . -B build-sm86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_NATIVE=ON

cmake --build build-sm86 -j8 \
  --target llama-server llama-bench llama-cli test-backend-ops
```

The decode frontier's working build is:

`/home/jake-k/TheTom-llama-cpp-turboquant-qwen38-frontier-decode/build-sm86`

The prefill worktree has historically used `build-cuda`; either name is fine if
the CMake architecture is explicitly 86. Do not use a stale build directory
after switching between candidate commits without rebuilding `libggml-cuda.so`.

Before timing, check for competing CUDA jobs with:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Desktop compositor/browser allocations are normal. Do not kill unrelated user
processes. Keep clocks, power, model, prompt, sampler, seed, and output length
fixed for A/B tests.

## What is already banked

Original comparison baseline means TheTom's TurboQuant+ commit
`f97400563641837efeeb2c6b9b45badfc5d35530`, not upstream stock llama.cpp.

The accepted product at `c387d38b` includes grouped Q8/Turbo3 MTP verification
attention and exact Q4_K/Q5_K narrow-verification reuse. Directly measured
results include:

- final EXACT versus original TurboQuant+ across ten bounded three-way cells:
  +17.92% median, +16.41% geometric mean, ten wins in ten cells;
- accepted narrow-MMVQ contribution against the accepted attention stack:
  +4.46% equal-cell median at 16K, +4.37% in the 2K/64K expansion, and +3.91%
  on held-out 100K RAG;
- accepted UB1024 prefill configuration versus UB512: +4.40% median;
- production-shaped prompt throughput confirmation: +5.92% to +7.72%.

MTP3 remains the product default. MTP3 and MTP4 were effectively tied in the
matched aggregate; MTP3 was selected for lower round latency, waste, and
workspace. MTP4 remains useful on explicitly qualified high-acceptance tasks.

## Why the 200K speedup looks smaller

The existing 200K number is not an apples-to-apples estimate of the complete
64K gain:

1. It is one 100-output-token canary on a highly repetitive prompt, not the
   multi-workload, multi-seed qualification used for aggregate claims.
2. It measures accepted `c387d38b` against original `f9740056` at one MTP3
   schedule. A single near-ceiling acceptance trajectory has low statistical
   power and different weighting from the broader cells.
3. The accepted narrow-MMVQ work removes a mostly context-independent amount of
   repeated weight unpacking per verification round. Attention, by contrast,
   scans a KV history whose cost grows with context.
4. Measured attention share rose from about 26--28% at 64K to 35.9% at 100K
   and 40.7--43.8% at 140K. The fixed MMVQ saving is therefore diluted at 200K.

The 200K canary still passed: 51.538 versus 50.027 tok/s, +3.02%, identical
output and acceptance, and identical 22,104 MiB peak VRAM. Treat +3.02% as a
capacity/performance canary, not a precise 200K aggregate. A proper estimate
would need the same prompt/workload manifest and 512--1024 output tokens at 64K
and 200K, paired and alternated. Do not rerun it merely to seek a larger number.

Also distinguish the accepted 64K stack from the historical maximum stack. The
accepted safe 64K coding measurements were approximately +10.94% for MTP3 and
+8.54% for MTP4. The often-cited +21.44% 64K/MTP3 result included the archived
fast MMVQ implementation that changed the continuation and verification-round
count and was removed from the product. Comparing that historical +21.44%
directly with the accepted Q3 200K +3.02% mixes different kernels, quants,
workloads, and qualification standards.

## Capacity facts

With Q8_0 K, Turbo3 V, MTP3, and UB1024:

- Q3_K_XL passed a 200,000-token prompt at about 22.1 GiB peak;
- Q3_K_XL passed 215,000 tokens at 22,764 MiB peak;
- Q3_K_XL failed the 225K request after 206,848 tokens;
- Q4_K_M passed 149,000 tokens and failed at 155,000;
- Q4_K_M cannot initialize the full 200K target/MTP configuration.

Any replacement/repack design must preserve the Q3 200K tier. Replace the
device layout of hot tensors rather than retaining a multi-GiB duplicate.

## Completed frontier experiments

### Decode attention

`b6d539ceb` implemented warp-owned packed-byte-sharing singleton GQA attention.
It was correct and spill-free but made the kernel 81--87% slower. `b73c8eb4e`
split each query head across two consumer warps; it remained correct/spill-free
but increased 100K attention time by 26.2% and regressed the request 2.42%.

Do not repeat “put more query heads in one CTA” without a new occupancy/traffic
argument. The existing accepted grouped verification path remains enabled.

### Q3 exact narrow reuse

The relaxed initial-development round implemented/restored:

- IQ3_S: `d9b9b0f0f`, flag `GGML_CUDA_SM86_IQ_REUSE`;
- IQ4_XS: `ca2b07ad6`, flag `GGML_CUDA_SM86_IQ4_REUSE`;
- IQ3_XXS: `a04f21195`, flag `GGML_CUDA_SM86_IQXXS_REUSE`;
- IQ2_S: `25ad45d27`, flag `GGML_CUDA_SM86_IQ2S_REUSE`.

IQ3_XXS and IQ2_S passed 32/32 CUDA-versus-reference cases each. Width-five
microkernels improved about 1.1% and 1.45%, but the clean combined 64K/MTP4
test measured -0.17% mean and -0.11% median with identical output, acceptance,
and VRAM. Keep all four flags off.

`2bb855846` routed Q3 width four/five through the existing Ampere large-N MMQ
engine. It passed its reference cases but regressed 20--25% at width four and
6--13% at width five. `47b3ef28f` reverts it. The next path needs a genuine
narrow N=8 layout/kernel, not a dispatch change.

### Prefill MMQ and attention

- async activation-stage MMQ `1c235461c`: correct, approximately -0.3% E2E;
- exact-48 GQA attention `20c3bc440`: correct, only 0.25% attention reduction;
- direct Q8/Turbo3 prefill attention `27bc49fb6`: correct canary but 2.97x
  slower attention and -12.33% prompt throughput because compressed KV was
  decoded repeatedly for successive query tiles;
- paired prepared-Q8_1 `af87de28a`: +0.36% Q4 / +0.26% Q3 at 16K, then -0.32%
  median prompt throughput in the 64K portfolio. Leave it off.

### GDN graph reference

`26f20a32d` exposed the existing graph-level chunk algorithm behind
`GGML_CUDA_SM86_GDN_GRAPH_REFERENCE`. A Q/K head-broadcast assumption was fixed
so the path could run Qwen. The canary was numerically/visibly correct, but Q4
16K UB1024 prompt throughput fell 13.44%. `9e3b868ca` reverts it.

This rejects the graph realization, not chunked GDN mathematics.

## Priority 1: fused CUDA WY/triangular GDN

### Exact starting points

Work on a new branch from accepted prefill `dd66db0cc`, not from a tree with an
enabled frontier experiment. Suggested branch:

`research/qwen38-sm86-fused-gdn-chunk`

Inspect these files first:

- `ggml/src/ggml-cuda/gated_delta_net.cu`: current fused operator; it loops
  serially over `t < n_tokens` and contains `TODO: Add chunked kernel for even
  faster pre-fill`;
- `ggml/src/ggml-cuda/gated_delta_net.cuh`: CUDA operator interface;
- `src/models/delta-net-base.cpp`, especially `build_delta_net_chunking()`:
  working graph expression of the decay mask, lower-triangular solve, WY-like
  transformed values/keys, outputs, and final state;
- `src/models/delta-net-base.cpp`, `build_delta_net()`: singleton versus
  multi-token dispatch;
- `src/models/qwen35.cpp`: Qwen broadcast and fused-GDN integration;
- `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatch and recurrent-state snapshot-copy
  fusion.

The Qwen path has 48 recurrent GDN layers. Preserve FP32 recurrent state and
leave singleton decode/rollback untouched.

### Implementation sequence

1. Add `GGML_CUDA_SM86_GDN_CHUNK=0|32|64`; default zero must reproduce the
   accepted product exactly.
2. Dispatch only multi-token prefill on SM86 and only the qualified Qwen shape.
   Keep the serial fused and generic graph fallbacks.
3. Implement chunk 32 first. One CTA should own a sequence/head/chunk unit and
   keep the compact triangular/WY factors and state tile on chip as far as the
   SM86 register/shared-memory budget permits.
4. Fuse the operations currently materialized by the graph: decay-prefix
   preparation, lower-triangular system/factor construction, transformed K/V,
   within-chunk output, and final FP32 state update.
5. Use Tensor Core-friendly small matrix operations where they reduce total
   work, but do not force every triangular operation through MMA if packing and
   padding dominate.
6. Avoid writing full `[chunk, chunk, head, chunks]` intermediates to global
   memory. This is the principal difference from the rejected graph path.
7. Prove chunk tails and state boundaries before measuring. UB512 has 16
   chunk-32 groups and UB1024 has 32 per microbatch.
8. Try chunk 64 only after chunk 32 is correct and faster. It is the one bounded
   refinement, not a broad tile sweep.

### Correctness and quality

Strict token identity is diagnostic, not the product gate. Preserve represented
model values and FP32 state, but tolerate floating-point scheduling differences
if all of the following pass:

- fixed-token replay at actual chunk dispatch with cross-entropy/PPL ratio,
  mean/p95/p99/max logit KL, and target-token log-probability deltas;
- final recurrent-state comparison at every chunk boundary;
- ragged lengths 1, 31, 32, 33, 63, 64, 65 and actual UB tails;
- frozen singleton decode and every MTP partial-rejection/rollback canary;
- Compute Sanitizer memcheck, racecheck, and synccheck;
- independent coding, code-edit, reasoning, and retrieval canaries under the
  production sampler.

### Performance gates

Start with Q4 and Q3 at 16K, UB1024, five alternating repetitions. Record GDN
kernel time as well as prompt tok/s. Advance if the first correct fused kernel
is consistently positive or reduces total GDN time materially. Product
promotion still requires at least 25% total-GDN reduction and approximately
2.5% end-to-end prefill improvement, or a directly dependent portfolio that
clears the ordinary aggregate gate.

Qualify survivors at UB512 and UB1024, 16K/64K/100K, both quants. Include TTFT,
short fixed decode tail, peak VRAM, and the 200K Q3 capacity canary. Do not add
component percentages; measure the combined runtime directly.

### Reproduction templates

Baseline/screen:

```bash
GGML_CUDA_SM86_GDN_CHUNK=0 build-sm86/bin/llama-bench \
  -m Q3_MODEL -p 16384 -n 0 -b 4096 -ub 1024 \
  -ctk q8_0 -ctv turbo3 -t 8 -ngl 99 -fa on \
  -r 5 -o json --progress -fitt 0

GGML_CUDA_SM86_GDN_CHUNK=32 build-sm86/bin/llama-bench \
  -m Q3_MODEL -p 16384 -n 0 -b 4096 -ub 1024 \
  -ctk q8_0 -ctv turbo3 -t 8 -ngl 99 -fa on \
  -r 5 -o json --progress -fitt 0
```

Profile only after the screen proves dispatch and correctness:

```bash
GGML_CUDA_SM86_GDN_CHUNK=32 nsys profile \
  --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
  --stats=false --cuda-flush-interval=1000 --cuda-graph-trace=node \
  -o /tmp/qwen38_gdn_chunk32 \
  build-sm86/bin/llama-bench -m Q3_MODEL -p 16384 -n 0 \
  -b 4096 -ub 1024 -ctk q8_0 -ctv turbo3 -t 8 -ngl 99 \
  -fa on -r 1 -o json --progress -fitt 0
```

Use `/home/jake-k/qwen38-bench/kernel_cache_investigation/analyze_nsys.py`
to normalize the Nsight report.

## Priority 2: Q3 replacement layout plus narrow IMMA

After or independently from fused GDN:

1. Reprofile Q3 at 64K, 100K, and 200K to rank exact IQ type/tensor/shape calls.
2. Repack one hottest width-five tensor family at model load into an equivalent
   SM86 consumption order. Replace its device representation; do not duplicate.
3. Implement a dedicated physical N=8 integer-MMA kernel, width five first.
   Eight derives from SM86 MMA fragment geometry; width five uses five of eight
   output columns and width four uses four.
4. Producer warps stage/prepare packed weights and Q8_1 activations; consumer
   warps execute MMA and apply the exact scales/mins in FP32.
5. Extend to width four/MTP3 only if width five proves the dataflow.
6. Under the relaxed development rule, any reproducible positive E2E canary may
   receive one refinement. Final product promotion remains +2% aggregate or
   +3% at 64K--200K with no important regression and preserved 200K capacity.

Use the existing reference shapes/tests introduced in `f002298a5` and
`268d2547a`, and the raw microkernel evidence under
`frontier/D2_q3_narrow_mmvq/reopen/`.

## Paper-trail requirements

For every new experiment create an artifact directory under:

`/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/`

Include:

- `HYPOTHESIS.md` before coding;
- exact baseline and candidate SHAs;
- `commands.txt`;
- raw benchmark/profile/sanitizer output;
- normalized measurements;
- compiler resource report: registers, shared memory, occupancy, spills;
- `ANALYSIS.md` explaining the mechanism in plain English;
- `DECISION.md` stating accept/refine/reject;
- one append-only row in `FRONTIER_LEDGER.csv`.

Do not erase unfavorable data or repeatedly rerun until a result improves.
Measure the final portfolio directly. Preserve rejected source in an isolated
commit followed by a revert or keep it behind a default-off flag.

## Current safe recommendation

Use accepted EXACT with MTP3, Q8_0 K/Turbo3 V, and UB1024. Use Q3_K_XL for
200K. Leave these frontier flags off:

- `GGML_CUDA_SM86_Q8_T3_GQA_WARP`;
- `GGML_CUDA_SM86_IQ_REUSE`;
- `GGML_CUDA_SM86_IQ4_REUSE`;
- `GGML_CUDA_SM86_IQXXS_REUSE`;
- `GGML_CUDA_SM86_IQ2S_REUSE`;
- `GGML_CUDA_PREPARED_Q8_1`;
- `GGML_CUDA_SM86_GDN_GRAPH_REFERENCE`.

The handoff objective is not to retune these flags. It is to build a genuinely
fused chunked-GDN CUDA implementation and, secondarily, a genuinely narrow Q3
layout/IMMA dataflow.
