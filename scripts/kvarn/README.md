# KVarN KV-cache backend notes

This branch keeps KVarN separate from `ggml_type`. `--kv-cache-quant kvarn`
selects a KV-cache backend mode and uses `llama_kvarn_params` for layout and
runtime policy.

Current implemented pieces:

- Public API and common CLI flags for `--kv-cache-quant none|kvarn`.
- `kvarn_k4v2_g128` preset defaults: group 128, 4-bit K, 2-bit V, 128 sink
  tokens, 128 tail tokens, 16 Sinkhorn iterations.
- Runtime K/V head dimensions of 128, 256, and 512 are supported by
  validation, layout/memory-estimator tests, runtime metadata tests, CUDA
  parity tests, and local `llama-cli` smokes. The 256-dimensional Qwen path is
  the broadest acceptance path. The 512-dimensional Gemma 4 path now runs
  through the KVarN+ISWA composite: non-SWA layers use KVarN storage, SWA
  layers use the normal sliding-window KV cache, and Gemma-style physical KV
  layer reuse is preserved.
- CPU reference layout, Hadamard rotation, Sinkhorn-style balancing,
  asymmetric RTN, bit packing, body-record dequant, and a sink/body/tail
  reference cache.
- CUDA body dequant primitive in `ggml/src/ggml-cuda/kvarn.cu`.
- CUDA multi-record scratch dequant primitive that materializes packed KVarN
  body records into F32 K/V scratch tensors. This is a correctness/reference
  building block for the future graph-level scratch path.
- CUDA reference body-store primitive for the preset min/max RTN path. It
  transforms one full K/V body group into packed K/V bytes plus FP32 scales on
  device, supports RTN scale quantiles in `(0, 1]`, and is byte-parity tested
  against the CPU reference store.
- CUDA packed-K QK scoring primitive that computes per-body-token attention
  scores directly from packed KVarN K records and scale metadata.
- CUDA packed-V AV accumulation primitive that computes the weighted value
  vector directly from packed KVarN V records and scale metadata.
- CUDA packed-body attention primitive that runs QK, stable softmax, and AV
  over one KVarN body record using packed K/V and scale metadata.
- CUDA multi-record packed-body attention primitive that runs one stable
  softmax over multiple packed KVarN body records.
- CUDA batched multi-query variant of the multi-record packed-body attention
  primitive, with explicit query/output/probability strides.
- CUDA mixed sink/body/tail attention primitive that attends over FP16
  sink/tail tokens and packed KVarN body records in one stable softmax.
- The runtime packed mixed-attention path has a fused CUDA kernel for one-query
  decode batches. Multi-query prompt batches use the per-row serial fused CUDA
  kernel by default; the faster multi-block fused-batch variant still diverges
  from the scratch-reference path under Qwen3.6 prompt batching. Forcing
  `LLAMA_KVARN_ATTN_FUSED_BATCH=1` is rejected at context initialization with
  an explicit error instead of falling back silently. Set
  `LLAMA_KVARN_ATTN_SERIAL_FUSED=1` or `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` for
  supported A/B debugging of the packed attention variants.
- CUDA F32 scratch mixed-attention primitive that consumes dequantized body
  scratch tensors and is used as a device-side reference for the packed mixed
  attention primitive.
- CUDA batched F16 sink/body/pending/tail attention wrapper that maps MHA
  query heads to KV heads and reads directly from the runtime's F16 sink/tail
  and FP32 pending-body tensor layouts. Wrapped tail rings are consumed in
  chronological order via an explicit `tail_start` op parameter instead of raw
  numeric slot order. The F16 mixed-attention score kernel writes normalized
  softmax probabilities back to the global score/probability buffer consumed by
  the AV kernel.
- Runtime-selectable CUDA scratch-reference attention mode. Setting
  `LLAMA_KVARN_ATTN_REF_SCRATCH=1` expands the KVarN graph scratch tensor,
  dequants active packed body records into F32 scratch tensors on device, and
  runs a F16 sink/body/pending/tail mixed-attention wrapper over the scratch
  body tensors. This is a live-runtime correctness oracle for the packed mixed
  attention path.
- Debug-only prompt-batch controls are available for bring-up:
  `LLAMA_KVARN_DEBUG_UBATCH=<n>` overrides the bounded KVarN ubatch size only
  within the tail-ring safety limit. Values above `tail_tokens` are rejected
  explicitly because they can evict a tail slot before the same graph finishes
  copying it into pending body storage.
  `LLAMA_KVARN_ATTN_FUSED_BATCH=1` is intentionally disabled in llama.cpp
  runtime contexts because Qwen3.6 prompt-batch logits still diverge; this is
  an explicit unsupported-mode error, not a fallback to split kernels.
  For diagnostics only, pair it with
  `LLAMA_KVARN_UNSAFE_ALLOW_FUSED_BATCH=1` to bypass the context guard and
  reproduce the unsafe fused-batch path. Do not use this combination as a
  production mode until Qwen3.6 packed-repeat logits pass at the normal
  `llama-results` threshold.
  KVarN diagnostic environment flags are parsed strictly as boolean `0` or
  `1`: malformed values such as `LLAMA_KVARN_ATTN_REF_SCRATCH=bogus` and
  out-of-range values such as `LLAMA_KVARN_ATTN_REF_SCRATCH=2` now fail with an
  explicit invalid environment-flag error instead of being silently treated as
  disabled or enabled.
- KVarN runtime memory/context skeleton with native slot metadata, sequence
  operations, memory estimates, and production-shaped per-layer/per-KV-head
  storage. Runtime storage keeps sink/tail tokens in FP16 and seals full body
  groups into packed KVarN body records plus scale metadata. It does not
  allocate or use the normal KV cache as a fallback.
- KVarN cache layer-reuse metadata is wired for models whose later layers
  reuse an earlier physical KV layer. Reuse-only graph layers attend from the
  mapped KVarN storage and skip duplicate sink/tail writes, tail-eviction
  staging, and body-record sealing. This is used by Gemma-style K/V reuse.
- The `llama_kv_cache_kvarn_iswa` memory composite is routed for Gemma 4. It
  owns non-SWA layers with KVarN storage and SWA layers with the existing
  normal sliding-window KV cache, preserving the normal SWA sizing, state, and
  sequence operations. Non-Gemma SWA/ISWA models still fail explicitly until a
  model-specific cache/reuse policy is added.
- KVarN batch preparation now admits bounded prompt ubatches up to one tail-ring
  span (`min(n_ubatch, tail_tokens)`) for dense, hybrid, and MoE models, so
  prompt processing can use masked multi-query KVarN attention without evicting
  a tail slot written earlier in the same graph. Qwen3.6 MoE bounded prompt
  batching is correct through the serial fused multi-query CUDA path; the
  forced multi-block fused-batch path is explicitly rejected because packed
  repeats still diverge.
- The shared `llama_batch_allocr::split_equal()` path has regression coverage
  ensuring it does not emit more sequence sets than its `n_ubatch` limit. This
  protects KVarN's tail-ring prompt bound and the other memory backends that
  rely on equal-sequence ubatches.
- Hybrid recurrent/full-attention models can compose recurrent memory for SSM
  layers with KVarN storage for full-attention layers. This is used by local
  Qwen3.5/Qwen3.6-family validation instead of rejecting the whole model for
  recurrent layers.
- KVarN backend tensor allocation for FP16 sink/tail, packed K body bytes,
  packed V body bytes, FP32 K scales, FP32 V scales, and FP32 pending body
  staging buffers. Scale tensors are
  FP32 because the current CUDA KVarN primitives consume FP32 scale metadata;
  reducing them to FP16 needs matching half-scale kernels. These tensors are
  allocated per KV layer on the same layer device when `offload_kqv` is
  enabled, or CPU otherwise.
- Graph-visible FP16 sink/tail store primitives backed by `ggml_set_rows`.
  Current K/V tensors can be reshaped and written into the allocated KVarN
  sink/tail tensors with explicit KVarN sink/tail slot indices. Body sealing
  and packed body/scales still need a KVarN-specific graph/backend op.
- Graph-visible KVarN body plan input. For each token the graph receives
  `[record, offset, seal_record]`, using `-1` for sink-only or pending fields,
  so body pack/seal ops can write deterministic packed records without
  re-deriving token placement inside backend code.
- Graph-visible pending body offset input and tail-eviction K/V stores. When
  the FP16 tail ring is full, graph construction copies the evicted FP16
  sink/tail row into FP32 pending group slots with `ggml_get_rows`,
  `ggml_cast`, and `ggml_set_rows` before overwriting the tail slot. Completed
  body groups can then be packed without reading from the normal KV cache.
- Public KVarN layer-view metadata for graph/runtime integration. The memory
  context now exposes per-layer sink/tail tensors, packed body tensors, FP32
  scale tensors, K/V record layouts, and CUDA body-store scratch sizing through
  a narrow typed view instead of requiring graph code to inspect private cache
  storage.
- `GGML_OP_KVARN_STORE_BODY` plus `ggml_kvarn_store_k_body()` and
  `ggml_kvarn_store_v_body()` cache-write constructors. These return a view of
  the destination packed body tensor and carry the matching FP32 scale tensor
  and scratch tensor as sources, so K and V body stores can be scheduled as
  independent backend ops. CUDA dispatch is wired to the KVarN min/max body
  store primitives.
- `GGML_OP_KVARN_ATTN_MIXED` plus `ggml_kvarn_attn_mixed()` constructor. The
  op produces the normal `[head_dim, n_head, n_tokens]` F32 attention output
  shape from F32 Q, F16 sink/tail tensors, packed body tensors, FP32 scales,
  FP32 pending-body tensors, a scratch score buffer, and an optional KQ/causal
  mask in `src[10]`. CUDA dispatch passes F32/F16 masks through to the batched
  F16 sink/body/pending/tail wrapper. The llama graph now builds and fills the
  KQ mask for KVarN graph inputs. Production runtime preparation bounds prompt
  chunks to one tail-ring span while prompt batching is being brought up. The
  graph computes active sink/body/pending/tail counts and the wrapped-tail start
  slot for each ubatch from the last token position. Context reserve graphs may
  be built for larger synthetic ubatches.
- Graph construction writes FP16 sink/tail, stages evicted FP16 tail rows into
  FP32 pending body slots, emits packed K/V body-store nodes when a graph
  completes one body record, and uses KVarN mixed attention for decode and
  bounded prompt batches. It still refuses graph reuse across graphs
  that include body-store ops or shape changes.
- When `LLAMA_KVARN_ATTN_REF_SCRATCH=1` is set during graph construction,
  `kvarn_attn_scores` is sized for score probabilities plus K/V body scratch
  for active body records in the current layer. Graph reuse validates this
  workspace before reusing scratch-reference graphs.
- KVarN reserve graphs are built with a synthetic single-token ubatch at the
  end of the cache, rather than position zero. This makes reserve-time compute
  buffers include the worst-case body-record scratch required by
  `LLAMA_KVARN_ATTN_REF_SCRATCH=1` and avoids late scratch-reference buffer
  growth during long decode validation.
- One-token KVarN decode graphs can be reused when the active sink/body/pending
  /tail window fits the originally allocated tensors and the next token does
  not seal a body record. KVarN mixed attention nodes update their dynamic
  active-window op params before each reused graph execution.
- CUDA graph capture is disabled for graphs containing
  `GGML_OP_KVARN_ATTN_MIXED`. The KVarN decode graph mutates active-window
  op params between executions, and CUDA graph replay keeps stale kernel
  arguments for that case. Llama-level graph reuse remains enabled.
- Common argument parsing normalizes server auto-parallel to one slot when
  `--kv-cache-quant kvarn` is selected, and rejects explicit `--parallel`
  values above `1`. KVarN context initialization also rejects `n_seq_max > 1`
  as a lower-level guard until multi-stream KVarN storage and scheduling are
  implemented.
- Unit tests in `tests/test-kvarn-kv.cpp` and a standalone CUDA parity test in
  `tests/test-kvarn-cuda-dequant.cpp`, registered as
  `test-kvarn-cuda-scratch-ref`.
  The runtime unit test covers layout/packing, cache sealing, graph-visible
  sink/tail/body-plan/body-record inputs, and causal/non-causal KQ-mask graph
  inputs for both F32 and F16 masks.
  The standalone CUDA test covers store/dequant, packed body attention,
  sink/body/tail mixed attention, pending tokens, batched F16 mixed attention,
  wrapped-tail decode order, padded full-attention KQ-mask strides for the
  256-dimensional forced-fused path, an exact Qwen3.6-shaped 512-token active
  window (`128` sink, `2` body records, `128` tail, `128` queries, `16` query
  heads, `2` KV heads), multi-record scratch dequant, and packed
  mixed-attention versus scratch-dequant mixed-attention parity without linking
  the full llama runtime.
- `tests/test-kvarn-server-load-failure.cpp` covers the server-backed loader
  failure path for unsupported KVarN fixtures, so validation errors return a
  clean model-load failure instead of dereferencing a null context.

The runtime still refuses unsupported KVarN graph shapes rather than silently
using the normal FP16/`ggml_type` KV cache. KVarN memory construction succeeds
after compatibility validation, and KVarN batch/full contexts can now be
constructed with native slot metadata. Graph construction identifies
`llama_kv_cache_kvarn_context` before the normal `llama_kv_cache_context` casts.
The KVarN graph path stores sink/tail tensors, can seal one completed body
record from pending K/V staging, and can consume KVarN sink/tail/body/scale
storage through CUDA mixed attention. Runtime execution now uses bounded prompt
ubatches with KQ masks, including Qwen3.6 MoE through the serial fused
multi-query CUDA attention path.

Verified local smoke:

- Compatible model downloaded with
  `hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF`.
- Reproducible local model metadata discovery:
  `python scripts\kvarn\discover_models.py --gpu-vram-gib 12 --vram-reserve-gib 1.5 C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf" "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"`.
  The script now reports GGUF tensor GiB, expert/non-expert tensor split,
  expert share, usable VRAM, and full-offload fit. Latest local result with a
  10.50 GiB usable VRAM budget reports Gemma4 12B dense Q3 as `5.59` tensor
  GiB and full-offload fit `yes` with `4.91` GiB margin
  (`design-512,swa/iswa-likely,swa-256`); Gemma4 26B A4B Q3 as `12.01`
  tensor GiB with `9.60` expert GiB (`80.0%`) and full-offload fit `no`
  with `-1.51` GiB margin (`design-512,swa/iswa-likely,swa-256,moe`);
  Qwen3.6 35B A3B MTP IQ3 as `14.28` tensor GiB with `12.29` expert GiB
  (`86.1%`) and full-offload fit `no` with `-3.78` GiB margin
  (`primary-256,hybrid-ssm,moe`); and Qwen3.6 35B A3B Q3 as `15.68` tensor
  GiB with `13.30` expert GiB (`84.8%`) and full-offload fit `no` with
  `-5.18` GiB margin (`primary-256,hybrid-ssm,moe`). These margins are model
  tensor bytes only; runtime KV, graph, and workspace allocations require
  additional headroom.
- Additional local 256 metadata discovery:
  `python scripts\kvarn\discover_models.py C:\Users\sjake\.cache\huggingface\hub\models--unsloth--Qwen3.5-4B-GGUF\snapshots\e87f176479d0855a907a41277aca2f8ee7a09523\Qwen3.5-4B-Q4_K_M.gguf C:\Users\sjake\.cache\huggingface\hub\models--unsloth--Qwen3.5-4B-GGUF\snapshots\e87f176479d0855a907a41277aca2f8ee7a09523\Qwen3.5-4B-UD-Q4_K_XL.gguf C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
  Latest local result reports both Qwen3.5 4B files as 256-dim hybrid SSM
  (`32` layers, `16` heads, `4` KV heads) and the non-MTP Qwen3.6 35B A3B
  Q3/Q4 files as 256-dim hybrid MoE (`40` layers, `256` experts, `8` active).
- CUDA FA-off build:
  `cmake -S . -B build-kvarn-cuda-nofa-vs -DGGML_CUDA=ON -DGGML_CUDA_FA=OFF -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a-real`.
- Short KVarN smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Hello" -n 1 -c 256 -ngl 99 --no-warmup --simple-io --single-turn -fa on --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
- Long raw KVarN completion smoke crossing the sink/tail threshold:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-completion.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Hello" -n 270 -c 512 -ngl 99 --no-warmup --simple-io -no-cnv --no-display-prompt --ignore-eos -fa on --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 -s 1234 --temp 0`.
  This completed successfully at 77.41 tokens/s eval on the RTX 5070 with
  `graphs reused = 268` and coherent deterministic output.
- Long KVarN completion smoke with quantile clipping enabled:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-completion.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Hello" -n 270 -c 512 -ngl 99 --no-warmup --simple-io -no-cnv --no-display-prompt --ignore-eos -fa on --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95 -s 1234 --temp 0`.
  CUDA body-store parity now covers `rtn_quantile = 0.95`, and the
  deterministic reuse comparison passed with reuse disabled `74.22` eval tok/s,
  reuse enabled `77.26` eval tok/s, and `graphs reused = 268`.
- Matching normal-KV baseline:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-completion.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Hello" -n 270 -c 512 -ngl 99 --no-warmup --simple-io -no-cnv --no-display-prompt --ignore-eos -fa off --kv-cache-quant none -s 1234 --temp 0`.
  This completed at 222.63 tokens/s eval with `graphs reused = 268`, so the current KVarN path is
  correctness-oriented and not yet performance-competitive.
- Gemma 4 12B 512-dimensional KVarN+ISWA body-record smoke on the static CUDA
  build:
  `build-kvarn-cuda-static-vs\bin\Release\llama-completion.exe -fit off -m C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf -p "Hello" -n 270 -c 512 -ngl 99 --no-warmup --simple-io -no-cnv --no-display-prompt --ignore-eos -fa on --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 -s 1234 --temp 0`.
  Latest local result passes after rebuilding stale static CLI tools at commit
  `1fc9833ff`: KVarN owns Gemma full-attention layers `5, 11, 17, 23, 29, 35,
  41, 47`, allocates `2` body records per KVarN layer at `-c 512`, reports
  `8.87 MiB` CUDA KVarN buffer, `4.87 MiB` estimated KVarN metadata cache, and
  runs at `32.21` eval tok/s with `graphs reused = 267`.
- Matching Gemma 4 12B normal-KV baseline:
  `build-kvarn-cuda-static-vs\bin\Release\llama-completion.exe -fit off -m C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf -p "Hello" -n 270 -c 512 -ngl 99 --no-warmup --simple-io -no-cnv --no-display-prompt --ignore-eos -fa off --kv-cache-quant none -s 1234 --temp 0`.
  Latest local result: `67.79` eval tok/s with `graphs reused = 267`, so the
  Gemma KVarN+ISWA path is functionally up but still roughly half normal-KV
  decode speed on this short smoke.
- Standard benchmark smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-bench.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p 0 -n 64 -r 1 -ngl 99 -fa on --no-warmup --kv-cache-quant none,kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95`.
  This verifies the `llama-bench` KVarN option plumbing. Latest local result:
  normal KV `154.75` tok/s, KVarN `84.48` tok/s for `tg64`.
- Standard prompt-processing benchmark:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -BuildDir build-kvarn-cuda-static-vs -CaseList "pp64:64:0" -RtnQuantile 0.95 -FlashAttn off -Repetitions 1`.
  Latest static local result on build `50f2196a0`: normal KV `585.88` tok/s,
  KVarN `220.31` tok/s for `pp64`; the benchmark harness verified 28 KVarN
  layer log lines, so the result cannot hide a normal-KV fallback.
- Standard benchmark crossing packed body records:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-bench.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p 0 -n 384 -r 1 -ngl 99 -fa on --no-warmup --kv-cache-quant none,kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95`.
  Latest local result: normal KV `148.34` tok/s, KVarN `35.96` tok/s for
  `tg384`, with KVarN allocating two body records per layer.
- Single-slot server smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-server.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8125 -c 256 -ngl 99 --no-warmup -fa on --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  `/completion` with `{"prompt":"Hello","n_predict":1,"temperature":0}`
  returned `"."` and reported `graphs reused = 1`. Because KVarN is selected,
  server auto-parallel is normalized to one slot.
- Single-slot server smoke with quantile clipping enabled:
  same server command with `--kvarn-rtn-quantile 0.95` on port `8126`.
  `/completion` with `{"prompt":"Hello","n_predict":1,"temperature":0}`
  returned `"."` and reported `graphs reused = 1`.
- Reusable single-slot server smoke:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-static-vs`.
  This starts `llama-server` on localhost, waits for `/health`, posts one
  deterministic `/completion` request, and always stops the child process. Add
  `-CheckSlotSaveRejection` to start the server with a temporary
  `--slot-save-path` and verify `/slots/0?action=save` rejects KVarN state
  serialization cleanly instead of writing an empty state.
  The script now also rejects a smoke as invalid unless captured server logs
  contain `llama_kv_cache_kvarn:` initialization lines, so a content response
  alone cannot hide a normal-KV fallback.
  The static server build is used on this Windows machine because Smart App
  Control blocks the unsigned shared `llama-server-impl.dll` from
  `build-kvarn-cuda-nofa-vs` with Code Integrity error `4551`.
  Latest local result with the log check enabled:
  `KVarN server smoke: PASS, content = '.'`,
  `KVarN server log check: PASS, KVarN layer lines = 56`, and
  `KVarN slot save rejection: PASS`.
- Larger context allocation/decode smoke:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_cuda_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs -CtxList "512 1024 2048" -RtnQuantile 0.95`.
  FP16 and KVarN smoke paths passed for all three context sizes. Reported
  KVarN cache estimates were `8.64 MiB` at 512 tokens, `11.92 MiB` at 1024
  tokens, and `18.48 MiB` at 2048 tokens.
  The script now rejects a successful KVarN CLI smoke unless output logs
  contain `llama_kv_cache_kvarn:` initialization lines, so a successful token
  response alone cannot hide a normal-KV fallback. Static CUDA rerun:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_cuda_smoke.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -BuildDir build-kvarn-cuda-static-vs -CtxList "256 512" -RtnQuantile 0.95 -MinKvarnLayerLogs 1`.
  Latest local result passed both contexts with
  `KVarN CLI log check: PASS, KVarN layer lines = 56`; the `-c 512` KVarN path
  allocated `2` body records per layer and reported an `8.64 MiB` metadata
  estimate.
  The script also accepts `-ExpectedKvarnLayers` to require exact routed layer
  IDs. Latest Gemma 4 12B exact-layer rerun:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_cuda_smoke.ps1 -Model "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" -BuildDir build-kvarn-cuda-static-vs -CtxList "256" -RtnQuantile 0.95 -MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5,11,17,23,29,35,41,47"`.
  This passed with `KVarN expected layer check: PASS, layers =
  5,11,17,23,29,35,41,47` and 16 KVarN layer log lines.
- `scripts\kvarn\kv_memory_estimate.py` now mirrors the runtime logical memory
  formula and has `--self-test` coverage against `test-kvarn-kv` 128-, 256-,
  and 512-dimensional reference totals. It reports full FP16 KV, ideal
  full-context low-bit KV, and KVarN's FP16 sink/tail, packed body, and scale
  breakdown.
- Memory estimator for Qwen2.5 1.5B geometry (`28` layers, `2` KV heads,
  `128` head dim, K4/V2/group128, 128 sink + 128 tail) reports KVarN totals of
  `31.61 MiB` at 4K context, `57.86 MiB` at 8K context, and `110.36 MiB` at
  16K context. The same selected KV geometry in FP16 would be `112.00 MiB`,
  `224.00 MiB`, and `448.00 MiB`.
- Memory estimator for 256-dim hybrid Qwen3.5 full-attention geometry (`6`
  layers, `2` KV heads, `256` head dim, K4/V2/group128, 128 sink + 128 tail)
  reports KVarN totals of `13.02 MiB` at 4K context and `23.71 MiB` at 8K
  context. The same geometry in FP16 KV would be `48.00 MiB` and `96.00 MiB`.
- Memory estimator for 256-dim hybrid Qwen3.5 4B full-attention geometry (`8`
  layers, `4` KV heads, `256` head dim, K4/V2/group128, 128 sink + 128 tail)
  reports KVarN totals of `34.72 MiB` at 4K context, `63.22 MiB` at 8K
  context, and `120.22 MiB` at 16K context. The same geometry in FP16 KV would
  be `128.00 MiB`, `256.00 MiB`, and `512.00 MiB`. At the default 512-cell
  `llama-bench` allocation, the estimator reports `9.78 MiB`, matching the
  runtime `KVarN metadata cache` log for the Qwen3.5 4B `tg384` benchmark.
- Memory estimator for 256-dim hybrid Qwen3.6 35B A3B MTP full-attention
  geometry (`10` layers, `2` KV heads, `256` head dim, K4/V2/group128, 128
  sink + 128 tail) reports KVarN totals of `21.70 MiB` at 4K context,
  `39.51 MiB` at 8K context, and `75.14 MiB` at 16K context. The same geometry
  in FP16 KV would be `80.00 MiB`, `160.00 MiB`, and `320.00 MiB`.
- Explicit multi-slot startup now fails cleanly before model load with:
  `KVarN currently supports only --parallel 1`.
- Focused tests passed:
  `ctest --test-dir build-kvarn-cpu -C Release -R "test-kvarn-kv|test-arg-parser|test-kvarn-server-load-failure" --output-on-failure`
  and
  `ctest --test-dir build-kvarn-cuda-nofa-vs -C Release -R "test-kvarn-cuda-scratch-ref|test-kvarn-cuda-mixed-tail" --output-on-failure`.
  Latest local result on 2026-06-05 passed. `test-kvarn-kv` now asserts 512
  memory estimates, runtime layer-view shapes, scale/body tensor sizing, and
  body-store graph op shapes, plus KVarN physical-layer reuse mapping. It also
  covers 512-dimensional record layout totals and CPU reference
  store/dequant, so the 512 path is not only covered through memory metadata.
  `test-arg-parser` now asserts KVarN server auto-parallel normalization,
  explicit `--parallel -1` normalization to one slot, explicit `--parallel 2`
  rejection, and invalid KVarN preset/RTN-quantile rejection. Static CUDA
  parser rerun:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-arg-parser" --output-on-failure`.
  Latest local result passed after adding the explicit auto-parallel and
  invalid KVarN scalar-argument regressions.
  Static CUDA layout/reference rerun:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-kvarn-kv" --output-on-failure`.
  Latest local result passed after adding the 512-dimensional layout
  total-record-byte and CPU reference store/dequant regressions. Matching
  script check `python scripts\kvarn\kv_memory_estimate.py --self-test` passed;
  `python scripts\kvarn\kv_memory_estimate.py --layers 8 --kv-heads 1 --head-dim 512 --ctx 512`
  reports KVarN total `5103616` bytes (`4.87 MiB`) and `2` body records per
  layer/head, matching the Gemma 4 12B runtime smoke geometry.
  `test-kvarn-cuda-scratch-ref` now runs 128, 256, and 512 head-dimension
  cases through the CUDA packed/scratch reference coverage.
- CUDA KVarN coverage now includes the wrapped-tail mixed-attention runtime
  test:
  `ctest --test-dir build-kvarn-cuda-nofa-vs -C Release -R "test-kvarn-cuda" --output-on-failure`.
  Latest local result: `test-kvarn-cuda-scratch-ref` and
  `test-kvarn-cuda-mixed-tail` passed.
  Focused CUDA and layout coverage also passed after tightening production
  support to 128- and 256-dimensional K/V heads only:
  `ctest --test-dir build-kvarn-cuda-nofa-vs -C Release -R "test-kvarn-kv|test-kvarn-cuda" --output-on-failure`.
  Latest local result on 2026-06-05 passed after adding F32/F16 KQ-mask graph
  input coverage to `test-kvarn-kv`.
  Static CUDA focused rerun:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-kvarn-kv|test-kvarn-cuda-scratch-ref|test-kvarn-cuda-mixed-tail|test-arg-parser" --output-on-failure`.
  Latest local result on 2026-06-05 passed all four tests.
  `test-kvarn-kv` also now covers the lower-level cache constructor rejection
  for asymmetric K/V head dimensions, matching the model-load compatibility
  guard that rejects such models before runtime allocation.
  The CUDA scratch-reference test now stresses the Qwen3.6 attention topology
  for 256-dimensional heads (`16` query heads, `2` KV heads) and includes a
  sink-only causal, padded-mask case matching the smallest unsafe fused-batch
  model failure shape (`49` queries, `49` sink tokens, `1024`-byte mask row
  stride). That sink-only fused-batch coverage now runs both F16 and F32 KQ
  masks, matching `-fa on` and `-fa off` runtime graph mask storage.
- Shared batch-split and focused KVarN CUDA coverage passed after tightening
  `split_equal()` sequence-set limits:
  `ctest --test-dir build-kvarn-cuda-nofa-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda" --output-on-failure`.
- 256-dim hybrid Qwen3.5 smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -p "Hello" -n 1 -c 256 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed with KVarN allocated on full-attention layers
  `3, 7, 11, 15, 19, 23` and recurrent memory handling SSM layers.
- 256-dim hybrid Qwen3.5 body-record smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -p "<300 hello tokens>" -n 1 -c 384 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed with two KVarN body records per full-attention
  layer and a `6.67 MiB` CUDA KVarN buffer.
- 256-dim hybrid Qwen3.5 bounded prompt-batch smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -p "<240 repeated a tokens>" -n 1 -c 512 -b 512 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --no-display-prompt`.
  Latest local result passed with two KVarN body records per full-attention
  layer, a `6.67 MiB` CUDA KVarN buffer, and prompt throughput `572.6` tok/s.
- 256-dim hybrid Qwen3.5 4B logits-distance comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\.cache\huggingface\hub\models--unsloth--Qwen3.5-4B-GGUF\snapshots\e87f176479d0855a907a41277aca2f8ee7a09523\Qwen3.5-4B-Q4_K_M.gguf -BuildDir build-kvarn-cuda-nofa-vs -Batch 512 -CheckPackedRepeat`.
  Latest local result passed with packed-repeat `NMSE = 0.000E+000` and
  packed-vs-scratch `NMSE = 0.000E+000`.
- 256-dim hybrid Qwen3.5 4B long-context smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\.cache\huggingface\hub\models--unsloth--Qwen3.5-4B-GGUF\snapshots\e87f176479d0855a907a41277aca2f8ee7a09523\Qwen3.5-4B-Q4_K_M.gguf -p "Hello" -n 256 -c 4096 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95 -s 1234 --temp 0 --ignore-eos --no-display-prompt`.
  Latest local result on build `b9529-f663181e4` passed with `30` body records
  per full-attention layer, a `42.72 MiB` CUDA KVarN buffer, and reported
  prompt/generation throughput `132.1`/`85.5` tok/s.
- 256-dim hybrid Qwen3.6 35B A3B MTP smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -p "Hello" -n 1 -c 256 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed on the RTX 5070 with KVarN allocated on
  full-attention layers `3, 7, 11, 15, 19, 23, 27, 31, 35, 39`.
- 256-dim hybrid Qwen3.5 smoke was rerun after adding KVarN graph support for
  reuse-only layers:
  `build-kvarn-cuda-static-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -p Hello -n 1 -c 256 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --no-display-prompt`.
  Latest local result passed with KVarN storage on the same full-attention
  layers and no fallback to normal KV.
- 256-dim hybrid Qwen3.6 35B A3B MTP body-record smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -p "<300 hello tokens>" -n 1 -c 384 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed with two KVarN body records per full-attention
  layer and an `11.11 MiB` CUDA KVarN buffer.
- Gemma 4 12B/26B metadata and KVarN+ISWA runtime validation:
  `C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf` and
  `C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q4_K_XL.gguf` are `gemma4`,
  48 layers, context `131072`, 16 attention heads, no SSM keys, full-attention
  K/V head length `512`, and SWA K/V head length `256`. Gemma 4 26B A4B files
  are also `gemma4`, 30 layers, context `262144`, full-attention K/V head
  length `512`, SWA K/V head length `256`, 128 experts, and 8 active experts.
  KVarN+ISWA now routes these Gemma 4 models without falling back to normal KV
  on full-attention layers.
  Gemma 4 12B short smoke:
  `build-kvarn-cuda-static-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf -p Hello -n 1 -c 256 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --no-display-prompt`.
  Latest local result passed with KVarN storage on full-attention layers
  `5, 11, 17, 23, 29, 35, 41, 47`.
  Gemma 4 12B body-record smoke at `-c 384` with a 300-token repeated prompt
  passed with two body records per KVarN layer and an `8.87 MiB` CUDA KVarN
  buffer. A larger `-c 2048` smoke with a 1500-token repeated prompt passed
  with 14 body records per KVarN layer and a `14.07 MiB` CUDA KVarN buffer.
  The strengthened CLI smoke harness also passed Gemma 4 12B at `-c 512`:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_cuda_smoke.ps1 -Model "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" -BuildDir build-kvarn-cuda-static-vs -CtxList "512" -RtnQuantile 0.95 -MinKvarnLayerLogs 8`.
  Latest local result reported `KVarN CLI log check: PASS, KVarN layer lines =
  16`, `2` body records per full-attention KVarN layer, and a `4.87 MiB`
  metadata estimate for the KVarN portion of the KVarN+ISWA composite.
  Gemma 4 12B server smoke passed via
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf -BuildDir build-kvarn-cuda-static-vs -Port 8152 -Context 256 -Predict 1 -Prompt Hello -RtnQuantile 0.95`.
  Latest rebuilt static-server rerun at `-c 512` with
  `-MinKvarnLayerLogs 8` passed with content `' and'` and
  `KVarN server log check: PASS, KVarN layer lines = 23`, proving the server
  initialized the KVarN+ISWA composite instead of relying on the normal KV
  cache for full-attention layers.
  Gemma 4 26B A4B short and body-record smokes passed on the local 12 GB RTX
  5070, with KVarN storage on full-attention layers `5, 11, 17, 23, 29`. The
  conservative tensor-budget report marks the Q3 file as exceeding a 10.50 GiB
  usable-VRAM budget by `1.51` GiB, but a forced `-ngl 99` run is accepted by
  the current local driver state. Current-build body-record smoke:
  `build-kvarn-cuda-static-vs\bin\Release\llama-completion.exe -fit off -m "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf" -p "Hello" -n 270 -c 512 -ngl 99 --no-warmup --simple-io -no-cnv --no-display-prompt --ignore-eos -fa on --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 -s 1234 --temp 0`.
  Latest local result: KVarN allocated `2` body records per KVarN layer,
  reported an `11.08 MiB` CUDA KVarN buffer and `6.08 MiB` metadata estimate,
  and ran at `14.14` eval tok/s with `graphs reused = 267`. Matching normal KV
  with `--kv-cache-quant none -fa off` ran at `9.12` eval tok/s with
  `graphs reused = 267`.
- Fresh `tg64` benchmark gates on the CUDA FA-off build:
  Qwen2.5 1.5B 128-dim normal KV `202.46` tok/s, KVarN `160.37` tok/s;
  Qwen3.5 0.8B 256-dim hybrid normal KV `360.12` tok/s, KVarN
  `148.25` tok/s; Qwen3.6 35B A3B MTP IQ3 256-dim hybrid normal KV
  `13.51` tok/s, KVarN `13.76` tok/s.
- Current-build 256 benchmark gates on build `f663181e4` with `-fa off`:
  Qwen3.5 4B `tg128` normal KV `142.77` tok/s, KVarN `97.79` tok/s;
  Qwen3.5 4B `pp128` normal KV `1124.22` tok/s, KVarN `245.93` tok/s;
  Qwen3.5 4B body-record-crossing `tg384` normal KV `145.27` tok/s, KVarN
  `34.90` tok/s; Qwen3.6 35B A3B MTP IQ3 `tg64` normal KV `9.10` tok/s,
  KVarN `9.81` tok/s.
- Current-build 512 Gemma 4 KVarN+ISWA benchmark on the static CUDA build:
  `build-kvarn-cuda-static-vs\bin\Release\llama-bench.exe -m C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf -p 512 -n 64 -r 1 -ngl 99 -fa off --no-warmup --kv-cache-quant none,kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95`.
  Latest local result: normal KV `pp512 = 1725.04` tok/s and
  `tg64 = 66.32` tok/s; KVarN `pp512 = 31.80` tok/s and
  `tg64 = 45.59` tok/s. This confirms the routed Gemma path is benchmarkable,
  but prompt prefill remains a major optimization target for 512-dimensional
  KVarN+ISWA.
  Latest strict-harness short result:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 -Model "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" -BuildDir build-kvarn-cuda-static-vs -CaseList "tg16:0:16" -RtnQuantile 0.95 -FlashAttn off -Repetitions 1 -MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5,11,17,23,29,35,41,47"`.
  This passed with normal KV `61.46` tok/s, KVarN `48.93` tok/s, exactly 8
  KVarN layer log lines, and the expected Gemma full-attention layers `5, 11,
  17, 23, 29, 35, 41, 47`.
- Reusable benchmark matrix harness:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 -Model C:\Users\sjake\.cache\huggingface\hub\models--unsloth--Qwen3.5-4B-GGUF\snapshots\e87f176479d0855a907a41277aca2f8ee7a09523\Qwen3.5-4B-Q4_K_M.gguf -BuildDir build-kvarn-cuda-nofa-vs -CaseList "tg64:0:64,pp128:128:0,tg384:0:384" -FlashAttn off -Repetitions 1`.
  The script runs each named `prompt_tokens:generation_tokens` case through
  `llama-bench`, compares `--kv-cache-quant none,kvarn`, fails on any nonzero
  benchmark exit, and writes per-case command/output logs under
  `artifacts\kvarn-bench\<timestamp>`. When `kvarn` is included in
  `--kv-cache-quant`, the harness now also requires
  `llama_kv_cache_kvarn:` initialization logs, at least `-MinKvarnLayerLogs`
  KVarN layer allocation lines, and a KVarN benchmark row, so benchmark
  artifacts cannot hide a normal-KV fallback or an under-routed KVarN layer set.
  `-ExpectedKvarnLayers` additionally requires exact routed layer IDs when a
  model family has a known KVarN layer set.
  Static CUDA log-check artifact:
  `artifacts\kvarn-bench\latest-log-check-qwen25`, generated by
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -BuildDir build-kvarn-cuda-static-vs -CaseList "tg16:0:16" -FlashAttn off -Repetitions 1 -OutputDir artifacts\kvarn-bench\latest-log-check-qwen25`.
  Latest local result: normal KV `217.13` tok/s, KVarN `181.42` tok/s, and
  `KVarN bench log check: PASS, KVarN layer lines = 28`.
  Latest stricter rerun with `-MinKvarnLayerLogs 28` passed on build
  `50f2196a0`: normal KV `212.41` tok/s, KVarN `184.42` tok/s, and 28 KVarN
  layer lines.
  Latest local 256-dim Qwen3.5 4B artifact directory:
  `artifacts\kvarn-bench\20260605-043201`. Results: `tg64` normal KV
  `139.47` tok/s, KVarN `99.06` tok/s; `pp128` normal KV `758.49` tok/s,
  KVarN `281.34` tok/s; body-record-crossing `tg384` normal KV `149.29`
  tok/s, KVarN `35.00` tok/s with two KVarN body records per full-attention
  layer.
- Arg-parser coverage passed:
  `ctest --test-dir build-kvarn-cpu -C Release -R test-arg-parser --output-on-failure`.
- Focused CPU KVarN coverage passed after adding multi-record body-plan seal
  coverage:
  `ctest --test-dir build-kvarn-cpu -C Release -R "test-kvarn-kv|test-arg-parser|test-kvarn-server-load-failure" --output-on-failure`.
- Deterministic CUDA reuse comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_reuse.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs`.
  This runs KVarN twice with the same seed and temperature, first with
  `LLAMA_GRAPH_REUSE_DISABLE=1`, then with graph reuse enabled. It asserts
  identical generated text, requires `graphs reused = 0` in the reference run,
  requires positive graph reuse in the optimized run, and rejects any KVarN run
  whose output lacks `llama_kv_cache_kvarn:` initialization and per-layer logs.
  Latest local result: reuse disabled `76.46` eval tok/s, reuse enabled `79.61`
  eval tok/s with `graphs reused = 268`, normal-KV baseline `355.53` eval tok/s.
  The script accepts `-RtnQuantile 0.95` to run the same deterministic reuse
  check with clipped RTN scaling; latest static short-context quantile result
  was reuse disabled `158.00` eval tok/s with `graphs reused = 0`, reuse enabled
  `167.94` eval tok/s with `graphs reused = 31`, normal-KV baseline `275.04`
  eval tok/s, and 56 KVarN layer log lines in each KVarN run.
  Latest stricter short rerun with `-MinKvarnLayerLogs 28` passed: reuse
  disabled `146.45` eval tok/s with `graphs reused = 0`, reuse enabled
  `153.36` eval tok/s with `graphs reused = 15`, and 56 KVarN layer log lines
  in each KVarN run.
- Runtime packed-vs-scratch attention comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_scratch_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs -RtnQuantile 0.95`.
  This runs the same deterministic long decode once with packed KVarN mixed
  attention and once with `LLAMA_KVARN_ATTN_REF_SCRATCH=1`, then asserts
  identical generated text. The harness now rejects any successful completion
  whose output lacks `llama_kv_cache_kvarn:` initialization logs, so
  packed-vs-scratch parity cannot pass on a normal-KV fallback. Latest static
  128-dim regression result with `-Predict 32 -Context 256`: both packed and
  scratch-reference runs logged `KVarN completion log check: PASS, KVarN layer
  lines = 56`; packed attention `167.28` eval tok/s and scratch-reference
  attention `165.99` eval tok/s, both with `graphs reused = 31`.
  Latest stricter short Qwen2.5 rerun with `-MinKvarnLayerLogs 28` passed:
  packed attention `83.77` eval tok/s, scratch-reference attention `101.51`
  eval tok/s, both with `graphs reused = 15` and 56 KVarN layer log lines.
  Latest strict Gemma 4 12B 512-dim KVarN+ISWA generated-text rerun with
  `-MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5,11,17,23,29,35,41,47"`
  passed: packed attention `36.89` eval tok/s, scratch-reference attention
  `37.75` eval tok/s, both with `graphs reused = 14`, 16 KVarN layer log
  lines, and the exact expected full-attention layer IDs.
  Historical longer local result with fused packed attention:
  packed attention `78.98` eval tok/s, scratch-reference attention `78.13`
  eval tok/s, both with `graphs reused = 268`. Forcing the old split kernels
  with `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` also passed, with packed attention
  `80.50` eval tok/s and scratch-reference attention `80.88` eval tok/s, both
  with
  `graphs reused = 268`.
- Runtime fused-vs-split packed attention comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_fused_split.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs -RtnQuantile 0.95`.
  This runs the production default packed attention path, explicitly forces
  serial fused attention with `LLAMA_KVARN_ATTN_SERIAL_FUSED=1`, and then
  forces the previous split score/AV kernels with
  `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1`, asserting deterministic generated-text
  parity. Pass `-IncludeUnsafeFusedBatch` only for the explicitly unsafe
  multi-block fused-batch diagnostic. The harness now rejects any successful
  completion whose output lacks `llama_kv_cache_kvarn:` initialization logs, so
  dispatch parity cannot pass on a normal-KV fallback. Latest static 128-dim
  regression result with `-Predict 32 -Context 256`: all three runs logged
  `KVarN completion log check: PASS, KVarN layer lines = 56`; default
  attention `172.69` eval tok/s, explicit serial fused `98.34` eval tok/s,
  split attention `89.41` eval tok/s, all with `graphs reused = 31`.
  Latest stricter short rerun with `-MinKvarnLayerLogs 28` passed: default
  attention `83.77` eval tok/s, explicit serial fused `77.50` eval tok/s,
  split attention `84.78` eval tok/s, all with `graphs reused = 15` and 56
  KVarN layer log lines.
  Historical pre-routing result: batched fused
  attention `181.47` eval tok/s, serial fused attention `99.00` eval tok/s,
  split attention `79.92` eval tok/s, all with `graphs reused = 268`.
- Runtime packed-vs-scratch logits-distance comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs`.
  This saves logits from the packed KVarN path with `llama-results`, reruns
  the same prompt with `LLAMA_KVARN_ATTN_REF_SCRATCH=1 --check`, and requires
  llama.cpp's logits NMSE threshold to pass. Latest local result:
  `KVarN packed-vs-scratch logits: PASS, NMSE = 0.000E+000`. The
  `-CheckPackedRepeat` diagnostic also passed with packed-repeat
  `NMSE = 0.000E+000`. The harness now parses `NaN`/infinity NMSE values
  explicitly so diagnostic failures can be reported without script parser
  failures. It also accepts `-CheckPackedSplit`, which reruns the saved logits
  against `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` to distinguish packed fused-vs-split
  divergence from packed-vs-scratch divergence. The harness now rejects any
  successful `llama-results` invocation whose output lacks
  `llama_kv_cache_kvarn:` initialization logs, so logits NMSE passes cannot
  hide a normal-KV fallback. It also accepts `-MinKvarnLayerLogs` and
  `-ExpectedKvarnLayers` to require the expected KVarN layer-routing count and
  exact layer IDs for a model family. Static CUDA rerun:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -BuildDir build-kvarn-cuda-static-vs -Context 512 -Batch 512 -Repeat 16 -CheckPackedRepeat -CheckPackedSplit -FlashAttn off`.
  Latest local result logged `KVarN llama-results log check: PASS, KVarN layer
  lines = 56` for packed save, packed repeat, split-kernel check, and
  scratch-reference check; packed-repeat, packed-vs-split, and
  packed-vs-scratch all passed with `NMSE = 0.000E+000`.
  Latest strict short rerun:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -BuildDir build-kvarn-cuda-static-vs -Context 256 -Batch 256 -Repeat 2 -CheckPackedRepeat -CheckPackedSplit -FlashAttn off -MinKvarnLayerLogs 28 -ExpectedKvarnLayers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"`.
  Packed save, packed repeat, split-kernel, and scratch-reference checks all
  logged 56 KVarN layer lines, passed the exact layer-ID check for layers
  `0..27`, and passed with `NMSE = 0.000E+000`.
- 256-dim runtime packed-vs-scratch logits-distance comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -BuildDir build-kvarn-cuda-nofa-vs -Batch 512`.
  Latest local result passed on the bounded prompt-batch path with
  packed-repeat `NMSE = 0.000E+000` and packed-vs-scratch
  `NMSE = 0.000E+000`.
- 512-dim Gemma 4 KVarN+ISWA packed-vs-scratch logits-distance comparisons:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf -BuildDir build-kvarn-cuda-static-vs -Context 384 -Batch 512 -Repeat 4 -FlashAttn off -CheckPackedRepeat`.
  Latest local 12B result passed with packed-repeat `NMSE = 0.000E+000` and
  packed-vs-scratch `NMSE = 0.000E+000`. Hardened harness rerun with
  `-Repeat 2` also passed and logged `KVarN llama-results log check: PASS,
  KVarN layer lines = 16` for packed save, packed repeat, and
  scratch-reference check, proving the KVarN+ISWA logits comparison used the
  KVarN full-attention layer storage rather than normal KV.
  Latest strict short rerun:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" -BuildDir build-kvarn-cuda-static-vs -Context 256 -Batch 256 -Repeat 1 -CheckPackedRepeat -FlashAttn off -MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5,11,17,23,29,35,41,47"`.
  Packed save, packed repeat, and scratch-reference checks all logged 16 KVarN
  layer lines, passed the exact full-attention layer-ID check, and passed with
  `NMSE = 0.000E+000`.
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf -BuildDir build-kvarn-cuda-static-vs -Context 384 -Batch 512 -Repeat 2 -FlashAttn off -CheckPackedRepeat`.
  Latest local 26B A4B result also passed with packed-repeat
  `NMSE = 0.000E+000` and packed-vs-scratch `NMSE = 0.000E+000`.
- Unsupported runtime-mode rejection check:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_unsupported_smoke.ps1 -SupportedModel C:\Users\sjake\.cache\huggingface\hub\models--unsloth--Qwen3.5-4B-GGUF\snapshots\e87f176479d0855a907a41277aca2f8ee7a09523\Qwen3.5-4B-Q4_K_M.gguf -BuildDir build-kvarn-cuda-static-vs`.
  Latest local result failed forced fused-batch initialization before graph
  execution with
  `KVarN forced fused-batch attention is disabled because multi-query correctness is not proven`
  and also verified unsafe `LLAMA_KVARN_DEBUG_UBATCH=129` rejection, invalid
  `LLAMA_KVARN_DEBUG_UBATCH=0` positive-integer rejection, and server
  `--parallel 2` rejection. Latest static-build rerun on the Qwen2.5 1.5B
  regression model passed all five checks:
  `KVarN forced fused-batch rejection: PASS`,
  `KVarN invalid scratch-reference env rejection: PASS`,
  `KVarN unsafe debug ubatch rejection: PASS`,
  `KVarN invalid debug ubatch rejection: PASS`, and
  `KVarN server multi-slot rejection: PASS`. The optional
  `-UnsupportedDimModel` argument is now reserved for truly unsupported head
  dimensions or non-Gemma SWA/ISWA fixtures; Gemma 4 is a supported KVarN+ISWA
  path. Latest KVarN+ISWA rerun with
  `-SupportedIswaModel C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf`
  also passed `KVarN+ISWA invalid debug ubatch rejection: PASS`, proving the
  Gemma composite path no longer silently ignores malformed
  `LLAMA_KVARN_DEBUG_UBATCH` values.
- 256-dim Qwen3.6 runtime packed-vs-scratch logits-distance comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -BuildDir build-kvarn-cuda-nofa-vs -Context 384 -Batch 512 -Repeat 24`.
  Latest local result passed with `NMSE = 0.000E+000`; this path now uses
  bounded MoE KVarN prompt ubatches through the serial fused multi-query CUDA
  kernel.
  The logits script also accepts `-FlashAttn on|off|auto`; with `-FlashAttn off`
  the same Qwen3.6 serial-fused production path passed packed-repeat and
  packed-vs-scratch checks at `NMSE = 0.000E+000`.
  The packed-repeat diagnostic
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -BuildDir build-kvarn-cuda-nofa-vs -Context 384 -Batch 512 -Repeat 24 -DebugUbatch 128 -CheckPackedRepeat`
  also passed with packed-repeat `NMSE = 0.000E+000` and packed-vs-scratch
  `NMSE = 0.000E+000`. The earlier false divergence came from overallocating
  scratch-reference workspace for inactive body records, perturbing large MoE
  prompt graphs before the scratch path was actually active.
  `-DebugUbatch 128 -PackedFusedBatch -CheckPackedRepeat` exposed a real
  forced fused-batch divergence (`NMSE` around `1e-2` to `2.5e-2`), so the
  runtime now rejects `LLAMA_KVARN_ATTN_FUSED_BATCH=1` explicitly.
  A minimized Qwen3.6 fused-batch run under CUDA `compute-sanitizer --tool
  memcheck` found an invalid dynamic shared-memory read in
  `kvarn_attn_mixed_f16_fused_batch_kernel`; padding the KVarN attention
  shared-memory allocation fixed that sanitizer error. Latest sanitizer result
  on the short Qwen3.6 fused-batch reproducer and on the body-record prompt
  shape reports `ERROR SUMMARY: 0 errors`. The runtime logits guard still
  fails for forced fused-batch Qwen3.6 at packed-repeat `NMSE = 2.443e-03`, so
  the production guard remains in place until the remaining correctness issue
  is fixed.
  Follow-up narrowing: forced fused-batch Qwen3.6 passes packed-repeat and
  packed-vs-scratch at `-c 256 -Repeat 1` and `-c 256 -Repeat 4`, and
  Qwen3.5 0.8B passes the same forced fused-batch body-record logits guard at
  `-c 384 -Repeat 4`. Qwen3.6 still fails at
  `-c 384 -Repeat 4 -DebugUbatch 128` with packed-repeat
  `NMSE = 4.865e-03`; disabling CUDA graph capture still fails at
  `NMSE = 7.621e-03`, and disabling graph reuse still fails at
  `NMSE = 1.237e-03`. The standalone CUDA primitive test now matches the real
  failing prompt-batch shape more closely with a 49-query Qwen3.6-shaped
  body-record case, and that primitive test passes, so the remaining issue is
  above the primitive arithmetic or depends on full-runtime graph/state
  interaction. Using the diagnostic unsafe override now emits an explicit
  runtime warning before allowing the path.
  With the unsafe diagnostic override enabled after removing an unnecessary
  scratch/probability write from the fused-batch kernel, Qwen3.6 still failed
  packed repeat at `NMSE = 1.569e-02`. Disabling graph reuse still failed at
  `NMSE = 1.269e-02`, and disabling CUDA graphs with
  `GGML_CUDA_DISABLE_GRAPHS=1` still failed a shorter repeat-4 fused-vs-scratch
  check at `NMSE = 1.556e-03`, so the remaining issue is not stale graph reuse
  or CUDA graph capture. Running the repeat-4 unsafe fused-vs-scratch check
  with `-FlashAttn off` still failed at `NMSE = 8.676e-04`, so F16 KQ-mask
  storage is not the root cause even though it affects the error magnitude.
  After expanding the standalone CUDA test to cover padded 1024-token full
  mixed-attention KQ-mask strides, the primitive still passed but the same
  Qwen3.6 repeat-4 unsafe fused-vs-scratch diagnostic with `-FlashAttn off`
  failed at `NMSE = 3.998e-04`. The supported split-kernel diagnostic under
  the same short setup passed packed-repeat and packed-vs-scratch checks at
  `NMSE = 0.000E+000`, keeping a reference comparison path intact while
  narrowing the fused issue to graph/model integration rather than isolated
  mask stride.
  The direct `-PackedFusedBatch -CheckPackedSplit` logits diagnostic on the
  same Qwen3.6 repeat-4 setup failed forced fused-vs-forced split at
  `NMSE = 7.774e-03`, so the remaining unsafe path is now isolated to fused
  packed attention relative to the supported split packed implementation.
  A standalone CUDA primitive test now covers the same 512-token active-window
  topology (`128` sink + `2*128` body + `128` tail, `128` queries, `16` query
  heads, `2` KV heads) and still passes split, forced fused, and CPU-reference
  comparisons, so the model-level failure is above a direct primitive call and
  remains tied to full graph/runtime integration.
  `LLAMA_KVARN_ATTN_TRACE=1` with `LLAMA_KVARN_ATTN_TRACE_LIMIT=N` now traces
  both the graph update and CUDA backend dispatch for `GGML_OP_KVARN_ATTN_MIXED`.
  On this Windows host, freshly rebuilt shared `llama-results.exe`/DLLs were
  blocked by Smart App Control / Device Guard (`CodeIntegrity` event 3077:
  `ggml-base.dll` did not meet Enterprise signing requirements), so traced
  model diagnostics used a separate static CUDA build:
  `cmake -S . -B build-kvarn-cuda-static-vs -G "Visual Studio 17 2022" -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_CUDA_FA=OFF -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CCACHE=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON`.
  Static llama builds must compile the `llama` target with `LLAMA_BUILD`;
  otherwise KVarN runtime storage falls back to CPU buffers and body-store
  graph execution can fail with
  `pre-allocated tensor ... in a buffer (CPU) that cannot run the operation (KVARN_STORE_BODY)`.
  After fixing the static target definition, traced static runs report KVarN
  storage dev `CUDA0`.
  The traced Qwen3.6 repeat-4 unsafe fused-vs-split diagnostic showed the first
  divergent model path entering CUDA as `mode=fused-batch` versus `mode=split`
  with identical tensor geometry: `n_queries=2`, `n_head=16`, `n_head_kv=2`,
  `n_sink=2`, `n_records=0`, `n_pending=0`, `n_tail=0`, `head_dim=256`,
  F32 mask, and a 512-token mask stride. Disabling CUDA graphs with
  `GGML_CUDA_DISABLE_GRAPHS=1` still failed (`NMSE = 2.061e-03` in the static
  traced run), so CUDA graph capture/replay is not the root cause. The direct
  CUDA primitive test now also covers this two-token sink-only shape plus the
  earlier 49-token sink-only shape and still passes split, forced fused,
  fused-vs-split, and CPU-reference checks. Current evidence points to small
  fused-vs-split attention math drift being amplified by the Qwen3.6 MoE graph;
  the production path uses the per-row serial fused multi-query KVarN kernel.
  The per-row serial fused kernel is now the default for multi-query KVarN
  prompt batches. `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` still forces the previous
  split score/AV kernels, and `LLAMA_KVARN_ATTN_FUSED_BATCH=1` remains the
  explicitly rejected unsafe multi-block diagnostic path unless paired with
  `LLAMA_KVARN_UNSAFE_ALLOW_FUSED_BATCH=1`. Before the routing fix,
  `LLAMA_KVARN_ATTN_SERIAL_FUSED=1` was masked by the `n_queries > 1` split
  guard; after fixing that routing, the traced Qwen3.6 repeat-4 serial-fused
  run entered CUDA as `mode=serial-fused` and passed both serial-fused-vs-split
  and serial-fused-vs-scratch at `NMSE = 0.000E+000`. The remaining unsafe
  correctness issue is specific to the multi-block
  `LLAMA_KVARN_ATTN_FUSED_BATCH=1` path, not the per-row fused kernel.
  Static Qwen3.5 0.8B benchmark at `-p 128 -n 64 -r 1 -fa off` measured the
  old split prompt path at `553.86` pp t/s and `125.18` tg t/s before the static
  build fix; after the static build fix and serial-fused default routing, the
  same command measured `592.08` pp t/s and `154.04` tg t/s with KVarN storage
  on `CUDA0`. The production routing uses serial fused only for multi-query
  prompt batches, preserving the existing single-query fused generation path.
  Static normal-vs-KVarN 256-dim benchmark on Qwen3.5 0.8B with active body
  records:
  `build-kvarn-cuda-static-vs\bin\Release\llama-bench.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -p 512 -n 128 -r 1 -ngl 99 -fa off --no-warmup --kv-cache-quant none,kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95`.
  Latest local result: normal KV `pp512 = 4023.35` tok/s and
  `tg128 = 437.93` tok/s; KVarN `pp512 = 77.03` tok/s and
  `tg128 = 137.07` tok/s. The KVarN run allocated `2` body records per KVarN
  layer on `CUDA0`, with reported KVarN buffer size `6.67 MiB` and metadata
  estimate `3.67 MiB`.
  The same unsafe fused-batch path passed Qwen3.5 0.8B repeats 4 through 32
  after the scratch/probability write removal, with the worst observed
  `NMSE = 4.735e-07`, so Qwen3.6 remains the active reproducer.
  CUDA primitive coverage now forces `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` for
  the split baselines before comparing against the diagnostic fused-batch path,
  so the MHA mixed, Qwen3.6-shaped, and sink-only primitive cases really cover
  split-vs-fused instead of serial-vs-fused. A row-local score workspace
  experiment for fused-batch was rejected: the Qwen3.6 repeat-4
  packed-vs-scratch run still failed, worsening from the previous
  `NMSE = 7.686e-04` to `NMSE = 1.495e-03`. The production guard remains the
  correct behavior until a different multi-block fused implementation passes
  Qwen3.6 model-level logits.
- 128-dim Qwen2.5 regression on the corrected static CUDA build:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-static-vs -Batch 512 -Repeat 4 -CheckPackedSplit -FlashAttn off`.
  Latest local result passed with packed-vs-split `NMSE = 0.000E+000` and
  packed-vs-scratch `NMSE = 0.000E+000`.
- Exact KVarN layer-routing evidence is now available in all runtime smoke and
  comparison harnesses through `-ExpectedKvarnLayers`. Latest local Qwen2.5
  exact fused/split dispatch parity:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_fused_split.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -BuildDir build-kvarn-cuda-static-vs -Context 256 -Predict 16 -RtnQuantile 0.95 -Prompt Hello -MinKvarnLayerLogs 28 -ExpectedKvarnLayers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"`.
  Latest local result passed with exact layers `0..27`, 56 KVarN layer log
  lines per run, graph reuse `15`, and eval rates: default `154.94` tok/s,
  serial fused `93.16` tok/s, split `85.58` tok/s.
- Latest local Qwen2.5 exact graph-reuse parity:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_reuse.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -BuildDir build-kvarn-cuda-static-vs -Context 256 -Predict 16 -RtnQuantile 0.95 -Prompt Hello -MinKvarnLayerLogs 28 -ExpectedKvarnLayers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27" -SkipNormalBaseline`.
  Latest local result passed with exact layers `0..27`, reuse-disabled
  `graphs reused = 0` at `141.05` eval tok/s, and reuse-enabled
  `graphs reused = 15` at `156.85` eval tok/s.
- Latest local Gemma 4 12B exact server smoke:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" -BuildDir build-kvarn-cuda-static-vs -Port 8164 -Context 256 -Predict 1 -Prompt Hello -RtnQuantile 0.95 -MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5,11,17,23,29,35,41,47"`.
  Latest local result passed with exact KVarN full-attention layers
  `5,11,17,23,29,35,41,47`, 16 KVarN layer log lines, and completion content
  `" and"`.
- Server smoke passed:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-static-vs`.
  Latest local result: `KVarN server smoke: PASS, content = '.'`.
- 256-dim server smoke passed:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -BuildDir build-kvarn-cuda-static-vs -Port 8135 -CheckSlotSaveRejection`.
  Latest local result: `KVarN server smoke: PASS, content = ','` and
  `KVarN slot save rejection: PASS`.
- 256-dim Qwen3.6 35B A3B MTP server smoke passed on the static CUDA build:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" -BuildDir build-kvarn-cuda-static-vs -Port 8146 -Context 384 -Predict 1 -Prompt "Hello" -RtnQuantile 0.95`.
  Latest local result: `KVarN server smoke: PASS, content = ','`.
- Long-context smoke observed with `llama-cli` at `-c 4096`, `-n 768`,
  `--kv-cache-quant kvarn`, and `--kvarn-rtn-quantile 0.95`. The run
  allocated `30` KVarN body records per layer and reported generation
  throughput `186.4` tok/s before exiting after EOS.
- KVarN no longer has a fake runtime dependency on the `--flash-attn` flag.
  This Windows build is compiled with `GGML_CUDA_NO_FA`, while KVarN uses the
  custom mixed-attention op. A smoke run with `-fa off` completed successfully
  and reported prompt `210.2` tok/s and generation `188.2` tok/s.
- Standard benchmark comparison passed:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-bench.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p 0 -n 384 -r 1 -ngl 99 -fa on --no-warmup --kv-cache-quant none,kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95`.
  Latest local result: normal KV `151.20` tok/s, KVarN `49.62` tok/s.

Required integration path:

1. Finish optimization of KVarN prompt batches. `GGML_OP_KVARN_ATTN_MIXED`
   carries an optional KQ/causal mask in `src[10]`, and runtime preparation now
   admits bounded prompt ubatches for dense, hybrid, and MoE models.
   Correctness is currently maintained by using the per-row serial fused CUDA
   kernel for `n_queries > 1`; the faster multi-block fused packed-attention
   path is disabled with an explicit unsupported-mode error until it can replace
   the serial fused path for all prompt batches.
2. Finish prompt-batch sealing semantics. The graph builder now collects all
   seal records in an ubatch and emits store ops for each record, and the body
   plan has multi-record seal coverage. Bounded production prompt ubatches now
   exercise this path; larger-than-tail-ring prompt chunks remain out of scope
   until same-graph tail overwrite hazards are handled explicitly.
3. Promote the device scratch-reference primitives into a graph path that
   materializes body records into scratch K/V tensors before
   `ggml_flash_attn_ext`. The standalone CUDA scratch-dequant primitives and
   runtime-selectable scratch mixed-attention path are present and parity-tested
   against the packed mixed-attention path, but the graph path still uses a
   custom KVarN attention op instead of `ggml_flash_attn_ext`.
4. Add a CUDA fused path by inlining KVarN unpack/scale absorption into the
   flash-attention K/V load path. Standalone packed-K QK scoring and packed-V
  AV accumulation primitives are present and parity-tested. A combined
  packed-body attention primitive is present and parity-tested for a single
  KVarN body record, and a multi-record variant is present and parity-tested
  across several body records. Batched multi-query and mixed sink/body/tail
  variants are also present and parity-tested. The runtime packed
  mixed-attention path now has a batched fused score/softmax/AV CUDA kernel
  that passes synthetic tests, but that forced runtime mode is disabled because
  Qwen3.6 prompt-batch logits still diverge. The mixed op is still a custom
  attention op rather than a flash-attention load-path integration. Until that
  integration exists, KVarN must not pretend that llama.cpp Flash Attention is
  required or used.
5. Compare fused output against the reference scratch path with fixed seeds
   and logits-distance thresholds. The current custom packed attention op now
   has a live packed-vs-scratch logits NMSE guard via
   `compare_cuda_logits_ref.ps1`; the remaining production gate is the same
   threshold check after flash-attention load-path fusion exists.
6. Implement multi-stream/server slot support. Until then, KVarN rejects
   explicit `--parallel` values above `1` and lowers server auto-parallel to
   one slot.
7. Optimize performance. `llama-bench` now accepts `--kv-cache-quant
   none|kvarn`, `--kvarn-preset`, and `--kvarn-rtn-quantile`, so production
   performance gates can use the standard benchmark tool; current KVarN rows
   remain materially slower than normal KV until flash-attention load-path
   fusion is implemented.

The current `compare_cuda_reuse.ps1` harness is not a replacement for the
future fused-versus-scratch logits-distance test. It is a guard for the current
decode implementation: graph reuse must not change deterministic KVarN output,
and optimization work should continue to use it as a fast regression check.
`compare_cuda_scratch_ref.ps1` is a stronger live-runtime generated-text guard
for the current packed attention op, and `compare_cuda_logits_ref.ps1` adds the
corresponding logits-distance guard.

For `llama-cli` smoke runs, use `--single-turn`; default conversation mode can
stay interactive after generation and causes external harnesses to terminate it
with `STATUS_CONTROL_C_EXIT`.

Local smoke models may fail KVarN initialization before graph construction if
they do not match the production constraints. Expected guarded failures include
K/V head dimensions other than 128, 256, or 512, MLA, non-Gemma SWA/ISWA,
unsupported backend placement, attention rotations/KQ bias/sinks, and other
explicit KVarN graph-backend guards.
