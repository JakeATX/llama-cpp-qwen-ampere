# KVarN KV-cache backend notes

This branch keeps KVarN separate from `ggml_type`. `--kv-cache-quant kvarn`
selects a KV-cache backend mode and uses `llama_kvarn_params` for layout and
runtime policy.

Current implemented pieces:

- Public API and common CLI flags for `--kv-cache-quant none|kvarn`.
- `kvarn_k4v2_g128` preset defaults: group 128, 4-bit K, 2-bit V, 128 sink
  tokens, 128 tail tokens, 16 Sinkhorn iterations.
- Runtime K/V head dimensions of 128, 256, and 512 are supported by
  validation, layout tests, and CUDA parity tests. Local `llama-cli` smokes
  currently cover 128- and 256-dimensional K/V heads. 256 remains the primary
  Qwen acceptance path; 512 is included because the local Gemma 4 12B files
  report 512-dimensional K/V heads, but Gemma4 uses SWA/ISWA cache routing and
  is explicitly rejected until KVarN has an ISWA cache implementation.
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
- The runtime packed mixed-attention path now has a batched fused CUDA kernel
  that combines score calculation, softmax normalization, and AV accumulation
  for all query heads in one grid launch. Set `LLAMA_KVARN_ATTN_SERIAL_FUSED=1`
  to force the previous per-query-head fused launch, or
  `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` to force the earlier split score/AV
  kernels for A/B debugging.
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
- KVarN runtime memory/context skeleton with native slot metadata, sequence
  operations, memory estimates, and production-shaped per-layer/per-KV-head
  storage. Runtime storage keeps sink/tail tokens in FP16 and seals full body
  groups into packed KVarN body records plus scale metadata. It does not
  allocate or use the normal KV cache as a fallback.
- KVarN batch preparation splits caller batches into one-token ubatches. This
  keeps prompt processing on the decode-shaped graph path until unsplit
  multi-token prompt storage/sealing is safely enabled and tested.
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
  KQ mask for KVarN graph inputs, but production runtime preparation still
  splits actual work into one-token ubatches until unsplit prompt storage and
  sealing are validated. The graph computes active sink/body/pending/tail counts
  and the wrapped-tail start slot for each decode token. Context reserve graphs
  may be built for larger synthetic ubatches.
- Graph construction writes FP16 sink/tail, stages evicted FP16 tail rows into
  FP32 pending body slots, emits packed K/V body-store nodes when a graph
  completes one body record, and uses KVarN mixed attention for one-token
  decode graphs. It currently refuses multi-token prompt batches and graphs
  that would seal multiple body records at once.
- When `LLAMA_KVARN_ATTN_REF_SCRATCH=1` is set during graph construction,
  `kvarn_attn_scores` is sized for score probabilities plus K/V body scratch
  for every allocated body record in the current layer. Graph reuse validates
  this larger workspace before reusing scratch-reference graphs.
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
  The standalone CUDA test covers store/dequant, packed body attention,
  sink/body/tail mixed attention, pending tokens, batched F16 mixed attention,
  wrapped-tail decode order, multi-record scratch dequant, and packed
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
storage through CUDA mixed attention for one-token decode. KVarN graph reserve
can build synthetic multi-token graphs for context initialization, but runtime
execution still uses one-token ubatches until causal prompt-mask support is
implemented. This trades prompt speed for correctness.

Verified local smoke:

- Compatible model downloaded with
  `hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF`.
- Reproducible local model metadata discovery:
  `python scripts\kvarn\discover_models.py C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`.
  Latest local result reports Gemma4 as 512-dimensional K/V plus
  `swa/iswa-likely`, Qwen3.5 0.8B as `primary-256,hybrid-ssm`, and Qwen3.6
  35B A3B MTP IQ3 as `primary-256,hybrid-ssm,moe`.
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
- Standard benchmark smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-bench.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p 0 -n 64 -r 1 -ngl 99 -fa on --no-warmup --kv-cache-quant none,kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95`.
  This verifies the `llama-bench` KVarN option plumbing. Latest local result:
  normal KV `154.75` tok/s, KVarN `84.48` tok/s for `tg64`.
- Standard prompt-processing benchmark:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-bench.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -p 64 -n 0 -r 1 -ngl 99 -fa on --no-warmup --kv-cache-quant none,kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile 0.95`.
  Latest local result: normal KV `1081.21` tok/s, KVarN `98.22` tok/s for
  `pp64`. This is expected to be slow until masked multi-token prompt attention
  replaces the current one-token KVarN ubatch splitting path.
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
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs`.
  This starts `llama-server` on localhost, waits for `/health`, posts one
  deterministic `/completion` request, and always stops the child process.
  Latest local result: `KVarN server smoke: PASS, content = '.'`.
- Larger context allocation/decode smoke:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_cuda_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs -CtxList "512 1024 2048" -RtnQuantile 0.95`.
  FP16 and KVarN smoke paths passed for all three context sizes. Reported
  KVarN cache estimates were `8.64 MiB` at 512 tokens, `11.92 MiB` at 1024
  tokens, and `18.48 MiB` at 2048 tokens.
- Memory estimator for Qwen2.5 1.5B geometry (`28` layers, `2` KV heads,
  `128` head dim, K4/V2/group128) reports KVarN body-record totals of
  `27,525,120` bytes at 4K context, `55,050,240` bytes at 8K context, and
  `110,100,480` bytes at 16K context, or `120` amortized bytes/token/layer/KV
  head. These estimator numbers are body-record estimates and do not include
  every runtime allocation reported by llama.cpp logs.
- Memory estimator for 256-dim hybrid Qwen3.5 full-attention geometry (`6`
  layers, `2` KV heads, `256` head dim, K4/V2/group128) reports KVarN totals
  of `10.69 MiB` at 4K context and `21.38 MiB` at 8K context. The same geometry
  in FP16 KV would be `48.00 MiB` and `96.00 MiB`.
- Memory estimator for 256-dim hybrid Qwen3.6 35B A3B MTP full-attention
  geometry (`10` layers, `2` KV heads, `256` head dim, K4/V2/group128) reports
  KVarN totals of `17.81 MiB` at 4K context and `35.62 MiB` at 8K context. The
  same geometry in FP16 KV would be `80.00 MiB` and `160.00 MiB`.
- Explicit multi-slot startup now fails cleanly before model load with:
  `KVarN currently supports only --parallel 1`.
- Focused tests passed:
  `ctest --test-dir build-kvarn-cpu -C Release -R "test-kvarn-kv|test-kvarn-server-load-failure" --output-on-failure`
  and
  `ctest --test-dir build-kvarn-cuda-nofa-vs -C Release -R test-kvarn-cuda-scratch-ref --output-on-failure`.
- CUDA KVarN coverage now includes the wrapped-tail mixed-attention runtime
  test:
  `ctest --test-dir build-kvarn-cuda-nofa-vs -C Release -R "test-kvarn-cuda" --output-on-failure`.
  Latest local result: `test-kvarn-cuda-scratch-ref` and
  `test-kvarn-cuda-mixed-tail` passed.
- 256-dim hybrid Qwen3.5 smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -p "Hello" -n 1 -c 256 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed with KVarN allocated on full-attention layers
  `3, 7, 11, 15, 19, 23` and recurrent memory handling SSM layers.
- 256-dim hybrid Qwen3.5 body-record smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -p "<300 hello tokens>" -n 1 -c 384 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed with two KVarN body records per full-attention
  layer and a `6.67 MiB` CUDA KVarN buffer.
- 256-dim hybrid Qwen3.6 35B A3B MTP smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -p "Hello" -n 1 -c 256 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed on the RTX 5070 with KVarN allocated on
  full-attention layers `3, 7, 11, 15, 19, 23, 27, 31, 35, 39`.
- 256-dim hybrid Qwen3.6 35B A3B MTP body-record smoke:
  `build-kvarn-cuda-nofa-vs\bin\Release\llama-cli.exe -m C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -p "<300 hello tokens>" -n 1 -c 384 -ngl 99 --no-warmup --simple-io --single-turn -fa off --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128`.
  Latest local result passed with two KVarN body records per full-attention
  layer and an `11.11 MiB` CUDA KVarN buffer.
- Gemma 4 12B dense metadata discovered locally:
  `C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf` and
  `C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q4_K_XL.gguf` are `gemma4`,
  48 layers, context `131072`, 16 attention heads, no SSM keys, and K/V head
  length `512`. They are not 256-dimensional despite the original target
  assumption. KVarN now rejects these files cleanly with
  `KVarN backend does not support SWA/ISWA models yet`; earlier builds could
  reach an invalid ISWA graph cast and crash with `0xC0000005`.
- Fresh `tg64` benchmark gates on the CUDA FA-off build:
  Qwen2.5 1.5B 128-dim normal KV `202.46` tok/s, KVarN `160.37` tok/s;
  Qwen3.5 0.8B 256-dim hybrid normal KV `360.12` tok/s, KVarN
  `148.25` tok/s; Qwen3.6 35B A3B MTP IQ3 256-dim hybrid normal KV
  `13.51` tok/s, KVarN `13.76` tok/s.
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
  and requires positive graph reuse in the optimized run. Latest local result:
  reuse disabled `76.46` eval tok/s, reuse enabled `79.61` eval tok/s with
  `graphs reused = 268`, normal-KV baseline `355.53` eval tok/s.
  The script accepts `-RtnQuantile 0.95` to run the same deterministic reuse
  check with clipped RTN scaling; latest quantile result was reuse disabled
  `71.43` eval tok/s, reuse enabled `74.67` eval tok/s with
  `graphs reused = 268`.
- Runtime packed-vs-scratch attention comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_scratch_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs -RtnQuantile 0.95`.
  This runs the same deterministic long decode once with packed KVarN mixed
  attention and once with `LLAMA_KVARN_ATTN_REF_SCRATCH=1`, then asserts
  identical generated text. Latest local result with fused packed attention:
  packed attention `78.98` eval tok/s, scratch-reference attention `78.13`
  eval tok/s, both with `graphs reused = 268`. Forcing the old split kernels
  with `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` also passed, with packed attention
  `80.50` eval tok/s and scratch-reference attention `80.88` eval tok/s, both
  with
  `graphs reused = 268`.
- Runtime fused-vs-split packed attention comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_fused_split.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs -RtnQuantile 0.95`.
  This runs default batched fused packed attention, forces serial fused
  attention with `LLAMA_KVARN_ATTN_SERIAL_FUSED=1`, and then forces the
  previous split score/AV kernels with `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1`,
  asserting deterministic generated-text parity. Latest local result: batched
  fused attention `181.47` eval tok/s, serial fused attention `99.00` eval
  tok/s, split attention `79.92` eval tok/s, all with
  `graphs reused = 268`.
- Runtime packed-vs-scratch logits-distance comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs`.
  This saves logits from the packed KVarN path with `llama-results`, reruns
  the same prompt with `LLAMA_KVARN_ATTN_REF_SCRATCH=1 --check`, and requires
  llama.cpp's logits NMSE threshold to pass. Latest local result:
  `KVarN packed-vs-scratch logits: PASS, NMSE = 0.000E+000`.
- 256-dim runtime packed-vs-scratch logits-distance comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -BuildDir build-kvarn-cuda-nofa-vs`.
  Latest local result also passed with `NMSE = 0.000E+000`.
- 256-dim Qwen3.6 runtime packed-vs-scratch logits-distance comparison:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -BuildDir build-kvarn-cuda-nofa-vs -Context 384 -Repeat 24`.
  Latest local result passed with `NMSE = 0.000E+000`.
- Server smoke passed:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf -BuildDir build-kvarn-cuda-nofa-vs`.
  Latest local result: `KVarN server smoke: PASS, content = '.'`.
- 256-dim server smoke passed:
  `powershell -ExecutionPolicy Bypass -File scripts\kvarn\run_server_smoke.ps1 -Model C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q4_K_M.gguf -BuildDir build-kvarn-cuda-nofa-vs -Port 8135`.
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

1. Enable unsplit KVarN prompt batches now that `GGML_OP_KVARN_ATTN_MIXED`
   carries an optional KQ/causal mask in `src[10]` and the CUDA packed and
   scratch-reference mixed-attention paths consume it. Runtime batch
   preparation still splits to one-token ubatches until prompt storage order,
   active-window construction, and body-record sealing are validated together.
2. Finish prompt-batch sealing semantics. The graph builder now collects all
   seal records in an ubatch and emits store ops for each record, and the body
   plan has multi-record seal coverage. Prompt ubatches are still split to one
   token, so this path is not yet exercised by production prefill batches.
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
  mixed-attention path now has a batched fused score/softmax/AV CUDA kernel,
  but the mixed op is still a custom attention op rather than a
  flash-attention load-path integration. Until that integration exists, KVarN
  must not pretend that llama.cpp Flash Attention is required or used.
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
K/V head dimensions other than 128, 256, or 512, MLA, SWA/ISWA, unsupported
backend placement, attention rotations/KQ bias/sinks, and other explicit
KVarN graph-backend guards.
