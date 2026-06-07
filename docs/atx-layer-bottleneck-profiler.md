# ATX Layer Bottleneck Profiler

This branch adds a low-friction runtime profiler for measuring which transformer
layers are likely bottlenecks for coding and agentic workloads. It is intended
for cross-hardware testing: run the same prompt manifest on Mac, CUDA, or another
backend, then compare per-layer timing, op-family timing, and candidate hot-layer
policies.

The profiler is disabled by default. If no `--layer-profile` flag is supplied,
the runtime follows the normal llama.cpp / Quins path.

## Runtime Flags

```text
--layer-profile FILE
    Append JSONL per-node timing records to FILE.

--layer-profile-detail off|summary|ops
    summary: profile graph nodes associated with blk.N layers.
    ops: also profile non-layer nodes for calibration/debugging.
    off: keep the callback installed but skip profiling.

--layer-profile-sync none|token|layer
    layer: ask the scheduler to synchronize profiled layer nodes so elapsed_us is
           a close attribution measurement.
    none: do not request synchronization; useful only to test callback overhead.
    token: reserved for compatibility with campaign manifests; currently behaves
           like the normal profiled callback path.

--layer-profile-warmup N
    Skip the first N profiled graph nodes in the output.

--layer-profile-max-tokens N
    Stop writing node records after N profiled records. Use -1 for unlimited.
```

## Build

Metal example:

```bash
cmake -S . -B build-atx-metal -DGGML_METAL=ON
cmake --build build-atx-metal --target llama-cli llama-server -j 8
```

CUDA example:

```bash
cmake -S . -B build-atx-cuda -DGGML_CUDA=ON
cmake --build build-atx-cuda --target llama-cli llama-server -j 8
```

## Single Prompt Smoke

```bash
./build-atx-metal/bin/llama-cli \
  -m /path/to/model.gguf \
  -p "Write a Python function that topologically sorts a graph." \
  -n 64 \
  --temp 0 \
  --layer-profile runs/layer-smoke/profile.jsonl \
  --layer-profile-detail summary \
  --layer-profile-sync layer
```

Expected output:

- `profile_start` header row.
- `layer_profile_node` rows with `layer` values such as `0..39`.
- finite `elapsed_us` values.
- `op_family` labels including `attention`, `moe_ffn`, `norm`,
  `residual_dense`, or `other`.

## Campaign Harness

The repo includes a stdlib-only helper:

```bash
python3 scripts/layer_bottleneck_campaign.py inventory \
  --out-dir runs/layer_bottleneck/local

python3 scripts/layer_bottleneck_campaign.py make-mini \
  --out runs/layer_bottleneck/local/request_manifest.jsonl

python3 scripts/layer_bottleneck_campaign.py run \
  --manifest runs/layer_bottleneck/local/request_manifest.jsonl \
  --out-dir runs/layer_bottleneck/local \
  --llama-cli ./build-atx-metal/bin/llama-cli \
  --model /path/to/model.gguf \
  --max-tokens 64 \
  --detail summary \
  --sync layer \
  --resume

python3 scripts/layer_bottleneck_campaign.py analyze \
  --out-dir runs/layer_bottleneck/local
```

The analysis command writes:

- `layer_timing_summary.csv`
- `layer_op_family_summary.csv`
- `request_summary.csv`
- `token_latency.csv`
- `layer_policy_candidates.json`
- `recommended_top10_layers.json`
- `layer_bottleneck_report.html`

## Interpretation Boundary

`--layer-profile-sync layer` is an attribution mode. It intentionally adds
synchronization so layer timing is easier to compare, and should not be treated
as production throughput. Use the generated top-layer policies as candidates,
then validate throughput separately with identical prompts and the intended
offload/residency configuration on target hardware.

This measures whole-layer/block criticality. It is deliberately separate from
expert or layer-expert saliency, because sparse expert residency can fail to
improve token cycle time when a block must wait for all routed expert work to
finish.
