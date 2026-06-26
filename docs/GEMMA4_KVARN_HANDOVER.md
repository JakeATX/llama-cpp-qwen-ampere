# Gemma4 KVarN Handover

Audience: GPT-5.5 Pro, Claude 4.6 Sonnet, or another senior reviewer/debugger.

Repo path:

```text
C:\Users\sjake\OneDrive\Documents\New project\llama.cpp-kvarn-cuda
```

Current branch and commit at handover:

```text
branch: kvarn-atx-integration
commit: c0d5c9ceb
```

Model used for the current Gemma diagnosis:

```text
C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf
```

## Executive Summary

Update after the 16k fixture rerun: the conservative hybrid route below should
be treated as historical rollback evidence, not the production default. The old
small sanity fixture made donor layers `11` and `47` look correctness-sensitive,
but a Gemma-rendered 16k chat fixture now passes with all full-attention donor
layers KVarN-routed. The code default is all-full routing again; set
`LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE=1` only to reproduce the old hybrid route.

Historical conservative route:

```text
KVarN-routed physical full-attention donor layers: 5,17,23,29,35,41
Normal full-KV fallback donor layers:             11,47
```

Do not present the synthetic 16k fixture as final broad Gemma quality evidence:
the negative PPL deltas imply a noisy/low-entropy gate. It is strong enough to
show that the old bypass of donor layers `11` and `47` is not justified as the
default, and that production speed work must measure the all-full route.

Your job is to find the actual mechanism and a better path:

1. Explain why donor layers `11` and `47` fail when KVarN-routed.
2. Fix the bug if it is an implementation issue.
3. If it is intrinsic quantization sensitivity, quantify it honestly and find a
   better mixed-precision route than simply disabling the sensitive layers.
4. Validate at 16k context on a Gemma-compatible corpus before any speed claims.

Do not run heavyweight model/perplexity tests concurrently. Run one GPU/model
job at a time. Previous concurrent runs artificially increased VRAM pressure and
invalidated the result.

## What Was Fixed Already

Harness/test setup fixes:

- `tools/perplexity/perplexity.cpp`
  - `params.parse_special` is now honored in both normal and strided perplexity
    tokenization paths.
  - Strided perplexity BOS placement was fixed so BOS is put into the batch
    before `llama_decode`, not after.
- `common/arg.cpp`
  - `--parse-special` is exposed for `LLAMA_EXAMPLE_PERPLEXITY`.
- `scripts/kvarn/check_perplexity_fixture.py`
  - Adds fixture checks for literal `<unk>`, chat-template marker mismatch, and
    actual tokenizer unknown-token IDs via `llama-tokenize`.
- `scripts/kvarn/run_accuracy_gate.ps1`
  - Runs fixture preflight before PPL.
  - Refuses KVarN comparison when the normal-KV baseline PPL is unhealthy.
  - Uses a native process capture helper instead of noisy PowerShell `2>&1`.
  - Normalizes `-OutputDir` to an absolute path for Windows/.NET file writes.
- `docs/KVARN_ACCURACY_GATE.md`
  - Updated to stop presenting the old Hadamard theory as the primary
    explanation for the Gemma baseline PPL issue.

Gemma4 code hardening already in the tree:

- `src/models/gemma4.cpp`
  - Gemma4 final logit softcapping defaults to `0.0f` before optional metadata
    load, avoiding inheritance of the generic `30.0f` default.
  - Requires `v_proj` for physical SWA KV layers.
  - Asserts against SWA `Vcur = Kcur` fallback.
- `src/llama-hparams.cpp`, `src/llama-hparams.h`
  - Adds `kv_reuse_layer_matching_attention_type()`.
- `src/llama-model.cpp`
  - Adds forced Gemma4+ISWA route mapping.
  - Current default routes all full-attention donors through KVarN.
  - `LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE=1` restores the old hybrid route for
    reproduction.

## Key Evidence So Far

The old high Gemma baseline PPL was not trustworthy. It came from bad fixtures
and corpus/model pairings: literal `<unk>`, Qwen-style markers, and text that is
not a meaningful Gemma health check.

A deliberately low-entropy Gemma sanity fixture proves that the model,
tokenizer, file-input path, and PPL scorer can produce sane absolute PPL:

```text
fixture: scripts\kvarn\fixtures\gemma4_predictable_sanity.txt
normal KV, Q4 GPU, ctx=512: PPL = 1.1071 +/- 0.07188
normal KV, Q4 CPU, ctx=512: PPL = 1.0306 +/- 0.02285
```

True forced Gemma4+ISWA KVarN then showed a separate correctness problem:

```text
all full-attention donors KVarN-routed, k8v8:
PPL = 484746.4909 +/- 256067.71554

conservative default route, k4v2:
PPL = 1.0724 +/- 0.03914

conservative default route, k8v8:
PPL = 1.0461 +/- 0.02530
```

Single-layer bisection on the same sanity fixture:

```text
layer 5:  PPL = 1.0545
layer 11: PPL = 2.9372
layer 17: PPL = 1.0474
layer 23: PPL = 1.037
layer 29: PPL = 1.0977
layer 35: PPL = 1.109
layer 41: PPL = 1.0755
layer 47: PPL = 11.162
```

Cumulative/set bisection:

```text
5,11:                    PPL = 29.0995
5,11,17:                 PPL = 74.3523
5,11,17,23,29,35,41:     PPL = 51.58
29,35,41,47:             PPL = 8.1821
all full donors:         PPL = 484746.4909
5,17,23,29,35,41 only:   PPL = 1.0461 to 1.0724 depending on preset
17,23,29,35,41 only:     PPL = 1.0459 with k4v2
```

Interpretation:

- The bad Gemma baseline was a testing-harness/corpus problem.
- The KVarN failure is real but route-specific.
- Donors `11` and `47` are behaviorally implicated.
- The current workaround is not a root-cause fix.

## Important Artifacts

Relevant logs are under:

```text
artifacts\gemma-fixture-diagnosis\
```

Highest-value logs:

```text
predictable-file-q4-c512-gpu.log.txt
predictable-file-q4-c512-gpu-kvarn-k4v2-default-route.log.txt
predictable-file-q4-c512-gpu-kvarn-k8v8-default-route.log.txt
predictable-file-q4-c512-gpu-kvarn-k8v8-all-full-override.log.txt
predictable-file-q4-c512-gpu-kvarn-layer-11.log.txt
predictable-file-q4-c512-gpu-kvarn-layer-47.log.txt
accuracy-gate-predictable-k4v2-default-route\summary.md
accuracy-gate-predictable-k4v2-default-route\fixture-preflight.log.txt
```

The current passing gate summary:

```text
artifacts\gemma-fixture-diagnosis\accuracy-gate-predictable-k4v2-default-route\summary.md
```

reports:

```text
PPL f16 = 1.1071
PPL KVarN = 1.0724
increase = -3.13%
```

## Reproduction Commands

Run from repo root:

```powershell
Set-Location "C:\Users\sjake\OneDrive\Documents\New project\llama.cpp-kvarn-cuda"
```

Build:

```powershell
cmake --build .\build-kvarn-cuda-static-vs --config Release --target llama-perplexity test-kvarn-kv -j 8
```

Unit test:

```powershell
.\build-kvarn-cuda-static-vs\bin\Release\test-kvarn-kv.exe
```

Fixture preflight:

```powershell
python .\scripts\kvarn\check_perplexity_fixture.py `
  --dataset .\scripts\kvarn\fixtures\gemma4_predictable_sanity.txt `
  --model "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf" `
  --tokenizer-exe .\build-kvarn-cuda-static-vs\bin\Release\llama-tokenize.exe `
  --max-unk-rate 0.001 `
  --max-token-unk-rate 0.001 `
  --fail-on-template-mismatch
```

Expected:

```text
literal_unk=0
tokenizer_unk=0
tokenizer_unk_rate=0.000000
```

Historical conservative KVarN gate:

```powershell
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = "1"
$env:LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE = "1"
Remove-Item Env:LLAMA_KVARN_LAYER_FILTER -ErrorAction SilentlyContinue

scripts\kvarn\run_accuracy_gate.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf" `
  -Dataset .\scripts\kvarn\fixtures\gemma4_predictable_sanity.txt `
  -BuildDir .\build-kvarn-cuda-static-vs `
  -OutputDir .\artifacts\gemma-fixture-diagnosis\accuracy-gate-predictable-k4v2-default-route `
  -ContextSize 512 `
  -BatchSize 128 `
  -Chunks 1 `
  -KvarnPreset kvarn_k4v2_g128 `
  -MaxPplIncrease 0.05 `
  -MaxBaselinePpl 100 `
  -ExpectedKvarnLayers "5,17,23,29,35,41"

Remove-Item Env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE -ErrorAction SilentlyContinue
```

Expected:

```text
PASS
PPL f16 = 1.1071
PPL KVarN = 1.0724
```

Reproduce the old all-full short-fixture failure:

```powershell
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = "1"
Remove-Item Env:LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_KVARN_LAYER_FILTER -ErrorAction SilentlyContinue

.\build-kvarn-cuda-static-vs\bin\Release\llama-perplexity.exe `
  -m "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf" `
  -f .\scripts\kvarn\fixtures\gemma4_predictable_sanity.txt `
  -ngl 999 -np 1 -b 128 -fit off -fa off -c 512 --chunks 1 `
  --kv-cache-quant kvarn --kvarn-preset kvarn_k8v8_g128 --kvarn-iters 4 --kvarn-rtn-quantile 1.0

Remove-Item Env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA -ErrorAction SilentlyContinue
```

Expected:

```text
PPL = 484746.4909 +/- 256067.71554
```

Run a diagnostic layer route:

```powershell
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = "1"
$env:LLAMA_KVARN_LAYER_FILTER = "11"

.\build-kvarn-cuda-static-vs\bin\Release\llama-perplexity.exe `
  -m "C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf" `
  -f .\scripts\kvarn\fixtures\gemma4_predictable_sanity.txt `
  -ngl 999 -np 1 -b 128 -fit off -fa off -c 512 --chunks 1 `
  --kv-cache-quant kvarn --kvarn-preset kvarn_k8v8_g128 --kvarn-iters 4 --kvarn-rtn-quantile 1.0

Remove-Item Env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_KVARN_LAYER_FILTER -ErrorAction SilentlyContinue
```

`LLAMA_KVARN_LAYER_FILTER` accepts comma-separated IDs and ranges with optional
steps, for example:

```text
11
47
5,17,23,29,35,41
5-47:6
```

## Code Areas To Review First

Route selection and hybrid fallback:

```text
src/llama-model.cpp
  llama_kvarn_layer_filter_from_env()
  Gemma4 KVarN+ISWA branch around create_memory()
  install_gemma_route()
  LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE
```

ISWA hybrid cache:

```text
src/llama-kv-cache-kvarn-iswa.cpp
src/llama-kv-cache-kvarn.cpp
src/llama-kv-cache.cpp
```

Gemma4 model/hparams:

```text
src/models/gemma4.cpp
src/llama-hparams.cpp
src/llama-hparams.h
```

KVarN CUDA kernels and store/read paths:

```text
ggml/src/ggml-cuda/kvarn.cu
src/llama-graph.cpp
```

Perplexity harness:

```text
tools/perplexity/perplexity.cpp
common/arg.cpp
scripts/kvarn/run_accuracy_gate.ps1
scripts/kvarn/check_perplexity_fixture.py
```

## Hypotheses To Test

Do not accept any one of these without falsification. The current evidence only
proves layer sensitivity behaviorally.

1. KV reuse / physical donor mapping bug
   - Are logical layers that reuse physical donor `11` or `47` being routed
     incorrectly?
   - Does KVarN route expansion match normal full-KV fallback expansion exactly?
   - Are SWA and full-attention reuse groups accidentally crossing?

2. Final-layer or near-final-layer sensitivity
   - Donor `47` may be qualitatively more sensitive because it is final or
     close to output.
   - Test higher precision on only `47`: k8v8, k8v16 if available, V-only
     higher precision, K-only higher precision.

3. Quantization distribution issue
   - Measure value/key quantization error for Gemma donors `5,11,17,23,29,35,41,47`.
   - Compare real Gemma donor `11` and `47` distributions against donor `5` and
     against Qwen layers that pass.
   - Do not infer from width alone. Earlier critique correctly noted that
     bit-width dominated width in the toy error analysis.

4. ISWA/full-cache mixing issue
   - The conservative route uses a compatibility normal full-KV cache for
     fallback layers.
   - Verify that fallback normal full-KV and KVarN layers are indexed, shifted,
     and read at the same positions under ISWA.
   - Check sink/tail/body boundaries for mismatches.

5. Hadamard/rotation mismatch
   - Earlier code review flagged that KVarN body KV rotation and sink/tail raw
     KV may be mixed in one attention op.
   - This is no longer the primary explanation for the bad Gemma baseline, but
     it remains a plausible correctness bug for KVarN.
   - Test by disabling rotation, rotating Q consistently, or isolating body vs
     sink/tail contributions.

6. Kernel path difference
   - Confirm whether donor `11` and `47` take the same CUDA path as donors that
     pass.
   - Compare CPU vs CUDA KVarN for one bad donor if CPU KVarN path is available.
   - Dump per-token logits/losses for baseline vs bad donor route to see whether
     failure starts at the first prediction, after a boundary, or after reuse.

7. Fixture overfit
   - The low-entropy sanity fixture is only a harness smoke test.
   - A fix must pass a validated Gemma-compatible heldout corpus and the planned
     16k context run.

## Suggested Debugging Plan

1. Establish a clean local baseline.
   - Run `test-kvarn-kv.exe`.
   - Run fixture preflight.
   - Run the conservative gate and all-full reproduction.

2. Reproduce single-layer sensitivity.
   - Run `LLAMA_KVARN_LAYER_FILTER=11`.
   - Run `LLAMA_KVARN_LAYER_FILTER=47`.
   - Run one known-good donor, e.g. `5` or `17`.

3. Instrument the route.
   - Log physical donor, logical layer expansion, attention type, reuse source,
     KV head count, head dim, sink/tail/body counts, and selected kernel path
     per KVarN layer.

4. Add correctness probes before changing algorithms.
   - Compare normal-KV vs KVarN logits per token.
   - Dump first token index where KL/loss diverges.
   - Dump per-layer attention output deltas if possible.
   - Compare reconstructed K/V for donor `11`, `47`, and a known-good donor.

5. Test precision alternatives on the bad donors.
   - All-full route with donor `47` at higher V precision only.
   - All-full route with donor `11` at higher precision only.
   - Mixed route where only sensitive donors use 8-bit or f16 fallback, not full
     normal KV, if KVarN supports that.

6. Test long context only after small-fixture causality is understood.
   - The user now only wants up to 16k context.
   - Use one model process at a time.
   - Use validated Gemma-compatible corpus only.
   - Do not present speed numbers until correctness is validated.

## What Would Count As A Real Fix

A real Gemma4 KVarN fix should meet at least one of these:

1. KVarN can route donors `11` and `47` without PPL/KL explosion on the small
   sanity fixture and on a validated real Gemma corpus.
2. If those donors are intrinsically sensitive, the implementation provides a
   principled mixed-precision policy for them, with measured memory/speed impact
   and 16k correctness.
3. The route-selection logic is model-metadata-driven or error-driven, not a
   hardcoded ordinal heuristic like "skip second and final full donor."

The current state does not meet those standards. It is a diagnostic workaround
that keeps the test harness usable.

## Validation Requirements Before Speed Work

Minimum correctness validation:

```text
1. Fixture preflight passes: literal_unk=0 and tokenizer_unk_rate acceptable.
2. Normal-KV baseline PPL is sane for the corpus/model pairing.
3. KVarN vs normal-KV PPL delta or KL passes at ctx=512.
4. Same route passes at ctx=16k.
5. Expected KVarN layers are verified from logs.
6. No KVarN fallback is silently accepted unless explicitly intended.
```

Only after that should speed be measured.

Speed target from the user:

```text
KVarN 8-bit and 2-bit versions should run within 95% of mainline llama.cpp
8-bit KV-cache quantization speed, measured against non-KVarN mainline.
```

That means speed comparisons must be against a non-KVarN implementation, not
KVarN vs KVarN internal variants.

## Known Environmental Issues

Windows Device Guard/Application Control blocked freshly rebuilt unsigned
executables during this work. The current binaries ran after signing/approval,
but if an executable suddenly exits with Windows code `3221225786` or says it
was blocked by policy, check Device Guard before assuming a model crash.

GPU was reset successfully with admin `pnputil /restart-device` after it entered
`CM_PROB_WILL_BE_REMOVED`. Current `nvidia-smi` sees:

```text
NVIDIA GeForce RTX 5070
Driver 595.95
CUDA 13.2
VRAM about 12 GB
```

## Non-Negotiable Test Discipline

- Do not run two correctness/perplexity jobs concurrently.
- Do not use the old Qwen-style multiturn fixture for Gemma health.
- Do not use literal-`<unk>` WikiText artifacts as Gemma health baselines.
- Do not report a KVarN pass unless logs show `llama_kv_cache_kvarn:` engaged.
- Do not present the conservative route as full Gemma4 KVarN support.
- Do not move to speed claims until Gemma correctness passes at 16k on a
  validated corpus.
