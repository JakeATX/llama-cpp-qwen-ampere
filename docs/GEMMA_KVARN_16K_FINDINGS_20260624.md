# Gemma 4 12B KVarN 16k Findings - 2026-06-24

## Context

Model:
`C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf`

Fixture:
`artifacts\gemma-fixture-diagnosis\gemma4_chat_sanity_long_700turns.txt`

Perplexity binary:
`build-kvarn-cuda-static-vs\bin\Release\llama-ppl-local.exe`

Current rebuilt SHA256:
`885E92A8F7E78488DB7F936BF8D04C6E89398E0879F9E060FC07E3C4787FADAD`

Baseline is now sane: f16/non-KVarN KL baseline PPL is about `3.7114`.
The earlier very high Gemma PPL was a harness/setup problem, not model quality.

## 16k Results

| Run | Artifact | Result |
| --- | --- | --- |
| K8/V4 active body | `artifacts\kvarn-accuracy\strict-gemma16k-k8v4-activebody-currentppl-20260624` | mean KL `0.181697`, p99 `4.606222`, PPL ratio `1.002007` |
| K8/V8 active body | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-activebody-currentppl-20260624` | mean KL `0.207214`, p99 `4.905724`, PPL ratio `1.009930` |
| K8/V2 active body | `artifacts\kvarn-accuracy\strict-gemma16k-k8v2-activebody-currentppl-20260624` | mean KL `0.519581`, p99 `14.965174`, PPL ratio `0.830038` |
| K8/V8 paper frame | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-paperframe-currentppl-20260624` | mean KL `18.641083`, PPL ratio `26,398,364.627409` |
| K8/V8 compact causal mask | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-compactmask-currentppl-20260624` | mean KL `0.207214`, unchanged from K8/V8 |
| K8/V8 GQA scalar disabled | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-disable-gqa-scalar-currentppl-20260624` | mean KL `0.207214`, unchanged from K8/V8 |
| K8/V8 layers 5,11,17,23 | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-layers-5-23-currentppl-20260624` | mean KL `0.205287`, PPL ratio `0.994866` |
| K8/V8 layers 29,35,41,47 | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-layers-29-47-currentppl-20260624` | mean KL `0.158321`, PPL ratio `1.025197` |
| K8/V8 layer 5 only | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-layer-5-currentppl-20260624` | mean KL `0.191388`, PPL ratio `0.990964` |
| K8/V8 layer 47 only | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-layer-47-currentppl-20260624` | mean KL `0.164316`, PPL ratio `1.027500` |
| K8/V8 layer 5 raw-body split | `artifacts\kvarn-accuracy\strict-gemma16k-k8v8-layer-5-rawbody-split-currentppl-20260624` | mean KL `0.181667`, p99 `4.853204`, PPL ratio `1.030788` |
| K8/V8 layer 5 raw-body scalar fallback smoke | `artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-rawbody-scalar-fallback-smoke-20260624` | completed with scalar-QT unavailable warning and raw-body split fallback; PPL `6.8481` |
| K8/V2 layer 5 only, ctx4096 | `artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-k8v2-kl-20260625` | mean KL `0.358022`, p99 `9.448116`, p99.9 `23.440683`, max `29.499249`, PPL ratio `0.933458`, RMS dp `14.991%`, same-top `92.428%` |
| K8/V2 layer 5 only, ctx16384 | `artifacts\kvarn-accuracy\gemma4-ctx16384-layer5-k8v2-kl-20260625` | mean KL `0.219788`, p99 `5.174049`, p99.9 `23.996983`, max `26.839872`, PPL ratio `1.013736`, RMS dp `12.667%`, same-top `95.312%` |

## Ruled Out

- The bad Gemma result is not a broken baseline anymore. The baseline PPL is low and stable.
- The failure is not primarily V-bit width. K8/V8 fails similarly to K8/V4; K8/V2 is worse but not the root cause. The 2026-06-25 K8/V2 layer-5 runs fail at both ctx4096 and ctx16384, but they are still running through the same KVarN mixed-body path that already fails with K8/V8 and raw-body mirrors.
- The active-body transferred F32 mask is not the causal mechanism. Forcing synthetic compact causal masking changed CUDA traces to `mask_type=3` and left the metrics unchanged.
- The GQA scalar-QT optimized kernel is not the causal mechanism. Disabling it changed the inner trace from `scalar-qt-gqa` to `scalar-qt` and left the metrics unchanged.
- The issue is not isolated to one layer group. A single compressed layer 5 or layer 47 is already far outside the KL gate.
- The issue is not primarily packed K8/V8 quantization. A raw-body split run for layer 5, using captured pre-quant K/V body mirrors, still produced mean KL `0.181667`, essentially the same failure as packed K8/V8 layer 5.
- The raw-body scalar-QT diagnostic no longer aborts when the Gemma 512d batch shape cannot use that kernel; it falls back to the raw-body split kernels and logs the fallback.
- `kvarn_k8v2_g128` is now covered by the default oracle wrapper and serialized safe full-gate preset lists, and `test-arg-parser` asserts that the common parser maps it to K=8, V=2, G=128.
- 2026-06-25 local verification confirms K8/V2 is implemented, not just documented: `test-arg-parser.exe`, `test-kvarn-kv.exe`, materialize-only CUDA scratch-ref, full CUDA scratch-ref, and `run_vllm_oracle_selftest.ps1` all pass with K8/V2 included.
- The ctx16k K8/V2 PPL ratio is misleading. Baseline PPL is healthy at `3.7114` and K8/V2 reports lower PPL `3.0806`, but KL/same-top fail badly: mean KL `0.519581`, p99 `14.965174`, p99.9 `27.308834`, max `33.894272`, same-top `92.724%`. Treat K8/V2 as incorrect for Gemma until KL passes.
- CLI/help strings now name K8/V2 as a supported KVarN preset example. The common parser and `llama-bench` parser now reject malformed preset fields such as trailing junk and non-`g128` groups.

## Boundary Diagnostics

2026-06-25 raw-body captures targeted the first high-KL region instead of only the final 4096-token boundary:

`artifacts\kvarn-rootcause\gemma4-layer5-k8v8-rawbody-boundaries-from2048-20260625`

This captured eight layer-5 raw-body boundaries from 2048 through 2944 tokens with `-DebugRawBody`, `-DisablePaperFrame`, `-ForceExperimentalIswa`, and layer filter `5`. Body-store replay matched the captured KVarN mixed-attention output at full-Q/O worst NMSE between about `4.94e-12` and `9.01e-12`.

The matching packed K8/V8 capture is:

`artifacts\kvarn-rootcause\gemma4-layer5-k8v8-packed-boundaries-from2048-20260625`

It covered the same eight boundaries. Packed K8/V8 replay matched the captured CUDA output at full-Q/O worst NMSE around `5.56e-12` to `1.25e-11`, while packed output versus raw body-store truth showed full-Q/O worst NMSE around `2.47e-5` to `3.81e-5`.

The capture-plan report:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vraw-kl-20260624\kl-boundaries-capture-plan.md`

confirms the worst KL rows were inside this capture span. Rank 1 (`logit_pos=2366`, `target_pos=2367`, KLD `15.669326`) maps to boundary `call_000003`, `iq=62`; rank 4 maps to `call_000006`, `iq=101`; rank 5 maps to `call_000004`, `iq=72`.

The row-level join report:

`artifacts\kvarn-rootcause\gemma4-layer5-k8v8-kl-boundary-replay-join-20260625.md`

shows the exact rank-1 row has raw-body local replay error `3.24e-12` NMSE and packed K8/V8 local error `9.75e-6` NMSE (`0.0248` max abs in the attention output). Other captured top rows show packed local errors in the `~6e-6` to `3.45e-5` NMSE range.

Tensor-dump source checks for the same rank-1 boundary:

- `artifacts\kvarn-rootcause\gemma4-layer5-k8v8-call3-tensordump-20260625`
- `artifacts\kvarn-rootcause\gemma4-layer5-k8v8-rawbody-call3-tensordump-20260625`

show both packed and raw-body paths consume the same K/V source values as the graph tensors for `call_000003`. Active sink/tail K/V matches `Kcur_pos-5` / `Vcur_normed-5` at worst NMSE `4.35e-8`; body records `0..16` match at worst NMSE `4.47e-8`.

That result means the sampled raw-body mixed-attention operation is locally exact against its own f16/f32 body-store truth for those boundaries, and the K/V source values for the rank-1 call match the native graph tensors. Packed K8/V8 local attention error is small but nonzero. It does not make the KL gate pass, and it does not validate K8/V2. It narrows the root cause away from simple K/V source order, active tail order, body record order, or CUDA replay mismatch for `call_000003`.

Hidden-state row comparison then found that the exact rank-1 row already differs before layer-5 attention:

`artifacts\kvarn-rootcause\gemma4-hidden-row2366-baseline-vs-rawbody.csv`

At `logit_pos=2366`, baseline `-b512` vs raw-body KVarN has:

- `attn_norm-5`: NMSE `4.759e-4`, max abs `0.3491`
- `attn_out-5`: NMSE `1.375e-4`, max abs `0.1737`
- `ffn_residual_out-5`: NMSE `9.640e-4`, max abs `0.9299`
- `l_out-5`: NMSE `9.640e-4`, max abs `0.3306`

A non-KVarN `-b128` baseline also differs from non-KVarN `-b512` (`PPL 6.7823` vs `7.1041`), so batch layout is a real Gemma harness confound. However, a matched `-b128` raw-body KL gate still fails:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vraw-kl-b128-20260625`

Mean KL `0.231444`, p99 `4.533959`, p99.9 `19.481840`, max `21.354631`, PPL ratio `0.991726`.

A matched `-b128` f32-cache baseline also fails:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vraw-kl-b128-f32baseline-20260625`

Mean KL `0.238554`, p99 `5.284986`, p99.9 `18.177629`, max `20.515261`, PPL ratio `1.013883`.

This means batch-size mismatch and f16-vs-f32 baseline KV precision are not the whole explanation.

Matched `-b128` layer bisect then localized the divergence:

`artifacts\kvarn-rootcause\gemma4-hidden-bisect-row3353-b128.csv`

For the top matched-batch KL row (`logit_pos=3353`), baseline and raw-body KVarN are bit-identical through layers 0..4 and at `attn_norm-5`. The first observed divergence is `l_out-5`, with NMSE `6.10e-5`.

The layer-5 detail dump:

`artifacts\kvarn-rootcause\gemma4-hidden-layer5-detail-row3353-b128.csv`

shows the divergence begins after attention:

- `attn_norm-5`: exact
- `attn_out-5`: NMSE `2.640e-6`, max abs `0.00982`
- `ffn_residual_out-5`: NMSE `6.101e-5`, max abs `0.0795`
- `l_out-5`: NMSE `6.101e-5`, max abs `0.0283`

The exact raw-body boundary for that row:

`artifacts\kvarn-rootcause\gemma4-layer5-k8v8-rawbody-b128-call11-20260625`

maps the top KL row to `call_000011`, `iq=25`; raw-body pre-WO mixed attention replay for `iq=25` is exact against body-store truth at worst NMSE `1.55e-12`.

Follow-up tensor-dump checks showed one important diagnostic trap: Gemma layer 5 is on the KVarN+ISWA path, so internal tensors are named `kvarn_iswa_kqv_out_2d-5` and `kvarn_iswa_kqv_wo-5`, not plain `kvarn_kqv_*`. Those internal ISWA tensors also start after two 128-token fallback chunks, while `attn_post_norm-5` is present from the start. A same-position native-vs-ISWA comparison therefore compares different token identities and can falsely report huge pre-WO error.

The corrected row comparison for `logit_pos=3353` is:

- `compare_boundary_attention_row_to_tensor.py` against native `kqv_out-5`: boundary truth vs KVarN `full_out.bin` NMSE `8.52e-13`; boundary truth vs native `kqv_out-5` NMSE `6.86e-6`.
- `compare_boundary_attention_row_to_tensor.py` against KVarN `kvarn_iswa_kqv_out_2d-5` with internal timeline `pos=3097`: boundary truth vs KVarN internal dump NMSE `8.52e-13`, and `full_out.bin` vs KVarN internal dump is exact.
- `compare_tensor_timeline_rows.py --left-pos 3353 --right-pos 3097`: native `kqv_out-5` vs KVarN `kvarn_iswa_kqv_out_2d-5` NMSE `6.87e-6`; native `kqv_wo-5` vs KVarN `kvarn_iswa_kqv_wo-5` NMSE `2.57e-6`.

A decisive control run then forced the KVarN+ISWA routed layer back onto the normal full-attention path:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-normal-iswa-fallback-kl-b128-20260625`

with `LLAMA_KVARN_ISWA_DEBUG_FULL_NORMAL_ATTN=1`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`, and layer filter `5`. It passed: mean KL `0.000000`, p99 `0.000132`, p99.9 `0.000262`, max `0.000327`, PPL ratio `1.000000`, same-top `100.000%`.

2026-06-25 added an env-gated diagnostic path:

`LLAMA_KVARN_ISWA_DEBUG_MATERIALIZE_MHA=1`

This path materializes the active compact KVarN window into regular f16 K/V tensors and routes Gemma KVarN+ISWA full-attention layers through `build_attn_mha()` instead of the custom `GGML_OP_KVARN_ATTN_MIXED` arithmetic. A synthetic CUDA smoke now covers `sink -> body records -> pending -> wrapped tail` materialization for K8/V8 and K8/V2 at `head_dim=512`, `group=128`.

Packed K8/V8 materialized-MHA still fails:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-packed-k8v8-materialized-mha-trace-kl-b128-20260625`

Mean KL `0.222732`, p99 `4.032655`, p99.9 `17.447098`, max `20.498528`, PPL ratio `1.003816`. The trace proves the diagnostic branch executed on active-body windows from `n_records=1` through `30`, with exact compact mask widths from `384` through `4096`.

Raw-body mirror materialized-MHA passes:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vraw-materialized-rawmirror-mha-kl-b128-20260625`

Mean KL `0.000000`, p99 `0.000132`, p99.9 `0.000262`, max `0.000327`, PPL ratio `1.000000`, same-top `100.000%`.

Packed K8/V2 materialized-MHA fails worse:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-packed-k8v2-materialized-mha-kl-b128-20260625`

Mean KL `0.277909`, p99 `5.253958`, p99.9 `18.387020`, max `19.897341`, PPL ratio `0.962124`, same-top `92.623%`.

K/V split materialized-MHA controls on layer 5 show both sides are unsafe in this layer:

- Raw K + packed V8: `artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vpacked-materialized-mha-kl-b128-20260625`, mean KL `0.189423`, p99 `3.421370`, p99.9 `17.947855`, max `21.577076`, PPL ratio `1.045169`, same-top `93.161%`.
- Packed K8 + raw V: `artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kpacked-vraw-materialized-mha-kl-b128-20260625`, mean KL `0.260023`, p99 `5.070845`, p99.9 `20.182333`, max `26.586639`, PPL ratio `0.962131`, same-top `93.112%`.

Boundary quant-error replay on calls `000000`, `000003`, and `000007` showed current packed dequant agrees with the vLLM-style oracle and with the captured packed CUDA boundary output, and the direct attention-output NMSE versus raw-body replay was only about `2.5e-5` to `3.8e-5`. That does not rescue correctness: the small layer-output perturbation is still amplified into large downstream logit KL. The conclusion is model-level fidelity, not byte-layout mismatch at the sampled boundary.

2026-06-25 follow-up found and fixed a real `GGML_OP_KVARN_MATERIALIZE_KV` dispatch stride bug: the ggml op must pass output `head_stride = dst->nb[1]` and `token_stride = dst->nb[2]` for the normal `[head_dim, n_head_kv, n_kv]` MHA cache layout. This fix is covered by the materialize smoke, but it is not causal for the Gemma layer-5 failure because Gemma full-attention has `n_head_kv=1`, making the bad and good strides equivalent for this case. Re-running layer 5 K8/V8 materialized-MHA after the fix reproduced the same failure:

`artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-packed-k8v8-materialized-mha-stridefix-kl-b128-20260625`

Mean KL `0.222732`, p99 `4.032655`, p99.9 `17.447098`, max `20.498528`, PPL ratio `1.003816`, same-top `92.916%`.

The same controlled materialized-MHA test on donor layer 11 also fails, so this is not a single donor-5 artifact:

- K8/V8 layer 11: `artifacts\kvarn-accuracy\gemma4-ctx4096-layer11-packed-k8v8-materialized-mha-kl-b128-20260625`, mean KL `0.188668`, p99 `3.781413`, p99.9 `17.165918`, max `19.750319`, PPL ratio `1.035874`, same-top `93.942%`.
- K8/V2 layer 11: `artifacts\kvarn-accuracy\gemma4-ctx4096-layer11-packed-k8v2-materialized-mha-kl-b128-20260625`, mean KL `0.298972`, p99 `6.241906`, p99.9 `17.301250`, max `19.627150`, PPL ratio `1.035405`, same-top `92.672%`.

TurboQuant-style V-only oracle follow-up:

`scripts\kvarn\analyze_boundary_quant_error.py` now includes a V-only TurboQuant oracle over raw KVarN body records. It keeps K and attention probabilities exact, rotates all V into TurboQuant's WHT domain, quantizes only body V with Turbo2/Turbo3/Turbo4 centroids plus corrected norm, computes `probs @ V_rot`, and inverse-WHTs the attention output. This matches TurboQuant's rotated-domain V contract and avoids the invalid shortcut of inverse-rotating each V row before attention.

On Gemma layer 5 boundaries `call_000000`, `call_000003`, and `call_000007`, using the K8/V2 preset for the KVarN V2 oracle:

| Boundary | KVarN V2 raw-K mean/max out NMSE | Turbo2 raw-K mean/max out NMSE | Turbo3 raw-K mean/max out NMSE | Turbo4 raw-K mean/max out NMSE |
| --- | ---: | ---: | ---: | ---: |
| `call_000000` | `0.088364` / `0.423430` | `0.057053` / `0.196196` | `0.011807` / `0.040733` | `0.003226` / `0.023074` |
| `call_000003` | `0.092600` / `0.477372` | `0.063276` / `0.234769` | `0.013259` / `0.049212` | `0.003344` / `0.016223` |
| `call_000007` | `0.097269` / `0.419193` | `0.064441` / `0.244952` | `0.013526` / `0.051705` | `0.003409` / `0.019798` |

The corresponding body-V original-domain NMSE was stable across these calls: KVarN V2 about `0.375`, Turbo2 about `0.117`, Turbo3 about `0.0335`, and Turbo4 about `0.0173`.

Interpretation: Turbo2 is materially better than current KVarN V2 but still too lossy to call functionally lossless on Gemma. Turbo3 is the first plausible compression/quality compromise, and Turbo4 is the safer first production-quality prototype target. Neither result rescues current packed K8/V8 correctness; this is evidence for a different V codec path, not evidence that the current KVarN body format is fine.

2026-06-25 graph-level Turbo frame follow-up:

The materialized-MHA Turbo frame diagnostic is explicitly frame-only, not compression. It rotates materialized V, runs native MHA, and inverse-rotates the attention output. Active body tests require raw K/V mirrors, so these results do not validate packed KVarN compression.

- Identity frame control passed: `artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vraw-materialized-turbovframe-identity-f16-ppllocal-kl-b128-20260625`, mean KL `0.000000`, p99 `0.000132`, max `0.000327`, same-top `100.000%`.
- Signed Turbo frame via sign/Hadamard failed: `artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vraw-materialized-turbovframe-signhadamard-ppllocal-kl-b128-20260625`, mean KL `0.232151`, p99 `4.683884`, max `27.405916`.
- Dense Turbo matrix orientation was fixed and dense keep-f32 now matches the Python Turbo oracle on dumped V (`rot_vs_oracle_nmse ~= 1.14e-08`), but the model KL still fails: `artifacts\kvarn-accuracy\gemma4-ctx4096-layer5-kraw-vraw-materialized-turbovframe-dense-keepf32-ppllocal-kl-b128-20260625`, mean KL `0.213089`, p99 `3.874340`, max `22.481022`, same-top `93.503%`.
- Tensor dumps show the sign/Hadamard graph route is not the Python Turbo transform on this backend shape: inverting dumped `v_turbo_rot` with the Python oracle gives roundtrip NMSE about `2.01`.
- The offline no-quant Turbo f16 oracle on `gemma4-layer5-k8v8-rawbody-boundary-fulltensor-20260625` is clean: `raw_k_turbo_f16_v_no_quant` mean out NMSE `2.03e-09`, body original-domain NMSE `4.32e-08`.

This rules out "Turbo WHT itself is numerically impossible for Gemma V" and rules out "f16 rotated V alone explains the KL." The problem is the current unfused graph-level frame wrapper. Any Turbo-style production implementation must be backend-native/fused and validated by tensor oracles before KL/speed claims.

The current best causal read is therefore narrower and stronger: there is no evidence of a broken Gemma baseline, fixture, layer routing, surrounding model graph, compact token order, compact mask width, or native MHA path. There is also no evidence that the custom mixed-attention CUDA arithmetic is the primary failure, because bypassing it with materialized native MHA does not rescue packed K8/V8 or K8/V2. The failure is now isolated to the fidelity of the current packed KVarN body representation for Gemma full-attention layers. Exact raw K/V body content is functionally lossless through the same compact window and native MHA path; packed K8/V8 is not.

## Current Causal Read

The current Gemma KVarN routed path is not functionally lossless for Gemma 4 12B full-attention layers at 16k. The strongest current evidence is the materialized-MHA split: exact raw body K/V passes the layer-5 KL gate, while packed K8/V8 and packed K8/V2 fail on the same compact window and same native MHA route. Raw-K/packed-V8 and packed-K/raw-V controls both fail for layer 5, and donor 11 repeats the K8/V8 and K8/V2 failures. This makes packed body fidelity the load-bearing correctness problem for Gemma, not the test harness and not the custom attention arithmetic alone.

The boundary replay/source evidence and the new materialized-MHA diagnostic prevent a lazy conclusion that "Gemma is broken" or that "the custom attention kernel is the whole bug." At the sampled boundaries, raw-body mixed attention is exact against body-store replay, active and body K/V sources match graph K/V, corrected ISWA tensor alignment shows `kqv_out`/`kqv_wo` are close to native, and raw-body materialized native MHA passes the KL gate. The matched-batch bisect rules out layers 0..4 and the layer-5 input. Focus next on why the packed KVarN body values are not faithful enough for Gemma layer 5 despite K8/V8: scale layout, K/V quantization objective, f16 restoration, per-record outliers, and whether Gemma needs a native/q8_0 K escape hatch plus safer V-only compression.

K8/V2 is therefore not a viable production candidate yet. It has better nominal compression, but the measured KL is much worse and the PPL ratio is misleadingly below baseline, showing that PPL alone is not adequate for correctness.

TurboQuant's production policy is also materially different: it generally keeps K at `f16` or `q8_0` on high-GQA models and compresses V, with boundary/layer protection for aggressive V settings. KVarN `K8` still routes K through KVarN's body representation and custom mixed-attention path, so it is not equivalent to TurboQuant's native `q8_0` K escape hatch. For Gemma layer 5, even packed V8 with raw K is not clean, so a K-only native escape hatch is not sufficient for all Gemma full-attention donors.

## Code Changes From This Investigation

- Added an opt-in diagnostic env:
  `LLAMA_KVARN_ATTN_FORCE_COMPACT_CAUSAL_MASK=1`
- The accuracy gate now records that env in summaries and allows it under `-AllowDiagnosticEnv`.
- Fixed `GGML_OP_KVARN_MATERIALIZE_KV` CUDA dispatch output strides for the normal `[head_dim, n_head_kv, n_kv]` MHA cache layout.
- Added Turbo2/Turbo3/Turbo4 V-only raw-body oracles to `scripts\kvarn\analyze_boundary_quant_error.py`.
- Added a no-quant Turbo f16 offline oracle and diagnostic graph-level Turbo V frame flags. The graph flags are for causal testing only and do not implement TurboQuant compression.
- Rebuilt and used `llama-ppl-local.exe` because `llama-perplexity.exe` remains App-Control blocked.

## Next Work

1. Do not proceed to production speed claims for Gemma KVarN until packed K8/V8 passes KL on at least single-layer layer-filter gates.
2. Keep `LLAMA_KVARN_ISWA_DEBUG_MATERIALIZE_MHA=1` as the first correctness separator: raw mirror should pass, packed K8/V8 should be the target under test, and K8/V2 should not be interpreted as viable while packed K8/V8 fails.
3. Investigate packed body fidelity directly: compare materialized packed K/V tensors against raw mirror/native K/V over the exact top-KL rows and quantify K vs V contribution separately.
4. For any temporary product route, Gemma must conservatively avoid KVarN on these full-attention layers or use a higher-fidelity body format for both sides where measured sensitive; a K-only native escape hatch should still be implemented for parity with TurboQuant, but it is not enough for layer 5.
5. Prototype the next V codec as native/raw/high-fidelity K plus Turbo-style rotated-domain V, starting with Turbo4 as the correctness target and Turbo3 as the compression target. K8/V2 should only be revisited after this path is KL-clean.
