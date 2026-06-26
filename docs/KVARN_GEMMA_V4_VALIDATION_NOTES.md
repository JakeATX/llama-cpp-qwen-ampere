# KVarN Gemma V4 Validation Notes

Date: 2026-06-25

## Fresh baseline

The valid Gemma 4 16k baseline for the current command surface is:

```text
artifacts\kvarn-accuracy\gemma4-q4-knownfixture16k-current-v4-b128-nowarmup-base\baseline-logits.base.bin
```

Command shape:

```powershell
-fit off --no-warmup
-m C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf
-f artifacts\kvarn-gemma-fixtures\gemma4_chat_16k_512turns.txt
-c 16384 -b 128 -ngl 99 -fa off --parse-special --chunks 1
```

Normal-vs-freshbase is clean:

```text
Mean PPL(Q) / PPL(base): 1.000000
Mean KLD:               ~0
Max KLD:                0.000355
Same top:               100%
```

Older baseline artifacts produced large KL even for normal-vs-base and should not be used for v4 conclusions.

## Donor5 K8/V8 status

Donor5 K8/V8 against the fresh baseline is still not clean:

```text
Artifact: artifacts\kvarn-rootcause\gemma4-donor5-k8v8-v4-freshbase-final-default-check2
Mean PPL ratio: 1.023509
Mean KLD:       0.180318
Max KLD:        32.292339
Same top:       95.947%
Top row:        logit_pos=13589
```

`LLAMA_KVARN_ISWA_DEBUG_FULL_NORMAL_ATTN=1` is clean against the same baseline. That isolates the failure to the KVarN mixed/body path, not the Gemma fixture, baseline, tokenizer, global/SWA routing, or compatibility fallback cache.

## Tensor dump safety

`LLAMA_KVARN_TENSOR_DUMP_DIR` uses scheduler `cb_eval`, which changes graph execution and perturbs KL. It is now disabled during `--kl-divergence` unless:

```powershell
$env:LLAMA_KVARN_TENSOR_DUMP_ALLOW_PERTURB = '1'
```

Use `LLAMA_KVARN_LOGIT_DUMP_DIR` and `LLAMA_KVARN_LOGIT_DUMP_TARGET_ROW` for non-perturbing KL-surface diagnostics.

The row-binding callback was also changed so `ask=true` only selects tensors that actually match the requested row, but even selected observation is not safe for measured KL.

## Paper-frame finding

Graph-side paper-frame is default-on from the v4 patch, but CUDA paper-frame body storage is intentionally left explicit opt-in:

```powershell
$env:LLAMA_KVARN_ENABLE_PAPER_FRAME = '1'
```

Reason: making CUDA paper-frame default-on caused catastrophic donor5 failure:

```text
Mean PPL ratio: ~7164
Mean KLD:       10.194588
Same top:       11.244%
```

`LLAMA_KVARN_DISABLE_DIRECT_RECORD_BATCH_PHASES=1` did not change that result, so the issue is not only the batched direct-record scheduler. The CUDA paper-frame body store/read contract for Gemma 512d body records needs a separate correctness fix before it can be made production default.

## Ablations already run

```text
disable direct prefill store: unchanged from default donor5 failure
disable paper frame:          PPL ratio 1.040465, mean KLD 0.160932
raw-frame materialized MHA:   PPL ratio 1.046809, mean KLD 0.179819
raw-body materialized MHA:    PPL ratio 1.015052, mean KLD 0.483636
raw-K split diagnostic:       PPL ratio 1.021049, mean KLD 0.189446
```

These do not justify moving to full Gemma K8/V8 or K8/V2 gates yet.

## Next work

1. Fix CUDA paper-frame body-store/read equivalence for 512d Gemma records, using a non-KL diagnostic or a small standalone oracle.
2. Once paper-frame body equivalence is proven, rerun donor5 K8/V8 against the fresh baseline.
3. Only after donor5 is clean, rerun full Gemma K8/V8 and K8/V2 quality.
4. Do not start speed work until the quality gate is clean.
