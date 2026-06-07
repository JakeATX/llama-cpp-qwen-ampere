# ATX Run Artifacts

Benchmark outputs and acceptance reports are stored outside this repo by default:

```text
../runs/atx_expert_residency/
../runs/atx_moe_metal/
../runs/atx_moe_direct/
```

Regenerate policy JSON locally:

```bash
python3 scripts/atx_moe_policy_compile.py --policy-source /path/to/saliency/policies --out-dir /tmp/atx_policies
```

Metal acceptance:

```bash
./scripts/atx_moe_session.sh
```

The default production target is the local MTP Q4_K_M model:

```text
/Users/jkooker/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

## Current Metal MTP 64K Results

The autonomous checkpoint below used the original acceptance `reference` shape, which included `-ncmoe 34` and therefore compared against CPU-MoE/offload behavior:

```text
../runs/atx_moe_metal/autonomous/iter_012/
```

Summary:

- `best_candidate` (`keep_layers: 0-40`, layer mode): **76.69 tok/s** decode at `--ctx-size 64000`, `--spec-type mtp`, `--parallel 1`, Q8 KV.
- CPU-MoE/offload reference (`-ncmoe 34`): **51.64 tok/s**.
- Known-fast tail (`25-28,31-39,40`): **55.50 tok/s**.
- Hybrid top-16 direct proof: `direct_kernel_dispatch_mmvq=1122`, `direct_kernel_dispatch_mmq=66`, `resident_staging_copy_calls=0`, `hot_staging_bytes=0`.
- MTP acceptance matched reference: `17 / 28 = 0.60714`.
- Minimum gate (>=72 tok/s) passed versus that offload baseline.

Corrected full-Metal baseline:

```text
../runs/atx_moe_metal/clean_bench/qwen35_a3b_q4km_true_metal_baseline_promptx16_1024tok/
```

Summary:

- True full-Metal reference, no `-ncmoe`: **87.22 tok/s** decode, **921.40 tok/s** prefill.
- `best_candidate` (`keep_layers: 0-40`): **89.22 tok/s** decode, **929.69 tok/s** prefill.
- Delta versus true full Metal: **+2.3% decode**, **+0.9% prefill**.
- MTP acceptance matched exactly: `629 / 788 = 0.79822`.

Takeaway: on Apple Silicon, the ATX/Kvarn Metal path is now roughly at full-Metal parity. The large apparent speedup came from avoiding CPU-MoE/offload behavior, not from making normal full-Metal inference materially faster.

Hardening matrix:

```text
../runs/atx_moe_metal/autonomous/iter_010_hardening_best/
```

The best candidate remained above 72 tok/s from 4K through 64K context, with 64K at **72.02 tok/s**.

CUDA should be tested separately with three baselines: true full CUDA, whole-layer policy, and hybrid/direct expert policy. The layer/expert residency controls are expected to matter more on discrete-memory CUDA systems than on Apple unified memory.
