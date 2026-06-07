# ATX MoE Direct Final Report

Status: first CUDA acceptance pass completed on the locally available Unsloth Qwen3.6-35B-A3B UD-Q4_K_M and UD-Q4_K_XL GGUFs.

Completion gate:

- `direct >= 1.50x exact-v1` decode tok/s
- `direct_or_hybrid >= 1.25x CPU/offload` decode tok/s
- zero hot staging under `--moe-direct-strict-hot-no-stage`
- zero direct fallback under `--moe-direct-require`
- no-flag regression <= 3%

Measured smoke results, RTX 5070, CUDA build `e94d681`, `--ctx-size 512`, `-n 4`, temp 0:

| model | mode | generation t/s | speedup vs exact-v1 | host bytes | resident staging | direct hits | fallback | hot violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_M | exact-v1 | 3.2 | 1.00x | 8053316096 | 1089 | 0 | 0 | 0 |
| UD-Q4_K_M | hybrid top-8 | 4.0 | 1.25x | 6226434560 | 0 | 1206 | 0 | 0 |
| UD-Q4_K_M | hybrid top-16 | 7.5 | 2.34x | 4847550976 | 0 | 675 | 0 | 0 |
| UD-Q4_K_M | no policy | 7.1 | 2.22x | 0 | 0 | 0 | 0 | 0 |
| UD-Q4_K_XL | exact-v1 | 3.2 | 1.00x | 8100277248 | 1095 | 0 | 0 | 0 |
| UD-Q4_K_XL | hybrid top-8 | 7.1 | 2.22x | 6271623680 | 0 | 1203 | 0 | 0 |

The direct-only path is functional and strict: the Q4_K_M direct smoke produced zero resident staging, zero fallback dispatches, zero hot staging violations, 1089 direct hot hits, 336 MMVQ direct dispatches, and 102 MMQ direct dispatches. Direct-only did not reduce cold host traffic, so throughput gains were small. The hybrid compiler is the first practical win because it promotes dense salience layers and uses direct exact hot cells for the remaining selected experts.

Acceptance outputs are under `runs/atx_moe_direct/cuda_acceptance/`. The remaining UD-Q4, UD-Q5_K_XL, UD-Q6_K_M, and UD-Q6_K_XL files were not present locally during this pass; the runtime target is limited to standard GGML K-quant tensor types used by these Unsloth dynamic quants and excludes TurboQuant.
