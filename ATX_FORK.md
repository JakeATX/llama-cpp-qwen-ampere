# ATX llama.cpp Fork

This local fork is based on the QuinsZouls TurboQuant llama.cpp fork at commit `1e5a46dccb8dd9b8d52817030cf4e334f76a8725`.

## Added Controls

- Exact arbitrary MoE layer residency:
  - `--moe-cpu-layers`
  - `--moe-gpu-layers`
  - `--moe-keep-layers`
- Packed-GGUF compatibility helper for heat-map layer-expert recommendations:
  - `--moe-promote-layer-experts`
- Exact expert and layer-expert residency through the ATX scheduler expert-slice cache:
  - `--moe-keep-experts`
  - `--moe-keep-layer-experts`
  - `--moe-residency-policy`
  - `--moe-residency-stats`

## Preserved

- Quins Qwen3.6/Qwen3.5 MoE trunk support.
- Quins MTP draft override path via `--spec-type mtp`.

## Validation

Validated locally on Qwen3.6-35B-A3B GGUFs:

- `UD-Q4_K_XL` deterministic no-policy vs `--moe-keep-experts 0-31` smoke matched output text.
- `UD-Q4_K_XL` layer-expert smoke produced nonzero resident cache hits.
- `UD-Q4_K_M` MTP smoke passed with `--parallel 1 --spec-type mtp`; logs included `MTP draft head registered`.
- `runs/atx_expert_residency/policy_matrix.json` was regenerated and validated (parquet optional via local pyarrow shim).

Notes:

- Expert residency intentionally disables CPU weight repacking so packed MoE source tensors remain host-visible to the selective scheduler copy path.
- MTP server runs require `--parallel 1`.

Metal direct decode (mul_mv_id + expert_map) is wired on Apple Silicon when `--moe-residency-mode direct|hybrid` is used.

See:

- `docs/atx-moe-residency.md`
- `docs/atx-exact-expert-residency-blocker.md`
- `scripts/atx_moe_metal_acceptance.py`
- `scripts/atx_moe_session.sh`
