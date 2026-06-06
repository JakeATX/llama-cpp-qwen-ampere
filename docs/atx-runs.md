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

Set `MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` before running the session script.
