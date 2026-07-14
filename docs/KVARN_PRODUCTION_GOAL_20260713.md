# KVarN Production Goal and Milestone Status - 2026-07-13

## Persistent goal

Autonomously develop, test, and refine KVarN K8/V2-class KV-cache
compression for Gemma 4-12B and Qwen3.6-35A3B. The production target is a
functionally near-lossless codec that beats TurboQuant-3 compression while
retaining at least 95% of mainline llama.cpp q8/q8 speed. Freeze and test the
D2/K50/R16 product-VQ CPU codec and its fully serialized rate accounting
first. Run a preregistered blind, matched-frame, multi-donor Gemma/Qwen
holdout only after that freeze. Implement CUDA only after every offline gate
passes. If product VQ fails, test exactly one frozen fixed-rate 22-bit/8D
analytic lattice arm under the same gates. Stop only when the production
gates pass or both frozen codec families are rigorously closed with
reproducible evidence.

## Agent runtime configuration

The root Codex context window and every custom subagent context window are
capped at 272000 tokens.

| Agent | Model | Reasoning | Context | Role |
| --- | --- | --- | ---: | --- |
| root | `gpt-5.6-sol` | `medium` | 272000 | primary autonomous coordinator |
| `devils_advocate` | `gpt-5.6-sol` | `medium` | 272000 | plan and delivery integrity gate |
| `explorer` | `gpt-5.6-luna` | `xhigh` | 272000 | read-only analyzer/capture audit |
| `worker` | `gpt-5.6-sol` | `medium` | 272000 | canonical capture validator and tests |

The corresponding personal files are under `C:\Users\sjake\.codex\agents`.
The global `C:\Users\sjake\.codex\config.toml` supplies the root Sol/medium
defaults and is also capped at 272000. Future assignments use each agent's
TOML model and reasoning setting rather than a blanket model configuration.

## Rollup scope and evidence portability

This document accompanies a runtime-hardening rollup, but the frozen codec
investigation and the pre-existing K8/V8 CUDA runtime are distinct. The
offline rejection below closes only the experimental D2/K50/R16 product-VQ
and IQ2_S codec families. It forbids implementing or promoting either failed
candidate, but it does not prohibit safety, rollback, persistence-policy, or
scratch-contract corrections to the existing K8/V8 runtime.

`production_ready=false`. The rollup improves runtime integrity; it does not
establish the required codec quality, model-scale parity, memory saving, or
performance gates.

The three included JSON files are compact, byte-identical terminal decision
records with hash-bound local provenance:

- [product-VQ result](../artifacts/kvarn-product-vq-production-20260713-v4/results/gemma4-L11-KV0.result.json)
- [IQ2_S result](../artifacts/kvarn-iq2-lattice-production-20260713/gemma4-L11-KV0.iq2_s.result.json)
- [IQ2_S closure](../artifacts/kvarn-iq2-lattice-production-20260713/gemma4-L11-KV0.iq2_s.closure.json)

They are not a self-contained reproducibility bundle. Historical absolute
paths inside the records identify omitted analyzers, registrations, source
seals, the compiled helper, container data, models, and raw captures retained
in the originating review worktree. Those larger inputs are intentionally not
published in this rollup, and the closure candidly records that the lattice
family and evaluation order were predeclared without a separate immutable
lattice preregistration manifest.

## Completed CPU-freeze infrastructure milestone

- `scripts/kvarn/analyze_product_vq.py` now defaults to D2/K50/R16 and
  requires disjoint calibration plus an immutable registration.
- Registration is canonical JSON, create-new or exact-match only, and is
  verified after source sealing but before raw replay observations or codec
  quality metrics.
- Payload capacity is frozen from calibration. Evaluation overflow raises an
  error; it cannot resize, truncate, or fall back.
- The fixed-stride container now serializes its header, fp16 centroids, rANS
  frequency table, record table, log8 scales, packed residual indices, fp16
  residual values, rANS payloads, alignment, and per-record checksum.
- Quality is measured from bytes parsed and decoded from that container, not
  from the prior in-memory approximation.
- R16 residual indices use the charged packed local-index representation.
  The earlier u32 declaration/15-bit accounting mismatch is removed.
- Capture sealing requires complete full-Q/O/mask artifacts, exact sizes,
  canonical frame/layout/topology, coherent record cohorts, no used mirror or
  fallback, a frozen transform descriptor hash, source hashes, and raw CPU
  replay parity against stored `full_out`.
- Save/load, reverse-order random access, malformed/truncated/tampered input,
  registration mismatch, frame mismatch, replay mismatch, and frozen-capacity
  overflow have focused CPU tests.

## Verification

- Python compilation: pass.
- Product-VQ analyzer self-test: pass, including K50, 876-byte D512/G128
  metadata, fixed-stride accounting, frozen overflow rejection, and byte-real
  reverse decode.
- Focused unit tests: 13 passed. The analyzer self-test separately covers
  malformed/trailing rANS, valid-checksum malformed payloads, and frozen
  overflow behavior.
- No GPU job and no CUDA implementation were run for this milestone.

## Terminal frozen-codec result

The production investigation reached its declared stop condition on
2026-07-13. Both permitted frozen experimental codec families are closed;
CUDA implementation or promotion of either failed candidate is therefore
forbidden by the offline-first plan.

- The sealed blind `gemma4-L11-KV0` D2/K50/R16 product-VQ result exceeded the
  global maximum output-NMSE limit: `0.024468337607298545` observed versus
  `0.024113` allowed. Rate (`2.8638671875` allocated bits/value), overflow,
  fallback, and source replay all passed. Because one unit failure rejects the
  pilot, no further product-VQ donor capture can change the decision.
- Product-VQ result:
  `artifacts/kvarn-product-vq-production-20260713-v4/results/gemma4-L11-KV0.result.json`
  (SHA-256 `0840471b7e25286cc87882ca6402cc7f7ba0567afc28248ad1180f93ace4f978`).
- The subsequent `gemma4-L41-KV0` attempt omitted the donor layer filter and
  was correctly rejected by the existing unexpected-layer gate before body
  capture. Its directory is retained only as a non-admissible diagnostic; no
  metric or result claim is made from it.
- The one permitted fixed 22-bit/8D analytic lattice arm (`iq2_s`, 16
  Sinkhorn iterations) was then tested against the valid sealed L11 capture.
  Its byte-real rate passed at `2.720703125` bits/value, but quality failed:
  mean output NMSE was `0.003144709379337122` versus `0.000594` allowed, and
  maximum output NMSE was `0.047938421569674414` versus `0.023191` allowed.
- Lattice result:
  `artifacts/kvarn-iq2-lattice-production-20260713/gemma4-L11-KV0.iq2_s.result.json`
  (SHA-256 `0d852c25dfa4ded31bc18847e17e08e885ad227d6cadeebc04de6bbe9f6f9099`).
- Canonical lattice closure report, binding the exact analyzer, helper,
  configuration, sealed capture/registration, source replay, result, complete
  gate vector, and inapplicable fixed-rate overflow/fallback fields:
  `artifacts/kvarn-iq2-lattice-production-20260713/gemma4-L11-KV0.iq2_s.closure.json`
  (SHA-256 `caa7a6f73185d1f7654d35bd1b935eba8c7dae0f0fb63e7da404f2ca7ff4473e`).
  The report records the exact limitation that the family and evaluation
  order were predeclared but no separate immutable lattice preregistration
  manifest existed; it does not retroactively claim one.
- The in-tree lattice helper self-test passed for `iq2_xxs`, `iq2_xs`, and
  `iq2_s`; the Python accounting/normalization self-test also passed. The
  quality rejection is therefore a measured codec outcome, not a broken
  executable or serialization failure.

## Terminal SCR2 calibration result

The preregistered SCR2 arm is also closed at its first-failure stop condition.
`production_ready=false`; no SCR2 CUDA integration or promotion is authorized.

- Protocol SHA-256:
  `e6ef0bcf9c94911d2b2da08a0825fc5f643b01104009202a4de291aef643cf62`.
- Successful 16K `gemma4-L11-KV0` capture receipt SHA-256:
  `99e82b1675afe5b0c0b5eed19360673ac621c2485bf927f03c074578cfcb5959`.
- Sealed source manifest SHA-256:
  `544bab3c4d33b826467b3319b6c40e46b1549cd324550e7b3d1e4a393a1dd767`.
- Exact terminal result:
  `artifacts/kvarn-scr2-calibration-20260714/gemma4-L11-KV0.scr2.result.json`
  (SHA-256 `27526d7ad90c8dc5e3c6a0185ece18b9b79bd686a15f86152e178afb43a33919`).
- Capture replay, aggregate rate, every-record rate, and the no-overflow /
  no-resize / no-truncation / no-fallback integrity gate passed. Allocated
  body rate was `2.759982638888889` bits/value.
- Quality failed decisively: mean output NMSE was
  `0.01376441831072871` and maximum output NMSE was
  `0.15791865566019775`. The `gemma_l11_call0`, `global_max`,
  `paired_v4_mean`, and `paired_v4_max` gates all failed.
- The protocol declares one unit failure terminal. No later SCR2 calibration
  unit, aggregation, calibration freeze, holdout, tuning, or CUDA integration
  was run or may be inferred from this result.

## Legacy validation rejection

The existing Gemma layer-5 disjoint paper-frame capture was tested only as a
validation input. It was rejected before registration and before codec
metrics because raw CPU replay did not match stored `full_out`:

- replay NMSE: `0.000180253466`
- replay maximum absolute error: `0.528406799`
- frozen limits: NMSE `<= 0.00001`, maximum absolute error `<= 0.005`

No corrected D2/K50/R16 rate or quality claim was emitted from this rejected
capture. This is not blind-holdout evidence and the codec is not
production-ready.

## Next causal action

None within these frozen codec investigations. Do not tune product-VQ,
IQ2_S, or SCR2 on exposed evidence; run additional rejected-arm donors; or
implement a CUDA path for a candidate that failed its offline quality gate.
Maintenance of the existing K8/V8 runtime remains separate. Any future codec
family must begin as a new preregistered investigation with fresh untouched
holdout evidence.
