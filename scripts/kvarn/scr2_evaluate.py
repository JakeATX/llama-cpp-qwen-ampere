#!/usr/bin/env python3
"""Evaluate one sealed 16K calibration or holdout capture with SCR2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import scr2_codec as scr2


GATES = {
    "aggregate_mean_nmse_max": 0.0060,
    "global_max_nmse_max": 0.024113,
    "paired_v4_mean_ratio_max": 1.5,
    "paired_v4_max_ratio_max": 1.25,
    "gemma_l11_call0_mean_nmse_max": 0.000594,
    "gemma_l11_call0_max_nmse_max": 0.023191,
    "allocated_bits_per_value_max": 2.90,
    "record_bits_per_value_strict_max": 3.0,
    "capture_replay_nmse_max": 1.0e-5,
    "capture_replay_max_abs_max": 5.0e-3,
}
UNIT_GATE_KEYS = (
    "capture_replay",
    "global_max",
    "paired_v4_mean",
    "paired_v4_max",
    "gemma_l11_call0",
    "aggregate_rate",
    "every_record_rate",
    "no_overflow_resize_truncation_fallback",
)


class EvaluationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise EvaluationError(f"required file is missing: {path}")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected_sha256):
        raise EvaluationError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    observed = sha256_file(path)
    if observed != expected_sha256.lower():
        raise EvaluationError(f"SHA-256 mismatch for {path}: {observed}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def write_new(path: Path, data: bytes) -> dict[str, Any]:
    if path.exists():
        raise EvaluationError(f"create-new output already exists: {path}")
    path.write_bytes(data)
    return identity(path)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def tensor_bytes(tensor: torch.Tensor, dtype: str = "<f8") -> bytes:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=dtype).tobytes(order="C")


def validate_protocol(
    protocol: dict[str, Any], support_root: Path, unit_id: str, role: str,
) -> dict[str, Any]:
    if protocol.get("schema") != "kvarn-scr2-protocol-preregistration-v1":
        raise EvaluationError("unsupported SCR2 protocol schema")
    if protocol.get("candidate", {}).get("name") != "SCR2":
        raise EvaluationError("protocol does not bind the SCR2 candidate")
    units = (
        protocol.get("calibration", {}).get("evaluation_order")
        if role == "calibration"
        else protocol.get("holdout", {}).get("evaluation_order")
    )
    if not isinstance(units, list) or unit_id not in units:
        raise EvaluationError(f"unit is not in the preregistered {role} roster")
    implementation = protocol.get("implementation", {})
    required = {
        "codec": SCRIPT_DIR / "scr2_codec.py",
        "evaluator": Path(__file__).resolve(),
    }
    observed: dict[str, Any] = {}
    for name, path in required.items():
        expected = implementation.get(name)
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise EvaluationError(f"protocol lacks implementation identity {name}")
        observed[name] = identity(path)
        if observed[name]["sha256"] != expected["sha256"]:
            raise EvaluationError(f"implementation identity mismatch: {name}")
    if protocol.get("gates") != GATES:
        raise EvaluationError("protocol gates differ from evaluator gates")
    if protocol.get("unit_gate_keys") != list(UNIT_GATE_KEYS):
        raise EvaluationError("protocol unit-gate schema differs from the evaluator")
    return observed


def validate_loaded_support(protocol: dict[str, Any], support_root: Path) -> dict[str, Any]:
    expected = protocol.get("implementation", {}).get("support_modules")
    if not isinstance(expected, dict) or not expected:
        raise EvaluationError("protocol lacks the loaded support-module closure")
    observed: dict[str, Any] = {}
    root = support_root.resolve()
    source_root = root / "scripts"
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix.lower() != ".py" or not path.is_relative_to(source_root):
            continue
        relative = path.relative_to(root).as_posix()
        observed[relative] = identity(path)
    observed_hashes = {name: item["sha256"] for name, item in sorted(observed.items())}
    if observed_hashes != expected:
        missing = sorted(set(expected) - set(observed_hashes))
        extra = sorted(set(observed_hashes) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(observed_hashes)
            if expected[name] != observed_hashes[name]
        )
        raise EvaluationError(
            f"loaded support closure mismatch: missing={missing} extra={extra} changed={changed}"
        )
    required = {
        "scripts/kvarn/analyze_product_vq.py",
        "scripts/kvarn/analyze_boundary_quant_error.py",
        "scripts/kvarn/canonical_capture.py",
    }
    if not required.issubset(observed):
        raise EvaluationError("loaded support closure lacks a required evaluator module")
    return observed


def validate_environment(protocol: dict[str, Any]) -> dict[str, object]:
    torch.set_num_threads(1)
    observed = scr2.environment_identity()
    if protocol.get("environment") != observed:
        raise EvaluationError("CPU/PyTorch/BLAS fitting environment differs from the protocol")
    return observed


def _resolved_text(path: Path) -> str:
    return path.resolve().as_posix().lower()


def validate_capture_admission(
    protocol: dict[str, Any],
    args: argparse.Namespace,
    receipt: dict[str, Any],
    receipt_identity: dict[str, Any],
    source_seal: dict[str, Any],
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("role") != args.role or receipt.get("unit_id") != args.unit_id:
        raise EvaluationError("capture receipt role or unit differs from the requested evaluation")
    section = protocol.get(args.role, {})
    admitted = section.get("admitted_units", {}).get(args.unit_id)
    if not isinstance(admitted, dict):
        raise EvaluationError("protocol lacks exact source admission for this unit")
    capture_root = args.capture_receipt.resolve().parent
    if args.role == "calibration":
        if _resolved_text(capture_root) != admitted.get("capture_root", "").lower():
            raise EvaluationError("calibration capture root is not the preregistered create-new root")
        forbidden = ("/holdout/", "/validation/", "/results/")
        if any(fragment in _resolved_text(capture_root) for fragment in forbidden):
            raise EvaluationError("calibration admission rejects prior holdout/validation/result paths")
    else:
        fixture_binding = read_bound_json(
            args.fixture_binding.resolve(), args.fixture_binding_sha256
        )
        capture_binding = read_bound_json(
            args.capture_binding.resolve(), args.capture_binding_sha256
        )
        if capture_binding.get("capture_root", "").lower() != _resolved_text(capture_root):
            raise EvaluationError("holdout capture root differs from its immutable binding")
        if capture_binding.get("capture_receipt_sha256") != receipt_identity["sha256"]:
            raise EvaluationError("holdout capture binding does not bind the launch receipt")

    if args.boundary.resolve().parent.parent != capture_root:
        raise EvaluationError("boundary directory is outside the bound capture root")
    if args.body_records.resolve() != capture_root / "body-records":
        raise EvaluationError("body-record directory is outside the bound capture root")
    if args.source_seal.resolve().parent != capture_root:
        raise EvaluationError("source-seal file is outside the bound capture root")
    if args.record_set != admitted.get("record_set"):
        raise EvaluationError("record-set selection differs from the preregistered unit")
    if Path(source_seal.get("body_session_root", "")).resolve() != args.body_records.resolve():
        raise EvaluationError("source seal body root differs from the requested capture")
    for source in source_seal.get("sources", []):
        if not Path(source.get("path", "")).resolve().is_relative_to(capture_root):
            raise EvaluationError("source seal contains a path outside the bound capture root")

    identities = receipt.get("identities", {})
    toolchain = protocol.get("capture_toolchain", {})
    for name in ("capture_executable", "capture_script", "delegated_operative_script"):
        if identities.get(name, {}).get("sha256") != toolchain.get(name):
            raise EvaluationError(f"capture receipt tool identity differs from the protocol: {name}")
    observed_operative = {
        name: value.get("sha256")
        for name, value in identities.get("operative_scripts", {}).items()
    }
    if observed_operative != toolchain.get("operative_scripts"):
        raise EvaluationError("capture receipt operative-script closure differs from the protocol")
    for name in ("fixture", "model", "fixture_registration"):
        if args.role == "calibration" or name == "model":
            expected_sha = admitted.get(f"{name}_sha256")
        elif name == "fixture":
            expected_sha = fixture_binding["units"][args.unit_id].get("fixture_sha256")
        else:
            expected_sha = fixture_binding.get("fixture_registration_sha256")
        if identities.get(name, {}).get("sha256") != expected_sha:
            raise EvaluationError(f"capture receipt {name} is not an admitted source identity")
    if receipt.get("family") != admitted.get("family"):
        raise EvaluationError("capture receipt family differs from the admitted source")
    if receipt.get("topology") != admitted.get("topology"):
        raise EvaluationError("capture receipt topology differs from the preregistered unit")
    if receipt.get("runtime_signature") != admitted.get("runtime_signature"):
        raise EvaluationError("capture receipt runtime signature differs from the preregistered command")
    return {"receipt": receipt_identity, "admission": admitted, "lifecycle": lifecycle}


def validate_holdout_lifecycle(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    protocol_identity: dict[str, Any],
    source_seal_identity: dict[str, Any],
) -> dict[str, Any]:
    names = ("calibration_freeze", "fixture_binding", "capture_binding")
    if args.role == "calibration":
        if any(getattr(args, name) is not None or getattr(args, f"{name}_sha256") for name in names):
            raise EvaluationError("calibration evaluation forbids holdout lifecycle manifests")
        return {}
    for name in names:
        if getattr(args, name) is None or not getattr(args, f"{name}_sha256"):
            raise EvaluationError(f"holdout evaluation requires --{name.replace('_', '-')}")
    calibration_freeze = read_bound_json(
        args.calibration_freeze.resolve(), args.calibration_freeze_sha256
    )
    fixture_binding = read_bound_json(
        args.fixture_binding.resolve(), args.fixture_binding_sha256
    )
    capture_binding = read_bound_json(
        args.capture_binding.resolve(), args.capture_binding_sha256
    )
    if calibration_freeze.get("schema") != "kvarn-scr2-calibration-freeze-v1":
        raise EvaluationError("invalid SCR2 calibration-freeze schema")
    if fixture_binding.get("schema") != "kvarn-scr2-fresh-fixture-binding-v1":
        raise EvaluationError("invalid SCR2 fixture-binding schema")
    if capture_binding.get("schema") != "kvarn-scr2-holdout-capture-binding-v1":
        raise EvaluationError("invalid SCR2 holdout-capture-binding schema")
    protocol_sha = protocol_identity["sha256"]
    if calibration_freeze.get("protocol_sha256") != protocol_sha:
        raise EvaluationError("calibration freeze does not link the protocol")
    calibration_roster = protocol.get("calibration", {}).get("evaluation_order")
    if (
        calibration_freeze.get("role") != "calibration"
        or calibration_freeze.get("roster") != calibration_roster
        or calibration_freeze.get("evaluated_units") != calibration_roster
        or calibration_freeze.get("pass") is not True
        or calibration_freeze.get("fresh_fixture_creation_authorized") is not True
        or set(calibration_freeze.get("unit_assets", {})) != set(calibration_roster or [])
    ):
        raise EvaluationError("calibration freeze is failed, incomplete, or not fixture-authorizing")
    for unit_id in calibration_roster:
        rotation = calibration_freeze["unit_assets"][unit_id].get("rotation_fp16le")
        if not isinstance(rotation, dict) or not isinstance(rotation.get("sha256"), str):
            raise EvaluationError(f"calibration freeze lacks a frozen rotation for {unit_id}")
    if (
        fixture_binding.get("protocol_sha256") != protocol_sha
        or fixture_binding.get("calibration_freeze_sha256") != args.calibration_freeze_sha256.lower()
    ):
        raise EvaluationError("fixture binding does not link the protocol and calibration freeze")
    if (
        capture_binding.get("protocol_sha256") != protocol_sha
        or capture_binding.get("calibration_freeze_sha256") != args.calibration_freeze_sha256.lower()
        or capture_binding.get("fixture_binding_sha256") != args.fixture_binding_sha256.lower()
        or capture_binding.get("unit_id") != args.unit_id
        or capture_binding.get("source_seal_sha256") != source_seal_identity["sha256"]
    ):
        raise EvaluationError("holdout capture binding does not link the sealed unit sources")
    fixture_unit = fixture_binding.get("units", {}).get(args.unit_id)
    if not isinstance(fixture_unit, dict):
        raise EvaluationError("fixture binding does not admit the requested holdout unit")
    if (
        capture_binding.get("family") != fixture_unit.get("family")
        or capture_binding.get("topology") != fixture_unit.get("topology")
        or capture_binding.get("fixture_sha256") != fixture_unit.get("fixture_sha256")
    ):
        raise EvaluationError("holdout capture binding differs from the fixture-bound unit")
    return {
        "calibration_freeze": identity(args.calibration_freeze),
        "fixture_binding": identity(args.fixture_binding),
        "capture_binding": identity(args.capture_binding),
    }


def full_value_window(dump: dict[str, Any], body: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            dump["sink_tail_v"][:dump["n_sink"]],
            body,
            dump["pending_v"],
            dump["sink_tail_v"][dump["n_sink"]:],
        ),
        axis=0,
    ).astype(np.float32)


def ratio_pass(value: float, reference: float, maximum_ratio: float) -> bool:
    if reference == 0.0:
        return value == 0.0
    return value <= maximum_ratio * reference


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = read_bound_json(protocol_path, args.protocol_sha256)
    protocol_identity = identity(protocol_path)
    support_root = args.support_root.resolve()
    implementation = validate_protocol(protocol, support_root, args.unit_id, args.role)
    environment = validate_environment(protocol)
    source_seal = read_bound_json(args.source_seal.resolve(), args.source_seal_sha256)
    source_seal_identity = identity(args.source_seal.resolve())
    capture_receipt = read_bound_json(
        args.capture_receipt.resolve(), args.capture_receipt_sha256
    )
    capture_receipt_identity = identity(args.capture_receipt.resolve())
    lifecycle = validate_holdout_lifecycle(
        args, protocol, protocol_identity, source_seal_identity
    )

    support_scripts = support_root / "scripts" / "kvarn"
    sys.path.insert(0, str(support_scripts))
    import analyze_product_vq as product  # pylint: disable=import-outside-toplevel
    import analyze_boundary_quant_error as boundary_analysis  # pylint: disable=import-outside-toplevel
    implementation["support_modules"] = validate_loaded_support(protocol, support_root)
    admission = validate_capture_admission(
        protocol, args, capture_receipt, capture_receipt_identity, source_seal, lifecycle
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise EvaluationError("output directory must be create-new")
    output_dir.mkdir(parents=True)
    try:
        dump = product.load_dump(args.boundary.resolve(), args.body_records.resolve(), args.record_set)
        if dump["n_records"] != scr2.PRODUCTION_RECORDS or dump["group_size"] != scr2.RECORD_TOKENS:
            raise EvaluationError("capture is not exactly 126 complete 128-token records")
        if dump["n_pending"] != 0:
            raise EvaluationError("production SCR2 capture must not contain a partial pending record")
        meta = dump["meta"]
        if int(meta["n_tokens"]) != scr2.PRODUCTION_CONTEXT_TOKENS:
            raise EvaluationError("capture context is not exactly 16384 tokens")
        if int(meta["n_sink"]) != scr2.PRODUCTION_SINK_TOKENS or int(meta["n_tail"]) != scr2.PRODUCTION_TAIL_TOKENS:
            raise EvaluationError("capture sink/tail policy differs from the protocol")
        if not dump["paper_frame"] or dump["paper_mixed_frame"]:
            raise EvaluationError("capture is not in the required KVarN paper frame")
        if dump["canonical_capture"]["frame"].get("input_already_rotated") is not True:
            raise EvaluationError("capture body input is not already paper-frame rotated")

        replay_receipt = product.verify_canonical_capture_replay(
            dump["boundary"],
            dump["record_paths"],
            product.CANONICAL_TRANSFORM,
            dump["canonical_capture"],
        )
        if source_seal.get("seal_sha256") != dump["canonical_capture"].get("seal_sha256"):
            raise EvaluationError("bound source seal differs from the canonical capture")

        head_dim = dump["head_dim"]
        n_queries = int(meta["n_queries"])
        n_head = int(meta["n_head"])
        n_head_kv = int(meta["n_head_kv"])
        fit: scr2.SCR2Fit | None = None
        frozen_rotation_identity: dict[str, Any] | None = None
        if args.role == "calibration":
            q_path = dump["boundary"] / "full_q_body.bin"
            if not q_path.is_file():
                raise EvaluationError("required canonical full_q_body.bin is missing")
            expected_q_bytes = n_queries * n_head * head_dim * 4
            if q_path.stat().st_size != expected_q_bytes:
                raise EvaluationError("full-Q fitting source has an unexpected byte length")
            q_sources = [
                source for source in source_seal.get("sources", [])
                if Path(source.get("path", "")).resolve() == q_path.resolve()
            ]
            if len(q_sources) != 1 or q_sources[0].get("sha256") != sha256_file(q_path):
                raise EvaluationError("full_q_body.bin is not uniquely bound by the source seal")
            full_q_body = np.fromfile(q_path, dtype="<f4").reshape(n_queries, n_head, head_dim)
            fit = scr2.fit_sst_rotation(
                torch.from_numpy(full_q_body.copy()),
                torch.from_numpy(dump["raw_body_k"].copy()),
                torch.from_numpy(dump["raw_body_v"].copy()),
                n_head=n_head,
                n_head_kv=n_head_kv,
                kv_head=dump["selected_ikh"],
            )
            rotation_bytes = fit.rotation_bytes
        else:
            calibration_freeze = read_bound_json(
                args.calibration_freeze.resolve(), args.calibration_freeze_sha256
            )
            frozen = calibration_freeze["unit_assets"][args.unit_id]["rotation_fp16le"]
            rotation_path = Path(frozen["path"]).resolve()
            frozen_rotation_identity = identity(rotation_path)
            if frozen_rotation_identity["sha256"] != frozen["sha256"]:
                raise EvaluationError("frozen holdout rotation asset SHA-256 mismatch")
            rotation_bytes = rotation_path.read_bytes()
            scr2._require_orthogonal_rotation(scr2.rotation_from_bytes(rotation_bytes, head_dim))

        record_tensors = [
            torch.from_numpy(dump["raw_body_v"][index * 128:(index + 1) * 128].copy())
            for index in range(scr2.PRODUCTION_RECORDS)
        ]
        container_data = scr2.encode_container(record_tensors, rotation_bytes, head_dim=head_dim)
        container_path = output_dir / f"{args.unit_id}.scr2.bin"
        container_identity = write_new(container_path, container_data)
        parsed = scr2.parse_container(
            container_path.read_bytes(),
            expected_head_dim=head_dim,
        )
        decoded_body = np.empty_like(dump["raw_body_v"], dtype=np.float32)
        for index in reversed(range(scr2.PRODUCTION_RECORDS)):
            decoded_body[index * 128:(index + 1) * 128] = parsed.decode_record(index).numpy()

        raw_window = full_value_window(dump, dump["raw_body_v"])
        decoded_window = full_value_window(dump, decoded_body)
        _, _, raw_out = product.replay(
            dump["q_rows"], dump["q_body_rows"], dump["mask_rows"],
            dump["raw_k"], raw_window, meta, dump["scale"],
        )
        _, _, decoded_out = product.replay(
            dump["q_rows"], dump["q_body_rows"], dump["mask_rows"],
            dump["raw_k"], decoded_window, meta, dump["scale"],
        )
        quality = product.summarize_rows(raw_out, decoded_out)
        quality["rows"] = int(raw_out.shape[0])

        first_record_meta = json.loads(
            (dump["record_paths"][0] / "body_record.json").read_text(encoding="utf-8")
        )
        sinkhorn_iters = int(first_record_meta["sinkhorn_iters"])
        _, paired_v4_body = boundary_analysis.build_vllm_body(
            dump["raw_body_k"], dump["raw_body_v"], head_dim, dump["group_size"],
            8, 4, sinkhorn_iters,
        )
        _, _, paired_v4_out = product.replay(
            dump["q_rows"], dump["q_body_rows"], dump["mask_rows"],
            dump["raw_k"], full_value_window(dump, paired_v4_body), meta, dump["scale"],
        )
        paired_v4 = product.summarize_rows(raw_out, paired_v4_out)
        layout = scr2.accounting(head_dim, scr2.PRODUCTION_RECORDS)
        gate_results = {
            "capture_replay": (
                float(replay_receipt["nmse"]) <= GATES["capture_replay_nmse_max"]
                and float(replay_receipt["max_abs"]) <= GATES["capture_replay_max_abs_max"]
            ),
            "global_max": quality["max_out_nmse"] <= GATES["global_max_nmse_max"],
            "paired_v4_mean": ratio_pass(
                quality["mean_out_nmse"], paired_v4["mean_out_nmse"], GATES["paired_v4_mean_ratio_max"]
            ),
            "paired_v4_max": ratio_pass(
                quality["max_out_nmse"], paired_v4["max_out_nmse"], GATES["paired_v4_max_ratio_max"]
            ),
            "gemma_l11_call0": (
                args.unit_id != "gemma4-L11-KV0"
                or (
                    quality["mean_out_nmse"] <= GATES["gemma_l11_call0_mean_nmse_max"]
                    and quality["max_out_nmse"] <= GATES["gemma_l11_call0_max_nmse_max"]
                )
            ),
            "aggregate_rate": layout.allocated_bits_per_value <= GATES["allocated_bits_per_value_max"],
            "every_record_rate": layout.per_record_charged_bits_per_value < GATES["record_bits_per_value_strict_max"],
            "no_overflow_resize_truncation_fallback": True,
        }
        if tuple(gate_results) != UNIT_GATE_KEYS:
            raise AssertionError("internal unit-gate schema mismatch")

        assets = {"container": container_identity}
        if fit is not None:
            assets.update({
                "query_covariance_f64le": write_new(output_dir / "query_covariance.f64le", tensor_bytes(fit.query_covariance)),
                "value_covariance_f64le": write_new(output_dir / "value_covariance.f64le", tensor_bytes(fit.value_covariance)),
                "weights_f64le": write_new(output_dir / "weights.f64le", tensor_bytes(fit.weights)),
                "eigenvalues_f64le": write_new(output_dir / "eigenvalues.f64le", tensor_bytes(fit.eigenvalues)),
                "canonical_u_f64le": write_new(output_dir / "canonical_u.f64le", tensor_bytes(fit.canonical_eigenvectors)),
                "hadamard_f64le": write_new(output_dir / "hadamard.f64le", tensor_bytes(fit.hadamard)),
                "permutation_f64le": write_new(output_dir / "permutation.f64le", tensor_bytes(fit.permutation)),
                "rotation_f64le": write_new(output_dir / "rotation.f64le", tensor_bytes(fit.rotation_f64)),
                "rotation_fp16le": write_new(output_dir / "rotation.fp16le", fit.rotation_bytes),
            })
            fit_result = {
                "mode": "calibration_fit",
                "rotation_sha256": fit.rotation_sha256,
                "orthogonality_max_abs": fit.orthogonality_max_abs,
                "weight_sum": float(fit.weights.sum()),
            }
        else:
            fit_result = {
                "mode": "frozen_calibration_asset",
                "rotation": frozen_rotation_identity,
                "calibration_freeze_sha256": args.calibration_freeze_sha256.lower(),
            }
        result = {
            "schema": "kvarn-scr2-unit-result-v1",
            "unit_id": args.unit_id,
            "role": args.role,
            "protocol": protocol_identity,
            "source_seal": source_seal_identity,
            "admission": admission,
            "implementation": implementation,
            "capture": {
                "boundary": dump["boundary"].resolve().as_posix(),
                "body_records": args.body_records.resolve().as_posix(),
                "canonical_seal_sha256": dump["canonical_capture"]["seal_sha256"],
                "replay": replay_receipt,
                "frame": dump["frame_manifest"],
                "record_sources": dump["record_source_hashes"],
                "boundary_inputs": dump["boundary_input_hashes"],
            },
            "topology": {
                "layer": dump["layer"], "kv_head": dump["selected_ikh"],
                "head_dim": head_dim, "n_head": n_head, "n_head_kv": n_head_kv,
                "n_queries": n_queries, "n_records": dump["n_records"],
            },
            "fit": fit_result,
            "assets": assets,
            "quality": quality,
            "paired_v4": paired_v4,
            "body_v_nmse": float(product.nmse_all(dump["raw_body_v"], decoded_body)),
            "accounting": {
                "container_bytes": layout.total_bytes,
                "represented_body_values": layout.represented_values,
                "allocated_bits_per_body_value": layout.allocated_bits_per_value,
                "per_record_charged_bits_per_value": layout.per_record_charged_bits_per_value,
                "full_window_effective_bits_per_value": layout.full_window_effective_bits_per_value,
            },
            "gates": GATES,
            "gate_results": gate_results,
            "pass": all(gate_results.values()),
            "environment": environment,
        }
        result_path = output_dir / f"{args.unit_id}.scr2.result.json"
        result_identity = write_new(result_path, canonical_json(result))
        return {"result": result_identity, "pass": result["pass"], "quality": quality}
    except Exception:
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--source-seal", type=Path, required=True)
    parser.add_argument("--source-seal-sha256", required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--capture-receipt-sha256", required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--body-records", type=Path, required=True)
    parser.add_argument("--record-set", default="earliest", choices=("earliest", "latest"))
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--role", required=True, choices=("calibration", "holdout"))
    parser.add_argument("--calibration-freeze", type=Path)
    parser.add_argument("--calibration-freeze-sha256", default="")
    parser.add_argument("--fixture-binding", type=Path)
    parser.add_argument("--fixture-binding-sha256", default="")
    parser.add_argument("--capture-binding", type=Path)
    parser.add_argument("--capture-binding-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = evaluate(args)
    except (EvaluationError, scr2.SCR2Error, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
