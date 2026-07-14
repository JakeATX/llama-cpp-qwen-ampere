#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import types
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scr2_evaluate as evaluate


class EvaluatorContractTests(unittest.TestCase):
    def test_ratio_zero_reference(self) -> None:
        self.assertTrue(evaluate.ratio_pass(0.0, 0.0, 1.5))
        self.assertFalse(evaluate.ratio_pass(1.0e-9, 0.0, 1.5))
        self.assertTrue(evaluate.ratio_pass(1.5, 1.0, 1.5))
        self.assertFalse(evaluate.ratio_pass(1.500001, 1.0, 1.5))

    def test_bound_json_and_create_new(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "value.json"
            payload = b'{"ok":true}\n'
            created = evaluate.write_new(path, payload)
            self.assertEqual(created["sha256"], hashlib.sha256(payload).hexdigest())
            parsed = evaluate.read_bound_json(path, created["sha256"])
            self.assertEqual(parsed, {"ok": True})
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.write_new(path, payload)
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.read_bound_json(path, "0" * 64)

    def test_protocol_binds_every_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            support = Path(temporary)
            support_scripts = support / "scripts" / "kvarn"
            support_scripts.mkdir(parents=True)
            analyzer = support_scripts / "analyze_product_vq.py"
            canonical = support_scripts / "canonical_capture.py"
            analyzer.write_text("# analyzer\n", encoding="ascii")
            canonical.write_text("# canonical\n", encoding="ascii")
            expected = {
                "codec": evaluate.identity(evaluate.SCRIPT_DIR / "scr2_codec.py"),
                "evaluator": evaluate.identity(Path(evaluate.__file__)),
            }
            protocol = {
                "schema": "kvarn-scr2-protocol-preregistration-v1",
                "candidate": {"name": "SCR2"},
                "calibration": {"evaluation_order": ["gemma4-L11-KV0"]},
                "holdout": {"evaluation_order": ["gemma4-L11-KV0"]},
                "implementation": expected,
                "gates": evaluate.GATES,
                "unit_gate_keys": list(evaluate.UNIT_GATE_KEYS),
            }
            observed = evaluate.validate_protocol(protocol, support, "gemma4-L11-KV0", "calibration")
            self.assertEqual(set(observed), set(expected))
            protocol["implementation"]["codec"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(evaluate.EvaluationError, "codec"):
                evaluate.validate_protocol(protocol, support, "gemma4-L11-KV0", "holdout")

    def test_canonical_json_is_stable_ascii(self) -> None:
        first = evaluate.canonical_json({"b": 2, "a": "x"})
        second = evaluate.canonical_json(json.loads(first))
        self.assertEqual(first, b'{"a":"x","b":2}\n')
        self.assertEqual(first, second)

    def test_environment_is_enforced_before_fit(self) -> None:
        import torch
        torch.set_num_threads(1)
        protocol = {"environment": evaluate.scr2.environment_identity()}
        self.assertEqual(evaluate.validate_environment(protocol), protocol["environment"])
        protocol["environment"] = {**protocol["environment"], "torch_threads": 8}
        with self.assertRaisesRegex(evaluate.EvaluationError, "environment"):
            evaluate.validate_environment(protocol)

    def test_loaded_support_closure_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (
                "scripts/kvarn/analyze_product_vq.py",
                "scripts/kvarn/analyze_boundary_quant_error.py",
                "scripts/kvarn/canonical_capture.py",
            )
            inserted = []
            expected = {}
            try:
                for index, relative in enumerate(paths):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"# support {index}\n", encoding="ascii")
                    module = types.ModuleType(f"scr2_fake_support_{index}")
                    module.__file__ = str(path)
                    sys.modules[module.__name__] = module
                    inserted.append(module.__name__)
                    expected[relative] = evaluate.sha256_file(path)
                protocol = {"implementation": {"support_modules": expected}}
                observed = evaluate.validate_loaded_support(protocol, root)
                self.assertEqual(set(observed), set(paths))
                protocol["implementation"]["support_modules"][paths[0]] = "0" * 64
                with self.assertRaisesRegex(evaluate.EvaluationError, "changed"):
                    evaluate.validate_loaded_support(protocol, root)
            finally:
                for name in inserted:
                    sys.modules.pop(name, None)

    def test_calibration_capture_admission_rejects_wrong_role_and_old_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "validation" / "unit"
            boundary = root / "boundary" / "call_000000"
            body = root / "body-records"
            boundary.mkdir(parents=True)
            body.mkdir()
            receipt_path = root / "capture_launch_receipt.json"
            receipt_path.write_text("{}\n", encoding="ascii")
            receipt_identity = evaluate.identity(receipt_path)
            fixture_sha, model_sha, registration_sha = "1" * 64, "2" * 64, "3" * 64
            topology = {"layer": 11, "kv_head": 0}
            signature = {"context_size": 16384, "body_record_limit": 126}
            admitted = {
                "capture_root": root.resolve().as_posix().lower(), "family": "gemma4",
                "fixture_sha256": fixture_sha, "model_sha256": model_sha,
                "fixture_registration_sha256": registration_sha,
                "topology": topology, "runtime_signature": signature, "record_set": "earliest",
            }
            protocol = {"calibration": {"admitted_units": {"gemma4-L11-KV0": admitted}}}
            receipt = {
                "role": "calibration", "unit_id": "gemma4-L11-KV0", "family": "gemma4",
                "identities": {"fixture": {"sha256": fixture_sha}, "model": {"sha256": model_sha},
                               "fixture_registration": {"sha256": registration_sha}},
                "topology": topology, "runtime_signature": signature,
            }
            source_file = boundary / "boundary.json"
            source_file.write_text("{}\n", encoding="ascii")
            source_seal_path = root / "source-seal.official.json"
            source_seal_path.write_text("{}\n", encoding="ascii")
            source_seal = {"body_session_root": str(body), "sources": [{"path": str(source_file)}]}
            args = SimpleNamespace(
                role="calibration", unit_id="gemma4-L11-KV0", capture_receipt=receipt_path,
                boundary=boundary, body_records=body, source_seal=source_seal_path,
                record_set="earliest",
            )
            receipt["role"] = "holdout"
            with self.assertRaisesRegex(evaluate.EvaluationError, "role"):
                evaluate.validate_capture_admission(
                    protocol, args, receipt, receipt_identity, source_seal, {}
                )
            receipt["role"] = "calibration"
            with self.assertRaisesRegex(evaluate.EvaluationError, "prior holdout"):
                evaluate.validate_capture_admission(
                    protocol, args, receipt, receipt_identity, source_seal, {}
                )

    def test_holdout_lifecycle_links_every_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            protocol.write_text("{}\n", encoding="ascii")
            protocol_id = evaluate.identity(protocol)
            source = root / "source.json"
            source.write_text("{}\n", encoding="ascii")
            source_id = evaluate.identity(source)
            rotation = root / "rotation.fp16le"
            rotation.write_bytes(b"rotation")
            rotation_id = evaluate.identity(rotation)

            calibration = root / "calibration.json"
            calibration.write_bytes(evaluate.canonical_json({
                "schema": "kvarn-scr2-calibration-freeze-v1",
                "protocol_sha256": protocol_id["sha256"],
                "role": "calibration",
                "roster": ["gemma4-L11-KV0"],
                "evaluated_units": ["gemma4-L11-KV0"],
                "pass": True,
                "fresh_fixture_creation_authorized": True,
                "unit_assets": {"gemma4-L11-KV0": {"rotation_fp16le": rotation_id}},
            }))
            calibration_id = evaluate.identity(calibration)
            fixture = root / "fixture.json"
            fixture.write_bytes(evaluate.canonical_json({
                "schema": "kvarn-scr2-fresh-fixture-binding-v1",
                "protocol_sha256": protocol_id["sha256"],
                "calibration_freeze_sha256": calibration_id["sha256"],
                "units": {"gemma4-L11-KV0": {
                    "family": "gemma4", "topology": {"layer": 11, "kv_head": 0},
                    "fixture_sha256": "1" * 64,
                }},
            }))
            fixture_id = evaluate.identity(fixture)
            capture = root / "capture.json"
            capture.write_bytes(evaluate.canonical_json({
                "schema": "kvarn-scr2-holdout-capture-binding-v1",
                "protocol_sha256": protocol_id["sha256"],
                "calibration_freeze_sha256": calibration_id["sha256"],
                "fixture_binding_sha256": fixture_id["sha256"],
                "unit_id": "gemma4-L11-KV0",
                "source_seal_sha256": source_id["sha256"],
                "family": "gemma4", "topology": {"layer": 11, "kv_head": 0},
                "fixture_sha256": "1" * 64,
            }))
            capture_id = evaluate.identity(capture)
            args = SimpleNamespace(
                role="holdout", unit_id="gemma4-L11-KV0",
                calibration_freeze=calibration,
                calibration_freeze_sha256=calibration_id["sha256"],
                fixture_binding=fixture,
                fixture_binding_sha256=fixture_id["sha256"],
                capture_binding=capture,
                capture_binding_sha256=capture_id["sha256"],
            )
            protocol_value = {"calibration": {"evaluation_order": ["gemma4-L11-KV0"]}}
            linked = evaluate.validate_holdout_lifecycle(args, protocol_value, protocol_id, source_id)
            self.assertEqual(set(linked), {"calibration_freeze", "fixture_binding", "capture_binding"})
            args.source_seal_sha256 = "unused"
            capture.write_bytes(evaluate.canonical_json({
                "schema": "kvarn-scr2-holdout-capture-binding-v1",
                "protocol_sha256": protocol_id["sha256"],
                "calibration_freeze_sha256": calibration_id["sha256"],
                "fixture_binding_sha256": fixture_id["sha256"],
                "unit_id": "wrong-unit",
                "source_seal_sha256": source_id["sha256"],
                "family": "gemma4", "topology": {"layer": 11, "kv_head": 0},
                "fixture_sha256": "1" * 64,
            }))
            args.capture_binding_sha256 = evaluate.identity(capture)["sha256"]
            with self.assertRaisesRegex(evaluate.EvaluationError, "sealed unit"):
                evaluate.validate_holdout_lifecycle(args, protocol_value, protocol_id, source_id)


if __name__ == "__main__":
    unittest.main()
