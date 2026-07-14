#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scr2_aggregate as aggregate


def result(unit: str, protocol_sha: str, *, hard_pass: bool, mean: float, rows: int = 2) -> dict:
    gates = {"aggregate_mean_nmse_max": 0.006}
    gate_results = {"global_max": hard_pass, "rate": hard_pass}
    return {
        "schema": "kvarn-scr2-unit-result-v1", "unit_id": unit, "role": "calibration",
        "protocol": {"sha256": protocol_sha}, "gates": gates,
        "gate_results": gate_results, "pass": hard_pass,
        "quality": {"mean_out_nmse": mean, "rows": rows}, "assets": {},
        "source_seal": {}, "admission": {}, "capture": {"canonical_seal_sha256": "a" * 64},
    }


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text("{}\n", encoding="ascii")
        self.protocol_id = aggregate.identity(self.protocol_path)
        self.protocol = {
            "calibration": {"evaluation_order": ["u0", "u1"]},
            "gates": {"aggregate_mean_nmse_max": 0.006},
            "unit_gate_keys": ["global_max", "rate"],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, value: dict) -> tuple[str, Path, str]:
        path = self.root / f"{name}.json"
        path.write_bytes(aggregate.canonical_json(value))
        return name, path, aggregate.sha256_file(path)

    def test_weighted_aggregate_mean_is_roster_only(self) -> None:
        bound = [
            self.write("u0", result("u0", self.protocol_id["sha256"], hard_pass=True, mean=0.010, rows=1)),
            self.write("u1", result("u1", self.protocol_id["sha256"], hard_pass=True, mean=0.001, rows=9)),
        ]
        report = aggregate.build_aggregate(self.protocol, self.protocol_id, "calibration", bound)
        self.assertAlmostEqual(report["aggregate_mean_out_nmse"], 0.0019)
        self.assertTrue(report["pass"])
        self.assertTrue(report["fresh_fixture_creation_authorized"])

    def test_first_hard_failure_closes_prefix_without_aggregate(self) -> None:
        bound = [self.write("u0", result("u0", self.protocol_id["sha256"], hard_pass=False, mean=0.1))]
        report = aggregate.build_aggregate(self.protocol, self.protocol_id, "calibration", bound)
        self.assertEqual(report["first_hard_failure"], "u0")
        self.assertIsNone(report["aggregate_mean_out_nmse"])
        self.assertFalse(report["pass"])

    def test_all_pass_prefix_and_wrong_order_rejected(self) -> None:
        one = self.write("u0", result("u0", self.protocol_id["sha256"], hard_pass=True, mean=0.001))
        with self.assertRaisesRegex(aggregate.AggregateError, "full roster"):
            aggregate.build_aggregate(self.protocol, self.protocol_id, "calibration", [one])
        wrong = self.write("u1", result("u1", self.protocol_id["sha256"], hard_pass=False, mean=0.1))
        with self.assertRaisesRegex(aggregate.AggregateError, "order"):
            aggregate.build_aggregate(self.protocol, self.protocol_id, "calibration", [wrong])

    def test_unit_cannot_claim_aggregate_gate(self) -> None:
        value = result("u0", self.protocol_id["sha256"], hard_pass=False, mean=0.1)
        value["gate_results"]["aggregate_mean"] = False
        bound = [self.write("u0", value)]
        with self.assertRaisesRegex(aggregate.AggregateError, "hard-gate schema"):
            aggregate.build_aggregate(self.protocol, self.protocol_id, "calibration", bound)

    def test_empty_or_partial_hard_gate_schema_rejected(self) -> None:
        value = result("u0", self.protocol_id["sha256"], hard_pass=True, mean=0.001)
        value["gate_results"] = {}
        bound = [self.write("u0", value)]
        with self.assertRaisesRegex(aggregate.AggregateError, "hard-gate schema"):
            aggregate.build_aggregate(self.protocol, self.protocol_id, "calibration", bound)


if __name__ == "__main__":
    unittest.main()
