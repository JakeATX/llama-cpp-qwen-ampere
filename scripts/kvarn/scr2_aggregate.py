#!/usr/bin/env python3
"""Create an immutable SCR2 calibration freeze or holdout closure report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


class AggregateError(ValueError):
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
        raise AggregateError(f"required file is missing: {path}")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = identity(path)
    if observed["sha256"] != expected_sha256.lower():
        raise AggregateError(f"SHA-256 mismatch for {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregateError("bound JSON root is not an object")
    return value


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def build_aggregate(
    protocol: dict[str, Any],
    protocol_identity: dict[str, Any],
    role: str,
    bound_results: Sequence[tuple[str, Path, str]],
) -> dict[str, Any]:
    roster = protocol.get(role, {}).get("evaluation_order")
    if not isinstance(roster, list) or not roster:
        raise AggregateError(f"protocol lacks the {role} roster")
    if len(bound_results) > len(roster):
        raise AggregateError("more results were supplied than the preregistered roster")

    results: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    hard_failure: str | None = None
    for position, (unit_id, path, expected_sha) in enumerate(bound_results):
        if unit_id != roster[position]:
            raise AggregateError("results are not in the preregistered evaluation order")
        result = read_bound_json(path, expected_sha)
        if (
            result.get("schema") != "kvarn-scr2-unit-result-v1"
            or result.get("unit_id") != unit_id
            or result.get("role") != role
            or result.get("protocol", {}).get("sha256") != protocol_identity["sha256"]
        ):
            raise AggregateError(f"unit result identity mismatch: {unit_id}")
        if result.get("gates") != protocol.get("gates"):
            raise AggregateError(f"unit result gate vector mismatch: {unit_id}")
        gate_results = result.get("gate_results")
        expected_gate_keys = protocol.get("unit_gate_keys")
        if not isinstance(gate_results, dict) or set(gate_results) != set(expected_gate_keys or []):
            raise AggregateError("unit result has an incomplete or unexpected hard-gate schema")
        hard_pass = all(value is True for value in gate_results.values())
        if result.get("pass") is not hard_pass:
            raise AggregateError(f"unit pass flag is inconsistent: {unit_id}")
        if hard_failure is not None:
            raise AggregateError("results continue after the first hard-gate failure")
        if not hard_pass:
            hard_failure = unit_id
        quality = result.get("quality", {})
        rows = quality.get("rows")
        mean = quality.get("mean_out_nmse")
        if not isinstance(rows, int) or rows <= 0 or not isinstance(mean, (int, float)):
            raise AggregateError(f"unit quality lacks a valid row mean: {unit_id}")
        results.append(result)
        identities.append(identity(path))

    if hard_failure is None and len(results) != len(roster):
        raise AggregateError("an all-hard-pass prefix cannot close before the full roster")
    if hard_failure is not None and bound_results[-1][0] != hard_failure:
        raise AggregateError("the first hard failure must be the final supplied result")

    aggregate_mean: float | None = None
    aggregate_mean_pass: bool | None = None
    if hard_failure is None:
        rows_total = sum(result["quality"]["rows"] for result in results)
        aggregate_mean = sum(
            result["quality"]["mean_out_nmse"] * result["quality"]["rows"]
            for result in results
        ) / rows_total
        aggregate_mean_pass = (
            aggregate_mean <= protocol["gates"]["aggregate_mean_nmse_max"]
        )
    passed = hard_failure is None and aggregate_mean_pass is True
    schema = (
        "kvarn-scr2-calibration-freeze-v1"
        if role == "calibration"
        else "kvarn-scr2-holdout-closure-v1"
    )
    return {
        "schema": schema,
        "protocol": protocol_identity,
        "protocol_sha256": protocol_identity["sha256"],
        "role": role,
        "roster": roster,
        "evaluated_units": [result["unit_id"] for result in results],
        "unit_results": identities,
        "unit_assets": {
            result["unit_id"]: result["assets"] for result in results
        },
        "unit_sources": {
            result["unit_id"]: {
                "source_seal": result["source_seal"],
                "admission": result["admission"],
                "canonical_seal_sha256": result["capture"]["canonical_seal_sha256"],
            }
            for result in results
        },
        "first_hard_failure": hard_failure,
        "aggregate_mean_out_nmse": aggregate_mean,
        "aggregate_mean_gate_pass": aggregate_mean_pass,
        "pass": passed,
        "fresh_fixture_creation_authorized": role == "calibration" and passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role", choices=("calibration", "holdout"), required=True)
    parser.add_argument(
        "--result", action="append", nargs=3, metavar=("UNIT", "PATH", "SHA256"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"create-new output already exists: {args.output}")
    try:
        protocol = read_bound_json(args.protocol.resolve(), args.protocol_sha256)
        protocol_identity = identity(args.protocol.resolve())
        expected_aggregator = protocol.get("implementation", {}).get("aggregator", {}).get("sha256")
        if expected_aggregator != sha256_file(Path(__file__).resolve()):
            raise AggregateError("aggregator implementation differs from the protocol")
        bound = [(unit, Path(path).resolve(), sha) for unit, path, sha in args.result]
        report = build_aggregate(protocol, protocol_identity, args.role, bound)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(report))
    except (AggregateError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"output": identity(args.output), "pass": report["pass"]}, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
