#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scr2_codec as scr2


def identity_bytes(head_dim: int = 128) -> bytes:
    return np.eye(head_dim, dtype="<f2").tobytes()


def records(count: int = 2, head_dim: int = 128) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20260714)
    return [torch.randn((128, head_dim), generator=generator, dtype=torch.float32) for _ in range(count)]


def resign(data: bytearray) -> bytes:
    data[-scr2.TRAILER_BYTES:] = hashlib.sha256(data[:-scr2.TRAILER_BYTES]).digest()
    return bytes(data)


def replace_header(data: bytes, field: int, value: int) -> bytes:
    changed = bytearray(data)
    fields = list(scr2._HEADER.unpack_from(changed))
    fields[field] = value
    changed[:scr2.HEADER_BYTES] = scr2._HEADER.pack(*fields)
    return resign(changed)


class AccountingTests(unittest.TestCase):
    def test_exact_production_rates(self) -> None:
        expected = {
            512: (2848832, 2.759982638888889, 2.966857910156250),
            256: (1294400, 2.508060515873016, 2.718872070312500),
            128: (615488, 2.385168650793651, 2.597900390625000),
        }
        for head_dim, (total, rate, full_window_rate) in expected.items():
            with self.subTest(head_dim=head_dim):
                result = scr2.accounting(head_dim, 126)
                self.assertEqual(result.total_bytes, total)
                self.assertEqual(result.record_stride, 36 * head_dim)
                self.assertAlmostEqual(result.allocated_bits_per_value, rate, places=15)
                self.assertAlmostEqual(
                    result.full_window_effective_bits_per_value,
                    full_window_rate,
                    places=15,
                )
                self.assertLess(result.allocated_bits_per_value, 2.90)
                self.assertLess(result.per_record_charged_bits_per_value, 3.0)

    def test_invalid_geometry(self) -> None:
        for head_dim in (0, 64, 192, 384):
            with self.subTest(head_dim=head_dim), self.assertRaises(scr2.SCR2Error):
                scr2.accounting(head_dim)
        with self.assertRaises(scr2.SCR2Error):
            scr2.accounting(128, 0)
        with self.assertRaisesRegex(scr2.SCR2Error, "uint32"):
            scr2.accounting(512, 0xFFFFFFFF)

    def test_crc32c_known_vector(self) -> None:
        self.assertEqual(scr2.crc32c(b"123456789"), 0xE3069283)


class EigenspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_degenerate_identity_is_canonical(self) -> None:
        values, vectors = scr2.canonical_eigendecomposition(torch.eye(128, dtype=torch.float64))
        self.assertTrue(torch.equal(values, torch.ones(128, dtype=torch.float64)))
        self.assertTrue(torch.equal(vectors, torch.eye(128, dtype=torch.float64)))

    def test_sign_is_canonical(self) -> None:
        diagonal = torch.arange(1, 129, dtype=torch.float64)
        values, vectors = scr2.canonical_eigendecomposition(torch.diag(diagonal))
        self.assertTrue(torch.equal(values, diagonal))
        self.assertTrue(torch.equal(vectors, torch.eye(128, dtype=torch.float64)))

    def test_sst_fit_is_byte_deterministic(self) -> None:
        generator = torch.Generator().manual_seed(9)
        q = torch.randn((12, 4, 128), generator=generator)
        k = torch.randn((128, 128), generator=generator)
        v = torch.randn((128, 128), generator=generator)
        first = scr2.fit_sst_rotation(q, k, v, n_head=4, n_head_kv=2, kv_head=1)
        second = scr2.fit_sst_rotation(q, k, v, n_head=4, n_head_kv=2, kv_head=1)
        self.assertEqual(first.rotation_bytes, second.rotation_bytes)
        self.assertEqual(first.rotation_sha256, hashlib.sha256(first.rotation_bytes).hexdigest())
        self.assertLessEqual(first.orthogonality_max_abs, scr2.ROTATION_ORTHOGONALITY_MAX)
        self.assertEqual(first.weights.shape, (128,))
        self.assertAlmostEqual(float(first.weights.sum()), 128.0, places=9)

    def test_sst_input_rejection(self) -> None:
        q = torch.ones((4, 4, 128))
        k = torch.zeros((128, 128))
        v = torch.ones((128, 128))
        with self.assertRaisesRegex(scr2.SCR2Error, "zero or invalid sum"):
            scr2.fit_sst_rotation(q, k, v, n_head=4, n_head_kv=2, kv_head=0)
        bad_q = q.clone(); bad_q[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(scr2.SCR2Error, "nonfinite"):
            scr2.fit_sst_rotation(bad_q, torch.ones_like(k), v, n_head=4, n_head_kv=2, kv_head=0)
        with self.assertRaisesRegex(scr2.SCR2Error, "divisible"):
            scr2.fit_sst_rotation(q, torch.ones_like(k), v, n_head=4, n_head_kv=3, kv_head=0)


class ContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = records(scr2.PRODUCTION_RECORDS)
        cls.rotation = identity_bytes()
        cls.encoded = scr2.encode_container(cls.source, cls.rotation, head_dim=128)

    def test_deterministic_roundtrip_and_reverse_random_access(self) -> None:
        duplicate = scr2.encode_container(self.source, self.rotation, head_dim=128)
        self.assertEqual(self.encoded, duplicate)
        parsed = scr2.parse_container(self.encoded, expected_head_dim=128)
        reverse = [parsed.decode_record(index) for index in (125, 0)]
        self.assertEqual(tuple(reverse[0].shape), (128, 128))
        self.assertTrue(torch.equal(reverse[0], parsed.decode_record(125)))
        self.assertTrue(torch.equal(reverse[1], parsed.decode_record(0)))
        with self.assertRaises(scr2.SCR2Error):
            parsed.decode_record(126)

    def test_decode_uses_only_container_bytes(self) -> None:
        local_source = records(scr2.PRODUCTION_RECORDS)
        local_encoded = scr2.encode_container(local_source, self.rotation, head_dim=128)
        parsed = scr2.parse_container(local_encoded)
        before = parsed.decode_record(0).clone()
        local_source[0].fill_(999.0)
        self.assertTrue(torch.equal(before, parsed.decode_record(0)))

    def test_code_packing_order(self) -> None:
        row = torch.arange(4, dtype=torch.float32).repeat(32)
        record = row.repeat(128, 1)
        encoded = scr2.encode_container([record] * scr2.PRODUCTION_RECORDS, self.rotation, head_dim=128)
        parsed = scr2.parse_container(encoded)
        payload_start = parsed.record_offsets[0] + 4 * 128
        self.assertEqual(encoded[payload_start], 0xE4)

    def test_constant_group_fails_closed_on_fp16_scale_underflow(self) -> None:
        with self.assertRaisesRegex(scr2.SCR2Error, "nonpositive scale"):
            scr2.encode_container(
                [torch.ones((128, 128))] * scr2.PRODUCTION_RECORDS,
                self.rotation,
                head_dim=128,
            )

    def test_record_shape_and_capacity_rejection(self) -> None:
        with self.assertRaises(scr2.SCR2Error):
            scr2.encode_container([], self.rotation, head_dim=128)
        with self.assertRaises(scr2.SCR2Error):
            scr2.encode_container(
                [torch.ones((127, 128))] + self.source[1:],
                self.rotation,
                head_dim=128,
            )
        with self.assertRaisesRegex(scr2.SCR2Error, "exactly 126"):
            scr2.parse_container(replace_header(self.encoded, 6, 125))

    def test_truncation_trailing_and_sha_rejection(self) -> None:
        for changed in (self.encoded[:-1], self.encoded + b"x"):
            with self.subTest(length=len(changed)), self.assertRaises(scr2.SCR2Error):
                scr2.parse_container(changed)
        changed = bytearray(self.encoded); changed[-1] ^= 1
        with self.assertRaisesRegex(scr2.SCR2Error, "SHA-256"):
            scr2.parse_container(bytes(changed))

    def test_header_offset_and_reserved_rejection(self) -> None:
        with self.assertRaisesRegex(scr2.SCR2Error, "offsets"):
            scr2.parse_container(replace_header(self.encoded, 7, 68))
        with self.assertRaisesRegex(scr2.SCR2Error, "header"):
            scr2.parse_container(replace_header(self.encoded, 14, 1))

    def test_table_index_rejection(self) -> None:
        changed = bytearray(self.encoded)
        fields = scr2._HEADER.unpack_from(changed)
        table_offset = fields[9]
        entry = list(scr2._TABLE_ENTRY.unpack_from(changed, table_offset))
        entry[0] = 7
        scr2._TABLE_ENTRY.pack_into(changed, table_offset, *entry)
        with self.assertRaisesRegex(scr2.SCR2Error, "table entry"):
            scr2.parse_container(resign(changed))

    def test_crc_rejection_with_valid_container_sha(self) -> None:
        changed = bytearray(self.encoded)
        fields = scr2._HEADER.unpack_from(changed)
        changed[fields[11] + 4 * 128] ^= 1
        with self.assertRaisesRegex(scr2.SCR2Error, "CRC32C"):
            scr2.parse_container(resign(changed))

    def _metadata_corruption(self, half_bits: int, scale: bool) -> bytes:
        changed = bytearray(self.encoded)
        fields = scr2._HEADER.unpack_from(changed)
        table_offset, records_offset = fields[9], fields[11]
        metadata_offset = records_offset + (2 if scale else 0)
        struct.pack_into("<H", changed, metadata_offset, half_bits)
        entry = list(scr2._TABLE_ENTRY.unpack_from(changed, table_offset))
        record = bytes(changed[records_offset:records_offset + fields[12]])
        entry[3] = scr2.crc32c(record)
        scr2._TABLE_ENTRY.pack_into(changed, table_offset, *entry)
        return resign(changed)

    def test_nonfinite_and_zero_scale_rejection(self) -> None:
        with self.assertRaisesRegex(scr2.SCR2Error, "nonfinite"):
            scr2.parse_container(self._metadata_corruption(0x7E00, False))
        with self.assertRaisesRegex(scr2.SCR2Error, "nonpositive"):
            scr2.parse_container(self._metadata_corruption(0, True))

    def test_nonfinite_and_nonorthogonal_rotation_rejection(self) -> None:
        fields = scr2._HEADER.unpack_from(self.encoded)
        rotation_offset = fields[7]
        changed = bytearray(self.encoded)
        struct.pack_into("<H", changed, rotation_offset, 0x7E00)
        with self.assertRaisesRegex(scr2.SCR2Error, "nonfinite"):
            scr2.parse_container(resign(changed))
        changed = bytearray(self.encoded)
        struct.pack_into("<e", changed, rotation_offset, 0.0)
        with self.assertRaisesRegex(scr2.SCR2Error, "orthogonality"):
            scr2.parse_container(resign(changed))


if __name__ == "__main__":
    unittest.main()
