#!/usr/bin/env python3
"""Deterministic byte-real CPU reference for the KVarN SCR2 value codec."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Sequence

import numpy as np
import torch


MAGIC = b"KVNSCR2\0"
VERSION = 1
HEADER_BYTES = 64
TRAILER_BYTES = 32
TABLE_ENTRY_BYTES = 16
RECORD_TOKENS = 128
QUANT_GROUP_FEATURES = 128
PRODUCTION_CONTEXT_TOKENS = 16384
PRODUCTION_SINK_TOKENS = 128
PRODUCTION_TAIL_TOKENS = 128
PRODUCTION_RECORDS = 126
ROTATION_ORTHOGONALITY_MAX = 5.0e-3

_HEADER = struct.Struct("<8sHHHHIIIIIIIIIIQ")
_TABLE_ENTRY = struct.Struct("<IIII")


class SCR2Error(ValueError):
    pass


@dataclass(frozen=True)
class SCR2Accounting:
    head_dim: int
    n_records: int
    rotation_bytes: int
    table_bytes: int
    record_stride: int
    total_bytes: int
    represented_values: int
    allocated_bits_per_value: float
    per_record_charged_bits_per_value: float
    full_window_effective_bits_per_value: float


@dataclass(frozen=True)
class SCR2Fit:
    query_covariance: torch.Tensor
    value_covariance: torch.Tensor
    weights: torch.Tensor
    eigenvalues: torch.Tensor
    canonical_eigenvectors: torch.Tensor
    hadamard: torch.Tensor
    permutation: torch.Tensor
    rotation_f64: torch.Tensor
    rotation_bytes: bytes
    rotation_sha256: str
    orthogonality_max_abs: float


def _validate_head_dim(head_dim: int) -> None:
    if head_dim < QUANT_GROUP_FEATURES or head_dim % QUANT_GROUP_FEATURES:
        raise SCR2Error("head_dim must be a positive multiple of quant_group_features")
    if head_dim & (head_dim - 1):
        raise SCR2Error("head_dim must be a power of two")
    if head_dim > 0xFFFF:
        raise SCR2Error("head_dim does not fit the container header")


def accounting(head_dim: int, n_records: int = 126) -> SCR2Accounting:
    _validate_head_dim(head_dim)
    if not 0 < n_records <= 0xFFFFFFFF:
        raise SCR2Error("n_records must be positive and fit uint32")
    rotation_bytes = 2 * head_dim * head_dim
    table_bytes = TABLE_ENTRY_BYTES * n_records
    record_stride = 36 * head_dim
    total_bytes = HEADER_BYTES + rotation_bytes + table_bytes + n_records * record_stride + TRAILER_BYTES
    if total_bytes > 0xFFFFFFFF:
        raise SCR2Error("container does not fit the uint32 offset and length fields")
    represented_values = n_records * RECORD_TOKENS * head_dim
    rate = total_bytes * 8.0 / represented_values
    protected_bytes = (
        (PRODUCTION_SINK_TOKENS + PRODUCTION_TAIL_TOKENS) * head_dim * 2
    )
    full_window_rate = (
        (total_bytes + protected_bytes) * 8.0
        / (PRODUCTION_CONTEXT_TOKENS * head_dim)
    )
    return SCR2Accounting(
        head_dim=head_dim,
        n_records=n_records,
        rotation_bytes=rotation_bytes,
        table_bytes=table_bytes,
        record_stride=record_stride,
        total_bytes=total_bytes,
        represented_values=represented_values,
        allocated_bits_per_value=rate,
        per_record_charged_bits_per_value=rate,
        full_window_effective_bits_per_value=full_window_rate,
    )


def build_hadamard(size: int) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise SCR2Error("Hadamard size must be a power of two")
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
        h /= math.sqrt(2.0)
    return h


def bit_reversal_permutation(size: int) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise SCR2Error("bit-reversal size must be a power of two")
    bits = int(math.log2(size))
    return torch.tensor(
        [int(f"{index:0{bits}b}"[::-1], 2) for index in range(size)],
        dtype=torch.int64,
    )


def _sign_canonicalize(vector: torch.Tensor) -> torch.Tensor:
    absolute = vector.abs()
    maximum = absolute.max()
    anchors = torch.nonzero(absolute == maximum, as_tuple=False).flatten()
    if anchors.numel() == 0 or maximum == 0:
        raise SCR2Error("cannot sign-canonicalize a zero eigenvector")
    anchor = int(anchors[0])
    return -vector if vector[anchor] < 0 else vector


def _eigen_blocks(eigenvalues: torch.Tensor) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start = 0
    for index in range(1, eigenvalues.numel()):
        left = float(eigenvalues[index - 1])
        right = float(eigenvalues[index])
        tolerance = 1.0e-10 * max(1.0, abs(left), abs(right))
        if abs(right - left) > tolerance:
            blocks.append((start, index))
            start = index
    blocks.append((start, eigenvalues.numel()))
    return blocks


def canonical_eigendecomposition(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if covariance.device.type != "cpu" or covariance.dtype != torch.float64:
        raise SCR2Error("covariance must be a CPU float64 tensor")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise SCR2Error("covariance must be square")
    if not bool(torch.isfinite(covariance).all()):
        raise SCR2Error("covariance contains nonfinite values")
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance, UPLO="L")
    canonical = torch.empty_like(eigenvectors)
    for start, end in _eigen_blocks(eigenvalues):
        width = end - start
        if width == 1:
            canonical[:, start] = _sign_canonicalize(eigenvectors[:, start])
            continue
        basis = eigenvectors[:, start:end]
        projector = basis @ basis.T
        accepted: list[torch.Tensor] = []
        for coordinate in range(projector.shape[0]):
            candidate = projector[:, coordinate].clone()
            for _ in range(2):
                for prior in accepted:
                    candidate -= torch.dot(prior, candidate) * prior
            norm = torch.linalg.vector_norm(candidate)
            if float(norm) <= 1.0e-12:
                continue
            accepted.append(_sign_canonicalize(candidate / norm))
            if len(accepted) == width:
                break
        if len(accepted) != width:
            raise SCR2Error("failed to construct a canonical degenerate eigenspace basis")
        canonical[:, start:end] = torch.stack(accepted, dim=1)
    return eigenvalues.contiguous(), canonical.contiguous()


def _rotation_from_covariance(covariance: torch.Tensor) -> tuple[torch.Tensor, ...]:
    eigenvalues, canonical = canonical_eigendecomposition(covariance)
    head_dim = covariance.shape[0]
    _validate_head_dim(head_dim)
    hadamard = build_hadamard(head_dim)
    descending = torch.argsort(eigenvalues, descending=True, stable=True)
    bit_reversal = bit_reversal_permutation(head_dim)
    perm = torch.zeros(head_dim, dtype=torch.int64)
    for index in range(head_dim):
        perm[int(bit_reversal[index])] = descending[index]
    permutation = torch.eye(head_dim, dtype=torch.float64)[:, perm]
    rotation = (canonical @ hadamard @ permutation).contiguous()
    return eigenvalues, canonical, hadamard, permutation, rotation


def rotation_to_bytes(rotation: torch.Tensor) -> bytes:
    if rotation.device.type != "cpu" or rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise SCR2Error("rotation must be a square CPU tensor")
    _validate_head_dim(rotation.shape[0])
    if not bool(torch.isfinite(rotation).all()):
        raise SCR2Error("rotation contains nonfinite values")
    return np.asarray(rotation.detach().to(torch.float64).numpy(), dtype="<f2").tobytes(order="C")


def rotation_from_bytes(data: bytes, head_dim: int) -> torch.Tensor:
    _validate_head_dim(head_dim)
    expected = 2 * head_dim * head_dim
    if len(data) != expected:
        raise SCR2Error("rotation byte length mismatch")
    array = np.frombuffer(data, dtype="<f2").astype(np.float32).reshape(head_dim, head_dim).copy()
    if not np.isfinite(array).all():
        raise SCR2Error("serialized rotation contains nonfinite values")
    return torch.from_numpy(array)


def _require_orthogonal_rotation(rotation: torch.Tensor) -> None:
    identity = torch.eye(rotation.shape[0], dtype=torch.float64)
    rotation64 = rotation.to(torch.float64)
    error = float((rotation64.T @ rotation64 - identity).abs().max())
    if not math.isfinite(error) or error > ROTATION_ORTHOGONALITY_MAX:
        raise SCR2Error("serialized rotation exceeds the orthogonality limit")


def fit_sst_rotation(
    full_q_body: torch.Tensor,
    selected_k: torch.Tensor,
    selected_v: torch.Tensor,
    *,
    n_head: int,
    n_head_kv: int,
    kv_head: int,
) -> SCR2Fit:
    torch.set_num_threads(1)
    tensors = (full_q_body, selected_k, selected_v)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise SCR2Error("SST fitting is CPU-only")
    if full_q_body.ndim != 3 or selected_k.ndim != 2 or selected_v.ndim != 2:
        raise SCR2Error("invalid Q/K/V ranks")
    if selected_k.shape != selected_v.shape:
        raise SCR2Error("selected K/V shapes differ")
    if n_head <= 0 or n_head_kv <= 0 or n_head % n_head_kv:
        raise SCR2Error("n_head must be divisible by n_head_kv")
    if full_q_body.shape[1] != n_head:
        raise SCR2Error("full Q head count mismatch")
    if not 0 <= kv_head < n_head_kv:
        raise SCR2Error("kv_head is out of range")
    head_dim = selected_k.shape[1]
    _validate_head_dim(head_dim)
    if full_q_body.shape[2] != head_dim or selected_k.shape[0] == 0 or full_q_body.shape[0] == 0:
        raise SCR2Error("invalid Q/K/V dimensions")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise SCR2Error("Q/K/V contains nonfinite values")

    gqa = n_head // n_head_kv
    first = kv_head * gqa
    query_group = full_q_body[:, first:first + gqa, :].reshape(-1, head_dim).to(torch.float64)
    key = selected_k.to(torch.float64)
    value = selected_v.to(torch.float64)
    query_covariance = (query_group.T @ query_group) / query_group.shape[0]
    query_covariance = (query_covariance + query_covariance.T) / 2.0
    weights = (key @ query_covariance * key).sum(dim=1)
    if not bool(torch.isfinite(weights).all()):
        raise SCR2Error("SST weights contain nonfinite values")
    maximum = float(weights.abs().max())
    epsilon = 1.0e-12 * max(1.0, maximum)
    if bool((weights < -epsilon).any()):
        raise SCR2Error("SST weights contain a materially negative value")
    weights = torch.where(weights < 0, torch.zeros_like(weights), weights)
    weight_sum = float(weights.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 1.0e-12:
        raise SCR2Error("SST weights have a zero or invalid sum")
    weights = weights / weight_sum * selected_k.shape[0]
    weighted_value = value * weights.sqrt().unsqueeze(1)
    value_covariance = (weighted_value.T @ weighted_value) / selected_k.shape[0]
    value_covariance = (value_covariance + value_covariance.T) / 2.0

    first_fit = _rotation_from_covariance(value_covariance)
    second_fit = _rotation_from_covariance(value_covariance)
    first_bytes = rotation_to_bytes(first_fit[-1])
    second_bytes = rotation_to_bytes(second_fit[-1])
    if first_bytes != second_bytes:
        raise SCR2Error("repeated fitting did not produce byte-identical rotation data")
    roundtrip = rotation_from_bytes(first_bytes, head_dim).to(torch.float64)
    identity = torch.eye(head_dim, dtype=torch.float64)
    orthogonality = float((roundtrip.T @ roundtrip - identity).abs().max())
    if not math.isfinite(orthogonality) or orthogonality > ROTATION_ORTHOGONALITY_MAX:
        raise SCR2Error("fp16 rotation roundtrip exceeds the orthogonality limit")
    eigenvalues, canonical, hadamard, permutation, rotation = first_fit
    return SCR2Fit(
        query_covariance=query_covariance,
        value_covariance=value_covariance,
        weights=weights,
        eigenvalues=eigenvalues,
        canonical_eigenvectors=canonical,
        hadamard=hadamard,
        permutation=permutation,
        rotation_f64=rotation,
        rotation_bytes=first_bytes,
        rotation_sha256=hashlib.sha256(first_bytes).hexdigest(),
        orthogonality_max_abs=orthogonality,
    )


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _encode_record(values: torch.Tensor, rotation: torch.Tensor, head_dim: int) -> bytes:
    if values.device.type != "cpu" or values.shape != (RECORD_TOKENS, head_dim):
        raise SCR2Error("each record must be a CPU [128, head_dim] tensor")
    if not bool(torch.isfinite(values).all()):
        raise SCR2Error("record contains nonfinite values")
    rotated = values.to(torch.float32) @ rotation
    groups = rotated.reshape(RECORD_TOKENS, head_dim // QUANT_GROUP_FEATURES, QUANT_GROUP_FEATURES)
    minimum = groups.amin(dim=-1)
    maximum = groups.amax(dim=-1)
    scale = torch.clamp(maximum - minimum, min=1.0e-8) / 3.0
    zero = -minimum / scale
    codes = (groups / scale.unsqueeze(-1) + zero.unsqueeze(-1) + 0.5).to(torch.int32).clamp(0, 3)

    metadata = np.empty((RECORD_TOKENS, head_dim // QUANT_GROUP_FEATURES, 2), dtype="<f2")
    metadata[..., 0] = minimum.detach().numpy().astype("<f2")
    metadata[..., 1] = scale.detach().numpy().astype("<f2")
    metadata_f32 = metadata.astype(np.float32)
    if not np.isfinite(metadata_f32).all() or np.any(metadata_f32[..., 1] <= 0):
        raise SCR2Error("fp16 metadata is nonfinite or has a nonpositive scale")

    flat = codes.reshape(-1).detach().numpy().astype(np.uint8)
    packed = flat.reshape(-1, 4)
    packed = packed[:, 0] | (packed[:, 1] << 2) | (packed[:, 2] << 4) | (packed[:, 3] << 6)
    result = metadata.tobytes(order="C") + packed.astype(np.uint8).tobytes(order="C")
    if len(result) != 36 * head_dim:
        raise AssertionError("internal SCR2 record size mismatch")
    return result


def encode_container(records: Sequence[torch.Tensor], rotation_bytes: bytes, *, head_dim: int) -> bytes:
    _validate_head_dim(head_dim)
    n_records = len(records)
    if n_records != PRODUCTION_RECORDS:
        raise SCR2Error("production SCR2 requires exactly 126 complete records")
    layout = accounting(head_dim, n_records)
    rotation = rotation_from_bytes(rotation_bytes, head_dim)
    _require_orthogonal_rotation(rotation)
    encoded = [_encode_record(record, rotation, head_dim) for record in records]
    rotation_offset = HEADER_BYTES
    table_offset = rotation_offset + len(rotation_bytes)
    records_offset = table_offset + n_records * TABLE_ENTRY_BYTES
    total_bytes = records_offset + n_records * layout.record_stride + TRAILER_BYTES
    header = _HEADER.pack(
        MAGIC, VERSION, 0, head_dim, QUANT_GROUP_FEATURES,
        RECORD_TOKENS, n_records, rotation_offset, len(rotation_bytes),
        table_offset, TABLE_ENTRY_BYTES, records_offset, layout.record_stride,
        total_bytes, 0, 0,
    )
    table = bytearray()
    for index, record in enumerate(encoded):
        table.extend(_TABLE_ENTRY.pack(index, records_offset + index * layout.record_stride, len(record), crc32c(record)))
    body = header + rotation_bytes + bytes(table) + b"".join(encoded)
    if len(body) + TRAILER_BYTES != total_bytes:
        raise AssertionError("internal SCR2 container size mismatch")
    return body + hashlib.sha256(body).digest()


@dataclass(frozen=True)
class SCR2Container:
    data: bytes
    head_dim: int
    n_records: int
    rotation: torch.Tensor
    record_offsets: tuple[int, ...]
    record_stride: int

    def decode_record(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.n_records:
            raise SCR2Error("record index is out of range")
        start = self.record_offsets[index]
        record = self.data[start:start + self.record_stride]
        metadata_bytes = 4 * self.head_dim
        metadata = np.frombuffer(record[:metadata_bytes], dtype="<f2").astype(np.float32)
        metadata = metadata.reshape(RECORD_TOKENS, self.head_dim // QUANT_GROUP_FEATURES, 2)
        minimum = torch.from_numpy(metadata[..., 0].copy())
        scale = torch.from_numpy(metadata[..., 1].copy())
        packed = np.frombuffer(record[metadata_bytes:], dtype=np.uint8)
        codes = np.empty(packed.size * 4, dtype=np.uint8)
        codes[0::4] = packed & 0x03
        codes[1::4] = (packed >> 2) & 0x03
        codes[2::4] = (packed >> 4) & 0x03
        codes[3::4] = (packed >> 6) & 0x03
        q = torch.from_numpy(codes.astype(np.float32).reshape(RECORD_TOKENS, self.head_dim // QUANT_GROUP_FEATURES, QUANT_GROUP_FEATURES))
        dequantized = minimum.unsqueeze(-1) + q * scale.unsqueeze(-1)
        return dequantized.reshape(RECORD_TOKENS, self.head_dim) @ self.rotation.T


def parse_container(
    data: bytes,
    *,
    expected_head_dim: int | None = None,
) -> SCR2Container:
    if len(data) < HEADER_BYTES + TRAILER_BYTES:
        raise SCR2Error("container is truncated")
    fields = _HEADER.unpack_from(data)
    (magic, version, flags, head_dim, quant_group_features, record_tokens, n_records,
     rotation_offset, rotation_bytes, table_offset, table_entry_bytes, records_offset,
     record_stride, total_bytes, reserved0, reserved1) = fields
    if magic != MAGIC or version != VERSION or flags != 0 or reserved0 != 0 or reserved1 != 0:
        raise SCR2Error("invalid or noncanonical SCR2 header")
    _validate_head_dim(head_dim)
    if quant_group_features != QUANT_GROUP_FEATURES or record_tokens != RECORD_TOKENS:
        raise SCR2Error("unsupported SCR2 geometry")
    if expected_head_dim is not None and head_dim != expected_head_dim:
        raise SCR2Error("head_dim does not match registration")
    if n_records != PRODUCTION_RECORDS:
        raise SCR2Error("production SCR2 requires exactly 126 complete records")
    layout = accounting(head_dim, n_records)
    expected_rotation_bytes = 2 * head_dim * head_dim
    expected_table_offset = HEADER_BYTES + expected_rotation_bytes
    expected_records_offset = expected_table_offset + n_records * TABLE_ENTRY_BYTES
    if (
        rotation_offset != HEADER_BYTES or rotation_bytes != expected_rotation_bytes
        or table_offset != expected_table_offset or table_entry_bytes != TABLE_ENTRY_BYTES
        or records_offset != expected_records_offset or record_stride != layout.record_stride
        or total_bytes != layout.total_bytes or len(data) != total_bytes
    ):
        raise SCR2Error("container offsets, lengths, or total size are noncanonical")
    if hashlib.sha256(data[:-TRAILER_BYTES]).digest() != data[-TRAILER_BYTES:]:
        raise SCR2Error("container SHA-256 mismatch")
    rotation_data = data[rotation_offset:table_offset]
    rotation = rotation_from_bytes(rotation_data, head_dim)
    _require_orthogonal_rotation(rotation)
    offsets: list[int] = []
    for index in range(n_records):
        entry = _TABLE_ENTRY.unpack_from(data, table_offset + index * TABLE_ENTRY_BYTES)
        record_index, offset, length, checksum = entry
        expected_offset = records_offset + index * record_stride
        if record_index != index or offset != expected_offset or length != record_stride:
            raise SCR2Error("record table entry is noncanonical")
        record = data[offset:offset + length]
        if len(record) != length or crc32c(record) != checksum:
            raise SCR2Error("record is truncated or fails CRC32C")
        metadata = np.frombuffer(record[:4 * head_dim], dtype="<f2").astype(np.float32)
        metadata = metadata.reshape(RECORD_TOKENS, head_dim // QUANT_GROUP_FEATURES, 2)
        if not np.isfinite(metadata).all() or np.any(metadata[..., 1] <= 0):
            raise SCR2Error("record metadata is nonfinite or has a nonpositive scale")
        offsets.append(offset)
    return SCR2Container(data, head_dim, n_records, rotation, tuple(offsets), record_stride)


def environment_identity() -> dict[str, object]:
    torch_build = torch.__config__.show().encode("utf-8")
    return {
        "python": __import__("sys").version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_build_sha256": hashlib.sha256(torch_build).hexdigest(),
        "device": "cpu",
        "fit_dtype": "float64",
        "rotation_storage": "little-endian-fp16",
    }
