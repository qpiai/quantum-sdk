"""Decode JSON-safe quantum result artifacts into typed numeric values."""

import math
import re
from numbers import Integral, Real
from typing import Any

from .exceptions import InvalidResponseError
from .result_contract import (
    ContractValidationError,
    DensityMatrixData,
    StatevectorData,
    decode_density_matrix_contract,
    decode_statevector_contract,
)


def decode_statevector(statevector: object) -> StatevectorData | None:
    """Decode a statevector and return the SDK's canonical column-vector shape."""
    try:
        return decode_statevector_contract(statevector)
    except ContractValidationError as exc:
        raise InvalidResponseError(str(exc)) from exc


def decode_density_matrix(density_matrix: object) -> DensityMatrixData | None:
    """Decode JSON complex entries in a density matrix."""
    try:
        return decode_density_matrix_contract(density_matrix)
    except ContractValidationError as exc:
        raise InvalidResponseError(str(exc)) from exc


def decode_counts(response: object) -> dict[str, int] | None:
    """Extract counts from current and legacy circuit result envelopes."""
    if not isinstance(response, dict):
        return None
    for field in ("counts", "histogram"):
        counts = response.get(field)
        if counts is not None:
            return normalize_counts(counts)
    for key in ("data", "results"):
        value = response.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        counts = decode_counts(value)
        if counts is not None:
            return counts
    return None


def normalize_counts(counts: object) -> dict[str, int]:
    """Validate nonnegative integer counts, including decimal string values."""
    try:
        if not isinstance(counts, dict):
            raise ValueError
        normalized: dict[str, int] = {}
        for outcome, value in counts.items():
            if not isinstance(outcome, str) or isinstance(value, bool):
                raise ValueError
            if isinstance(value, Integral):
                count = int(value)
            elif isinstance(value, Real) and math.isfinite(value):
                count = int(math.trunc(value))
                if value != count:
                    raise ValueError
            elif isinstance(value, str) and re.fullmatch(r"\+?\d+", value.strip()):
                count = int(value)
            else:
                raise ValueError
            if count < 0:
                raise ValueError
            normalized[outcome] = count
        return normalized
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResponseError(
            "The backend returned invalid measurement counts."
        ) from exc


def parse_circuit_response(response: dict[str, Any]) -> dict[str, Any]:
    """Decode circuit results by their envelope, independently of device names."""
    data = response
    layers = [response]
    for _ in range(8):
        nested = data.get("data", data.get("results"))
        if isinstance(nested, list):
            nested = nested[0] if nested else None
        if not isinstance(nested, dict):
            break
        data = nested
        layers.append(data)

    def field(name: str) -> Any:
        for layer in reversed(layers):
            if layer.get(name) is not None:
                return layer[name]
        return None

    return {
        "counts": decode_counts(response),
        "statevector": decode_statevector(field("statevector")),
        "density_matrix": decode_density_matrix(field("density_matrix")),
        "probabilities": field("probabilities"),
    }
