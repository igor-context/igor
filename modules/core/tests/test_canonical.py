import pytest

from igor_core import canonical_json, stable_identity


def test_canonical_json_sorts_nested_mappings() -> None:
    left = {"b": 2, "a": {"z": 1, "x": "é"}}
    right = {"a": {"x": "é", "z": 1}, "b": 2}

    assert canonical_json(left) == '{"a":{"x":"é","z":1},"b":2}'
    assert stable_identity(left) == stable_identity(right)


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})
