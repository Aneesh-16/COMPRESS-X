import math
import pytest
from analysis.redundancy import calculate_redundancy

def test_empty_input():
    result = calculate_redundancy("")
    assert result["entropy"] == 0.0
    assert result["maximum_entropy"] == 0.0
    assert result["redundancy"] == 0.0
    assert result["total_symbols"] == 0
    assert result["unique_symbols"] == 0

def test_single_repeated_symbol():
    result = calculate_redundancy("AAAA")
    assert result["entropy"] == 0.0
    assert result["maximum_entropy"] == 0.0
    assert result["redundancy"] == 0.0
    assert result["total_symbols"] == 4
    assert result["unique_symbols"] == 1

def test_two_equally_probable_symbols():
    result = calculate_redundancy("ABAB")
    assert result["entropy"] == pytest.approx(1.0)
    assert result["maximum_entropy"] == pytest.approx(1.0)
    assert result["redundancy"] == pytest.approx(0.0)

def test_three_equally_probable_symbols():
    result = calculate_redundancy("AAAABBBBCCCC")
    assert result["entropy"] == pytest.approx(math.log2(3))
    assert result["maximum_entropy"] == pytest.approx(math.log2(3))
    assert result["redundancy"] == pytest.approx(0.0)

def test_non_uniform_distribution():
    result = calculate_redundancy("AAAAABBC")
    assert result["entropy"] > 0
    assert result["maximum_entropy"] == pytest.approx(math.log2(3))
    assert result["redundancy"] > 0
    assert result["redundancy"] < 1

def test_bytes_input():
    result = calculate_redundancy(b"AAAABBBBCCCC")
    assert result["entropy"] == pytest.approx(math.log2(3))
    assert result["redundancy"] == pytest.approx(0.0)
