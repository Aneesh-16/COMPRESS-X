import math
import pytest
from analysis.entropy import calculate_entropy

def test_empty_input():
    result = calculate_entropy("")
    assert result["entropy"] == 0.0
    assert result["total_symbols"] == 0
    assert result["unique_symbols"] == 0
    assert result["frequencies"] == {}
    assert result["probabilities"] == {}

def test_single_repeated_symbol():
    result = calculate_entropy("AAAA")
    assert result["entropy"] == 0.0
    assert result["total_symbols"] == 4
    assert result["unique_symbols"] == 1
    assert result["frequencies"] == {"A": 4}
    assert result["probabilities"] == {"A": 1.0}

def test_two_equally_probable_symbols():
    result = calculate_entropy("ABAB")
    assert result["entropy"] == 1.0
    assert result["total_symbols"] == 4
    assert result["unique_symbols"] == 2
    assert result["frequencies"] == {"A": 2, "B": 2}
    assert result["probabilities"] == {"A": 0.5, "B": 0.5}

def test_three_equally_probable_symbols():
    result = calculate_entropy("AAAABBBBCCCC")
    assert result["entropy"] == pytest.approx(math.log2(3))
    assert result["total_symbols"] == 12
    assert result["unique_symbols"] == 3

def test_alphabet():
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = calculate_entropy(alphabet)
    assert result["entropy"] == pytest.approx(math.log2(26))
    assert result["total_symbols"] == 26
    assert result["unique_symbols"] == 26

def test_bytes_input():
    result = calculate_entropy(b"AAAABBBBCCCC")
    assert result["entropy"] == pytest.approx(math.log2(3))
    assert result["total_symbols"] == 12
    assert result["unique_symbols"] == 3
    # Check frequencies and probabilities for bytes input (65='A', 66='B', 67='C')
    assert result["frequencies"] == {65: 4, 66: 4, 67: 4}
    assert result["probabilities"] == {
        65: pytest.approx(1/3), 
        66: pytest.approx(1/3), 
        67: pytest.approx(1/3)
    }
