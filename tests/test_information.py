import math
import pytest
from analysis.information import calculate_information_value

def test_probability_one():
    assert calculate_information_value(1.0) == 0.0

def test_probability_half():
    assert calculate_information_value(0.5) == pytest.approx(1.0)

def test_probability_quarter():
    assert calculate_information_value(0.25) == pytest.approx(2.0)

def test_probability_eighth():
    assert calculate_information_value(0.125) == pytest.approx(3.0)

def test_probability_small():
    assert calculate_information_value(0.01) == pytest.approx(-math.log2(0.01))

def test_probability_zero():
    with pytest.raises(ValueError):
        calculate_information_value(0.0)

def test_negative_probability():
    with pytest.raises(ValueError):
        calculate_information_value(-0.5)

def test_probability_greater_than_one():
    with pytest.raises(ValueError):
        calculate_information_value(1.5)
