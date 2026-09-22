import pytest
from algorithms.lossless.rice import RiceCodec
from algorithms.lossless.golomb import GolombCodec

def test_empty_input():
    codec = RiceCodec(2)
    assert codec.encode([]) == ""
    assert codec.decode("") == []

def test_k_validation():
    with pytest.raises(TypeError):
        RiceCodec(True)
    with pytest.raises(TypeError):
        RiceCodec("2")
    with pytest.raises(TypeError):
        RiceCodec(2.5)
    with pytest.raises(ValueError):
        RiceCodec(-1)

def test_data_validation():
    codec = RiceCodec(1)
    with pytest.raises(TypeError):
        codec.encode([True])
    with pytest.raises(TypeError):
        codec.encode([1, "2"])
    with pytest.raises(ValueError):
        codec.encode([1, -1, 2])

def test_decode_validation():
    codec = RiceCodec(1)
    with pytest.raises(TypeError):
        codec.decode(["10"])

def test_invalid_characters():
    codec = RiceCodec(1)
    with pytest.raises(ValueError, match="Invalid character"):
        codec.decode("001a0")
    with pytest.raises(ValueError, match="Invalid character"):
        codec.decode("12")

def test_incomplete_unary():
    codec = RiceCodec(2)
    with pytest.raises(ValueError, match="Incomplete unary quotient"):
        codec.decode("00")

def test_incomplete_remainder():
    codec = RiceCodec(2)
    with pytest.raises(ValueError, match="Incomplete remainder"):
        codec.decode("10") # k=2, only 1 bit provided

@pytest.mark.parametrize("k, data, expected_bits", [
    (0, [0, 1, 2, 3], "1" + "01" + "001" + "0001"),
    (1, [0, 1, 2, 3, 4, 5], "10" + "11" + "010" + "011" + "0010" + "0011"),
    (2, [0, 1, 2, 3, 4, 5, 6, 7], "100" + "101" + "110" + "111" + "0100" + "0101" + "0110" + "0111"),
    (3, [0, 1, 7, 8, 15, 16, 31, 32], "1000" + "1001" + "1111" + "01000" + "01111" + "001000" + "0001111" + "00001000")
])
def test_hardcoded_examples(k, data, expected_bits):
    codec = RiceCodec(k)
    assert codec.encode(data) == expected_bits
    assert codec.decode(expected_bits) == data

@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 8, 16])
def test_roundtrip_large_values(k):
    codec = RiceCodec(k)
    data = [0, 10, 100, 1000, 10000, 100000]
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_cross_validation_with_golomb(k):
    rice_codec = RiceCodec(k)
    golomb_codec = GolombCodec(2**k)
    
    data = list(range(100)) + [1000, 2000, 5000]
    
    rice_encoded = rice_codec.encode(data)
    golomb_encoded = golomb_codec.encode(data)
    
    assert rice_encoded == golomb_encoded
    assert rice_codec.decode(golomb_encoded) == data
    assert golomb_codec.decode(rice_encoded) == data
