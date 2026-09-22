import pytest
from algorithms.lossless.golomb import GolombCodec

# 1. BASIC ROUND TRIPS
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 8, 10])
@pytest.mark.parametrize("data", [
    [0],
    [0, 1, 2, 3, 4],
    [5, 10, 15, 20],
    [0, 0, 0, 1, 1, 2, 10],
    [100, 101, 102, 1000]
])
def test_golomb_basic_round_trips(m, data):
    codec = GolombCodec(m)
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

# 2. EMPTY INPUT
def test_empty_input():
    assert GolombCodec(5).encode([]) == ""
    assert GolombCodec(5).decode("") == []

# 3. m = 1
def test_m_1():
    codec = GolombCodec(1)
    data = [0, 1, 2, 3, 4]
    encoded = codec.encode(data)
    # Expected: "1" + "01" + "001" + "0001" + "00001"
    assert encoded == "101001000100001"
    assert codec.decode(encoded) == data

# 4. POWER-OF-TWO PARAMETERS
@pytest.mark.parametrize("m", [2, 4, 8, 16])
def test_power_of_two_parameters(m):
    codec = GolombCodec(m)
    data = list(range(20))
    assert codec.decode(codec.encode(data)) == data

# 5. NON-POWER-OF-TWO PARAMETERS
@pytest.mark.parametrize("m", [3, 5, 6, 7, 10])
def test_non_power_of_two_parameters(m):
    codec = GolombCodec(m)
    data = list(range(30))
    assert codec.decode(codec.encode(data)) == data

# 6. KNOWN ENCODING CASES
def test_known_encoding_cases():
    # m = 1
    c1 = GolombCodec(1)
    assert c1.encode([0]) == "1"
    assert c1.encode([1]) == "01"
    assert c1.encode([2]) == "001"
    assert c1.encode([3]) == "0001"

    # m = 2
    c2 = GolombCodec(2)
    assert c2.encode([0]) == "10"
    assert c2.encode([1]) == "11"
    assert c2.encode([2]) == "010"
    assert c2.encode([3]) == "011"

    # m = 3
    c3 = GolombCodec(3)
    # cutoff = 1, b = 2
    # 0 -> r=0 < 1 => "0"
    # 1 -> r=1 >= 1 => "10"
    # 2 -> r=2 >= 1 => "11"
    assert c3.encode([0]) == "10"
    assert c3.encode([1]) == "110"
    assert c3.encode([2]) == "111"
    assert c3.encode([3]) == "010"
    assert c3.encode([4]) == "0110"
    assert c3.encode([5]) == "0111"

# 7. LARGE VALUES
@pytest.mark.parametrize("m", [1, 2, 5, 100])
def test_large_values(m):
    codec = GolombCodec(m)
    data = [0, 1, 10, 100, 1000, 10000, 100000]
    assert codec.decode(codec.encode(data)) == data

# 8. REPEATED VALUES
def test_repeated_values():
    codec = GolombCodec(4)
    data = [7] * 100
    assert codec.decode(codec.encode(data)) == data
    data2 = [0, 1, 0, 1] * 50
    assert codec.decode(codec.encode(data2)) == data2

# 9. DETERMINISTIC ENCODING
def test_deterministic_encoding():
    data = [0, 1, 2, 10, 10, 3, 5]
    codec1 = GolombCodec(5)
    codec2 = GolombCodec(5)
    assert codec1.encode(data) == codec2.encode(data)

# 10. INVALID m VALUES
def test_invalid_m_values():
    for invalid_m in [True, False, 3.5, "5", None]:
        with pytest.raises(TypeError):
            GolombCodec(invalid_m)
    for invalid_m in [0, -1, -10]:
        with pytest.raises(ValueError):
            GolombCodec(invalid_m)

# 11. INVALID INPUT VALUES
def test_invalid_input_values():
    codec = GolombCodec(5)
    for invalid_data in [[True], [False], [1, 2.5], ["1"], [None]]:
        with pytest.raises(TypeError):
            codec.encode(invalid_data)
    for invalid_data in [[-1], [1, -5]]:
        with pytest.raises(ValueError):
            codec.encode(invalid_data)

# 12. INVALID ENCODED DATA TYPE
def test_invalid_encoded_data_type():
    codec = GolombCodec(5)
    for invalid_encoded in [None, 123, ["0", "1"]]:
        with pytest.raises(TypeError):
            codec.decode(invalid_encoded)

# 13. MALFORMED BITSTREAMS
def test_malformed_bitstreams():
    # m = 1
    c1 = GolombCodec(1)
    with pytest.raises(ValueError, match="Incomplete unary"):
        c1.decode("0")

    # m = 3, cutoff = 1, b = 2
    c3 = GolombCodec(3)
    # quotient incomplete
    with pytest.raises(ValueError, match="Incomplete unary"):
        c3.decode("0")
    # remainder missing first bit
    with pytest.raises(ValueError, match="Incomplete remainder"):
        c3.decode("1") # expects 1 bit of remainder (b-1)
    # remainder missing second bit (value >= cutoff)
    with pytest.raises(ValueError, match="Incomplete remainder"):
        c3.decode("11") # expects 1 more bit since r_temp=1 >= cutoff=1

    # m = 5, cutoff = 3, b = 3
    c5 = GolombCodec(5)
    # r_temp takes 2 bits (b-1).
    with pytest.raises(ValueError, match="Incomplete remainder"):
        c5.decode("10") # missing second bit
    with pytest.raises(ValueError, match="Incomplete remainder"):
        c5.decode("111") # r_temp=3 >= 3, expects 3rd bit, but hits EOF

# 14. INVALID CHARACTERS
def test_invalid_characters():
    codec = GolombCodec(5)
    with pytest.raises(ValueError):
        codec.decode("X")
    with pytest.raises(ValueError):
        codec.decode("10102")
    with pytest.raises(ValueError):
        codec.decode("abc")

# 15. ITERABLE SUPPORT
def test_iterable_support():
    codec = GolombCodec(3)
    # tuple
    tup = (0, 1, 2, 3)
    assert codec.decode(codec.encode(tup)) == list(tup)
    # generator
    gen = (x for x in [0, 1, 2, 3])
    assert codec.decode(codec.encode(gen)) == [0, 1, 2, 3]

# 16. IMMUTABILITY / INPUT SAFETY
def test_input_immutability():
    codec = GolombCodec(5)
    data = [0, 1, 2, 3]
    data_copy = data[:]
    codec.encode(data)
    assert data == data_copy

# 17. INTERNAL PARAMETER VALUES
def test_internal_parameter_values():
    expected = {
        1: (0, 0),
        2: (1, 0),
        3: (2, 1),
        5: (3, 3),
        6: (3, 2),
        8: (3, 0)
    }
    for m, (exp_b, exp_cutoff) in expected.items():
        codec = GolombCodec(m)
        assert codec.b == exp_b
        assert codec.cutoff == exp_cutoff
