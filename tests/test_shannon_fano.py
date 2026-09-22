import pytest
from algorithms.lossless.shannon_fano import ShannonFanoCodec

def test_empty_input():
    codec = ShannonFanoCodec()
    codec.fit("")
    assert codec.frequencies == {}
    assert codec.codes == {}
    assert codec.encode("") == ""
    assert codec.decode("") == ""

def test_single_symbol_input():
    codec = ShannonFanoCodec()
    codec.fit("AAAAAA")
    assert codec.frequencies == {"A": 6}
    assert codec.codes == {"A": "0"}
    assert codec.encode("AAAAAA") == "000000"
    assert codec.decode("000000") == "AAAAAA"

def test_normal_multi_symbol_input():
    data = "ABABABCCCD"
    codec = ShannonFanoCodec()
    codec.fit(data)
    
    assert len(codec.codes) == 4
    for symbol, code in codec.codes.items():
        assert set(code).issubset({"0", "1"})
        
    assert len(codec.codes.values()) == len(set(codec.codes.values()))
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

def test_determinism():
    data = "ABABABCCCD"
    codec1 = ShannonFanoCodec()
    codec1.fit(data)
    
    codec2 = ShannonFanoCodec()
    codec2.fit(data)
    
    assert codec1.frequencies == codec2.frequencies
    assert codec1.codes == codec2.codes

def test_unknown_symbol():
    codec = ShannonFanoCodec()
    codec.fit("ABCD")
    with pytest.raises(ValueError):
        codec.encode("ABCDE")

def test_decode_invalid_data():
    codec = ShannonFanoCodec()
    # A single symbol gets code '0'.
    # If we pass '01', '0' decodes to 'A', and '1' is left in the buffer.
    # Since '1' doesn't map to anything and is trailing, decode() should raise ValueError.
    codec.fit("A")
    with pytest.raises(ValueError):
        codec.decode("01")

def test_round_trip_larger_text():
    text = "Hello, World! This is a test of the Shannon-Fano lossless compression algorithm."
    codec = ShannonFanoCodec()
    codec.fit(text)
    
    encoded = codec.encode(text)
    decoded = codec.decode(encoded)
    
    assert decoded == text

def test_code_validity():
    data = "Some multi-symbol string with various characters."
    codec = ShannonFanoCodec()
    codec.fit(data)
    
    for symbol, code in codec.codes.items():
        assert len(code) > 0
        assert set(code).issubset({"0", "1"})
        
    codes_list = list(codec.codes.values())
    assert len(codes_list) == len(set(codes_list))
