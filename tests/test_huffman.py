import pytest
from algorithms.lossless.huffman import HuffmanCodec

def test_empty_input():
    codec = HuffmanCodec()
    codec.fit("")
    assert codec.frequencies == {}
    assert codec.codes == {}
    assert codec.tree is None
    assert codec.encode("") == ""
    assert codec.decode("") == ""

def test_single_symbol_input():
    codec = HuffmanCodec()
    codec.fit("AAAAAA")
    assert codec.frequencies == {"A": 6}
    assert codec.codes == {"A": "0"}
    assert codec.encode("AAAAAA") == "000000"
    assert codec.decode("000000") == "AAAAAA"
    
    with pytest.raises(ValueError):
        codec.decode("000100")

def test_normal_multi_symbol_input():
    data = "ABABABCCCD"
    codec = HuffmanCodec()
    codec.fit(data)
    
    assert len(codec.codes) == 4
    for symbol, code in codec.codes.items():
        assert len(code) > 0
        assert set(code).issubset({"0", "1"})
        
    codes_list = list(codec.codes.values())
    assert len(codes_list) == len(set(codes_list))
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

def test_determinism():
    data = "ABCD" * 5 # equal frequencies
    
    codec1 = HuffmanCodec()
    codec1.fit(data)
    
    codec2 = HuffmanCodec()
    codec2.fit(data)
    
    assert codec1.frequencies == codec2.frequencies
    assert codec1.codes == codec2.codes
    
    # Re-fit the same instance
    codec1.fit(data)
    assert codec1.codes == codec2.codes

def test_unknown_symbol():
    codec = HuffmanCodec()
    codec.fit("ABCD")
    with pytest.raises(ValueError):
        codec.encode("ABE")

def test_invalid_encoded_data():
    codec = HuffmanCodec()
    codec.fit("AAABBC")
    
    # Test non-binary character
    with pytest.raises(ValueError):
        codec.decode("0120")
        
    # Find a code of length > 1 to create an incomplete sequence
    long_code = None
    for code in codec.codes.values():
        if len(code) > 1:
            long_code = code
            break
            
    assert long_code is not None, "Expected at least one multi-bit code"
    incomplete_sequence = long_code[:-1]
    with pytest.raises(ValueError):
        codec.decode(incomplete_sequence)

def test_larger_round_trip():
    text = "Hello, World! This is a test of the Huffman lossless compression algorithm with various characters."
    codec = HuffmanCodec()
    codec.fit(text)
    
    encoded = codec.encode(text)
    decoded = codec.decode(encoded)
    
    assert decoded == text

def test_prefix_free_property():
    data = "ABABABCCCDDEEEFFF"
    codec = HuffmanCodec()
    codec.fit(data)
    
    codes = list(codec.codes.values())
    for i in range(len(codes)):
        for j in range(len(codes)):
            if i != j:
                assert not codes[i].startswith(codes[j])

def test_tree_structure():
    data = "ABABABCCCD"
    codec = HuffmanCodec()
    codec.fit(data)
    
    assert codec.tree is not None
    assert codec.tree.frequency == len(data)

def test_frequency_correctness():
    data = "AAABBBC"
    codec = HuffmanCodec()
    codec.fit(data)
    
    assert codec.frequencies["A"] == 3
    assert codec.frequencies["B"] == 3
    assert codec.frequencies["C"] == 1
    assert len(codec.frequencies) == 3

def test_refitting_behavior():
    codec = HuffmanCodec()
    
    # Fit first data
    codec.fit("AAAA")
    assert codec.frequencies == {"A": 4}
    assert codec.codes == {"A": "0"}
    
    # Fit new data
    new_data = "ABBC"
    codec.fit(new_data)
    
    # Old frequencies replaced
    assert codec.frequencies == {"A": 1, "B": 2, "C": 1}
    assert "A" in codec.codes and "B" in codec.codes and "C" in codec.codes
    
    assert codec.tree is not None
    assert codec.tree.frequency == 4
    
    # Round-trip new data
    assert codec.decode(codec.encode(new_data)) == new_data
