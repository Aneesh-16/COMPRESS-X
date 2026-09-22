import pytest
from algorithms.lossless.adaptive_huffman import AdaptiveHuffmanCodec

@pytest.fixture
def codec():
    return AdaptiveHuffmanCodec()

def test_empty_input(codec):
    assert codec.encode("") == ""
    assert codec.decode("") == ""

@pytest.mark.parametrize("data", [
    "A",
    "AAAA",
    "BBBBBBBB"
])
def test_single_character_input(codec, data):
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

@pytest.mark.parametrize("data", [
    "ABAB",
    "AABB",
    "ABBA",
    "ABCABC",
    "AAAAABBBBB"
])
def test_basic_repeated_patterns(codec, data):
    assert codec.decode(codec.encode(data)) == data

@pytest.mark.parametrize("data", [
    "BANANA",
    "MISSISSIPPI",
    "ABRACADABRA"
])
def test_mixed_frequency_input(codec, data):
    assert codec.decode(codec.encode(data)) == data

def test_longer_text(codec):
    data = "Adaptive Huffman coding updates its tree as symbols are processed."
    assert codec.decode(codec.encode(data)) == data
    
    # Test a longer repeated sequence to exercise many tree updates
    repeated_data = "ABCDEFG" * 20
    assert codec.decode(codec.encode(repeated_data)) == repeated_data

@pytest.mark.parametrize("data", [
    "Hello 世界",
    "café",
    "😀😀AB"
])
def test_unicode_support(codec, data):
    assert codec.decode(codec.encode(data)) == data

def test_deterministic_encoding():
    data = "ABRACADABRA"
    codec1 = AdaptiveHuffmanCodec()
    codec2 = AdaptiveHuffmanCodec()
    
    encoded1 = codec1.encode(data)
    encoded2 = codec2.encode(data)
    
    assert encoded1 == encoded2

def test_fresh_codec_reset_behavior():
    codec = AdaptiveHuffmanCodec()
    codec.encode("ABBA")
    result1 = codec.encode("HELLO")
    
    fresh = AdaptiveHuffmanCodec()
    result2 = fresh.encode("HELLO")
    
    assert result1 == result2

@pytest.mark.parametrize("data", [
    "A",
    "ABBA",
    "MISSISSIPPI",
    "Hello 世界",
    "ABCDEFG" * 10
])
def test_tree_validation_after_encoding(codec, data):
    codec.encode(data)
    codec._validate_tree()
    codec._check_sibling_property()

@pytest.mark.parametrize("data", [
    "A",
    "ABBA",
    "MISSISSIPPI",
    "Hello 世界",
    "ABCDEFG" * 10
])
def test_tree_validation_after_decoding(codec, data):
    encoded = codec.encode(data)
    
    decoder = AdaptiveHuffmanCodec()
    decoded = decoder.decode(encoded)
    
    assert decoded == data
    decoder._validate_tree()
    decoder._check_sibling_property()

def test_invalid_encoded_data(codec):
    # Invalid binary character
    with pytest.raises(ValueError):
        codec.decode("0123")
        
    # Truncated NYT symbol data (less than 21 bits)
    with pytest.raises(ValueError):
        # 0 to traverse to NYT, but only a few bits of the symbol
        codec.decode("010101")
        
def test_unicode_boundary_behavior(codec):
    # Valid maximum Unicode
    data = chr(0x10FFFF)
    assert codec.decode(codec.encode(data)) == data
    
    # Surrogate characters should raise ValueError
    with pytest.raises(ValueError):
        codec.encode(chr(0xD800))
        
    with pytest.raises(ValueError):
        codec.encode(chr(0xDFFF))

def test_corrupted_truncated_streams(codec):
    # 1. Test a clearly incomplete first NYT payload.
    # A fresh codec expects a 21-bit Unicode payload for the first symbol.
    with pytest.raises(ValueError):
        codec.decode("0" * 20)
        
    # 2. Test a stream containing an invalid binary character:
    with pytest.raises(ValueError):
        codec.decode("0101X10")
        
    # 3. Test an incomplete NYT payload after a valid NYT traversal.
    # "AB" encodes: NYT(empty) + A(21 bits) + path_to_NYT + B(21 bits)
    # Truncating 5 bits from the end means we are strictly missing bits of B's 21-bit payload,
    # which is definitely invalid.
    data = "AB"
    encoded = codec.encode(data)
    with pytest.raises(ValueError):
        codec.decode(encoded[:-5])

@pytest.mark.parametrize("data", [
    "ABBA",
    "MISSISSIPPI"
])
def test_internal_node_symbol_mapping(codec, data):
    codec.encode(data)
    
    unique_chars = set(data)
    assert len(codec.symbols) == len(unique_chars)
    
    for char, node in codec.symbols.items():
        assert node.symbol == char
        assert node.left is None
        assert node.right is None

def test_node_dictionary_consistency(codec):
    data = "MISSISSIPPI"
    codec.encode(data)
    
    # _validate_tree() internally checks that self.nodes is consistent
    # with the tree traversal (e.g. all nodes in traversal are in dict and vice-versa)
    # We call it to ensure it passes.
    codec._validate_tree()
