import pytest
from algorithms.lossless.lz78 import LZ78Codec

def test_basic_round_trips():
    inputs = [
        "",
        "A",
        "AB",
        "ABC",
        "AAAAAA",
        "ABABABA",
        "ABRACADABRA",
        "TOBEORNOTTOBEORTOBEORNOT",
        "Hello World",
        "Hello 世界",
        "😀😀AB"
    ]
    for data in inputs:
        codec = LZ78Codec()
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)
        assert decoded == data

def test_expected_token_structure():
    codec = LZ78Codec()
    
    codec.encode("A")
    assert codec.tokens == [(0, "A")]
    
    codec.encode("AB")
    assert codec.tokens == [(0, "A"), (0, "B")]
    
    codec.encode("ABA")
    assert codec.tokens == [(0, "A"), (0, "B"), (1, "")]
    
    codec.encode("AA")
    assert codec.tokens == [(0, "A"), (1, "")]

    codec.encode("AAAAAA")
    # A (0,A) -> AA (1,A) -> AAA (2,A)
    assert codec.tokens == [(0, "A"), (1, "A"), (2, "A")]

def test_dictionary_state():
    codec = LZ78Codec()
    codec.encode("AA")
    
    # "AA" ends with an existing phrase "A", so the terminal
    # token (1, "") does not create a new dictionary entry.
    assert codec.dictionary == {
        0: "",
        1: "A"
    }

    codec_dec = LZ78Codec()
    encoded = codec.encode("AA")
    codec_dec.decode(encoded)

    assert codec_dec.dictionary == {
        0: "",
        1: "A"
    }

def test_empty_input():
    codec = LZ78Codec()
    encoded = codec.encode("")
    assert encoded == ""
    assert codec.tokens == []
    assert codec.dictionary == {0: ""}

    codec_dec = LZ78Codec()
    decoded = codec_dec.decode("")
    assert decoded == ""
    assert codec_dec.tokens == []
    assert codec_dec.dictionary == {0: ""}

def test_type_validation():
    codec = LZ78Codec()
    invalid_inputs = [None, 123, b"ABC", ["A", "B"], True, False]
    
    for invalid in invalid_inputs:
        with pytest.raises(TypeError):
            codec.encode(invalid)
            
        with pytest.raises(TypeError):
            codec.decode(invalid)

def test_malformed_stream_decoding():
    codec = LZ78Codec()
    
    malformed_streams = [
        "A",                   # Missing < >
        "<0,41",               # Missing >
        "<0>",                 # Missing comma
        "<0,41,42>",           # Too many fields
        "<,41>",               # Missing index
        "<abc,41>",            # Non-numeric index
        "<-1,41>",             # Negative index
        "<1,41>",              # Forward/unknown dictionary reference
        "<0,GG>",              # Malformed hex symbol
        "<0,110000>",          # Unicode code point > 0x10FFFF
        "<0,>",                # Terminal index 0
        "<0,41><1,><0,41>",    # Terminal token before the final token
        "<0, 41>",             # Whitespace inside
    ]
    
    for stream in malformed_streams:
        with pytest.raises(ValueError):
            codec.decode(stream)

def test_terminal_token_behavior():
    codec = LZ78Codec()
    encoded = codec.encode("ABA")
    
    assert "<1,>" in encoded
    assert encoded.endswith("<1,>")
    
    decoded = codec.decode(encoded)
    assert decoded == "ABA"
    
    # Verify terminal token doesn't add to dictionary
    assert len(codec.dictionary) == 3  # 0, 1(A), 2(B)
    assert 3 not in codec.dictionary

def test_reuse_of_codec_object():
    codec = LZ78Codec()
    
    first = "ABABABA"
    encoded1 = codec.encode(first)
    assert codec.decode(encoded1) == first
    
    second = "AAAAAA"
    encoded2 = codec.encode(second)
    assert codec.decode(encoded2) == second

def test_unicode_round_trips():
    inputs = [
        "こんにちは",
        "你好世界",
        "😀😃😄😁",
        "Hello 世界 😀"
    ]
    for data in inputs:
        codec = LZ78Codec()
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)
        assert decoded == data

def test_determinism():
    data_samples = ["ABABABA", "TOBEORNOTTOBEORTOBEORNOT", "Hello 世界"]
    
    for data in data_samples:
        codec1 = LZ78Codec()
        codec2 = LZ78Codec()
        
        assert codec1.encode(data) == codec2.encode(data)

def test_serialization_format():
    codec = LZ78Codec()
    
    assert codec.encode("A") == "<0,41>"
    assert codec.encode("B") == "<0,42>"
    assert codec.encode("😀") == "<0,1F600>"

def test_malformed_hex_unicode_edge_cases():
    codec = LZ78Codec()
    
    # Invalid hex rejected
    with pytest.raises(ValueError):
        codec.decode("<0,G1>")
        
    # > 0x10FFFF rejected
    with pytest.raises(ValueError):
        codec.decode("<0,110000>")
        
    # Valid bounds accepted
    decoded = codec.decode("<0,10FFFF>")
    assert decoded == chr(0x10FFFF)
    
    decoded_zero = codec.decode("<0,0>")
    assert decoded_zero == "\x00"

def test_algorithm_specific_cases():
    inputs = [
        "ABABABAB",
        "ABCABCABC",
        "AAAAABAAAAAB",
        "BANANABANDANA"
    ]
    
    for data in inputs:
        codec = LZ78Codec()
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)
        assert decoded == data
