import pytest
import math
from algorithms.lossless.tunstall import TunstallCodec

# ==============================================================================
# 1. Constructor validation
# ==============================================================================

def test_constructor_valid_sizes():
    for size in [1, 2, 10, 100, 1024]:
        codec = TunstallCodec(size)
        assert codec.dictionary_size == size

def test_constructor_invalid_type():
    with pytest.raises(TypeError):
        TunstallCodec(10.5)
    with pytest.raises(TypeError):
        TunstallCodec("10")
    with pytest.raises(TypeError):
        TunstallCodec(None)
    with pytest.raises(TypeError):
        TunstallCodec(True)
    with pytest.raises(TypeError):
        TunstallCodec(False)

def test_constructor_invalid_value():
    with pytest.raises(ValueError):
        TunstallCodec(0)
    with pytest.raises(ValueError):
        TunstallCodec(-5)

# ==============================================================================
# 2. fit() validation
# ==============================================================================

def test_fit_invalid_input_type():
    codec = TunstallCodec(10)
    with pytest.raises(TypeError):
        codec.fit(123)
    with pytest.raises(TypeError):
        codec.fit(None)

def test_fit_empty_string_resets():
    codec = TunstallCodec(10)
    codec.fit("ABC")
    assert len(codec.phrases) > 0
    codec.fit("")
    assert len(codec.phrases) == 0
    assert len(codec.phrase_to_code) == 0
    assert len(codec.code_to_phrase) == 0
    assert len(codec.phrase_by_first) == 0
    assert codec.width == 0

def test_fit_dictionary_too_small():
    codec = TunstallCodec(2)
    # alphabet size 3 (A, B, C) is greater than dictionary_size 2
    with pytest.raises(ValueError):
        codec.fit("ABC")

def test_fit_basic_properties():
    codec = TunstallCodec(10)
    codec.fit("ABRACADABRA")
    
    assert len(codec.phrases) > 0
    assert len(codec.phrases) <= 10
    
    # Phrases must be unique
    assert len(codec.phrases) == len(set(codec.phrases))
    
    # Phrases must be non-empty
    for phrase in codec.phrases:
        assert len(phrase) > 0
        
    # Consistency of phrase_to_code and code_to_phrase
    assert len(codec.phrase_to_code) == len(codec.code_to_phrase)
    for phrase, code in codec.phrase_to_code.items():
        assert codec.code_to_phrase[code] == phrase
        
    # Width is correct and fixed
    N = len(codec.phrases)
    if N <= 1:
        assert codec.width == 0
    else:
        assert codec.width == math.ceil(math.log2(N))
        
    for code in codec.code_to_phrase.keys():
        assert len(code) == codec.width
        
    # phrase_by_first correctly populated
    for phrase in codec.phrases:
        assert phrase in codec.phrase_by_first[phrase[0]]

# ==============================================================================
# 3. Basic round-trip tests
# ==============================================================================

@pytest.mark.parametrize("data", [
    "AAAAAA",
    "ABABABAB",
    "ABRACADABRA",
    "HELLO WORLD",
    "The quick brown fox jumps over the lazy dog",
    "Hello \u4e16\u754c",
    "\U0001f600\U0001f600AB",
    "Punctuation! @#&(*#$)",
    "Moderately repetitive moderately repetitive moderately repetitive"
])
def test_basic_roundtrip(data):
    # Dictionary size large enough for most small alphabets
    alphabet_size = len(set(data))
    dict_size = max(alphabet_size * 2, 10)
    
    codec = TunstallCodec(dict_size)
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

# ==============================================================================
# 4. Different dictionary sizes
# ==============================================================================

@pytest.mark.parametrize("dict_size", [1, 2, 3, 4, 5, 8, 16])
def test_different_dictionary_sizes(dict_size):
    # Use an alphabet of size compatible with dict_size
    alphabet_size = max(1, dict_size // 2)
    data = "".join([chr(65 + i) for i in range(alphabet_size)]) * 5
    
    codec = TunstallCodec(dict_size)
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data
    assert len(codec.phrases) <= dict_size

# ==============================================================================
# 5. Single-symbol case
# ==============================================================================

@pytest.mark.parametrize("data", [
    "A",
    "AAAA",
    "AAAAAAAAAA"
])
def test_single_symbol_case(data):
    codec = TunstallCodec(10)
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    
    assert decoded == data
    assert len(codec.phrases) == 1
    assert codec.width == 0
    
    # encoded payload should be empty, only header remains
    # Header: length(32) + N(32) + 1 phrase length(32) + char(21) = 117 bits
    assert len(encoded) == 32 + 32 + 32 + 21
    assert encoded[117:] == ""

# ==============================================================================
# 6. Empty input
# ==============================================================================

def test_empty_input():
    codec = TunstallCodec(10)
    assert codec.encode("") == ""
    assert codec.decode("") == ""
    
    codec.fit("ABC")
    codec.fit("")
    assert len(codec.phrases) == 0

# ==============================================================================
# 7. Unicode
# ==============================================================================

@pytest.mark.parametrize("data", [
    "Hello \u4e16\u754c",
    "\U0001f600\U0001f600AB",
    "Mixed Unicode \u0394 \u2211 \U0001F914 text"
])
def test_unicode_roundtrip(data):
    codec = TunstallCodec(16)
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

# ==============================================================================
# 8. Final suffix handling
# ==============================================================================

def test_final_suffix_handling():
    # Force a scenario where the input ends with a proper prefix of a terminal
    # Alphabet: A, B. Let dictionary size = 3.
    # Terminals will be A (if p(A) > p(B)), B, then A expands to AA, AB.
    # So terminals: AA, AB, B
    data = "B" + "AA" * 5 + "A" # ends with a single 'A', which is a proper prefix of 'AA' and 'AB'
    codec = TunstallCodec(3)
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    
    assert decoded == data
    
    # Ensure no arbitrary padding characters were permanently added.
    # The dictionary should contain exactly AA, AB, B
    assert set(codec.phrases) == {"AA", "AB", "B"}
    
    # Encode a suffix-heavy case
    data2 = "AB B AA A" # Last part is A
    encoded2 = codec.encode(data2)
    assert codec.decode(encoded2) == data2

# ==============================================================================
# 9. Malformed decoder inputs
# ==============================================================================

def test_malformed_decoder_streams():
    codec = TunstallCodec(10)
    
    # Streams shorter than 64 bits
    with pytest.raises(ValueError, match="Truncated bitstream: missing header"):
        codec.decode("0101")
        
    # Invalid header characters
    with pytest.raises(ValueError, match="Invalid character in bitstream header"):
        codec.decode("2" * 64)
        
    # original_length == 0 in a non-empty stream
    with pytest.raises(ValueError, match="Non-empty stream declared original length of 0"):
        # original length 0, N = 1, then rest of the stream
        header = "0"*32 + f"{1:032b}" + "0"*32
        codec.decode(header)
        
    # N == 0
    with pytest.raises(ValueError, match="Dictionary must contain at least one phrase"):
        header = f"{10:032b}" + "0"*32 + "0"*32
        codec.decode(header)
        
    # Truncated dictionary metadata (reading phrase length)
    with pytest.raises(ValueError, match="Bitstream too short to contain declared dictionary metadata"):
        header = f"{10:032b}" + f"{1:032b}" + "111"
        codec.decode(header)
        
    # Zero phrase length
    with pytest.raises(ValueError, match="Phrase length must be strictly positive"):
        header = f"{10:032b}" + f"{1:032b}" + "0"*32
        codec.decode(header)
        
    # Truncated phrase character data
    with pytest.raises(ValueError, match="Truncated bitstream while reading phrase characters"):
        header = f"{10:032b}" + f"{1:032b}" + f"{1:032b}" + "1010"
        codec.decode(header)
        
    # Invalid Unicode code point > 0x10FFFF
    with pytest.raises(ValueError, match="Invalid Unicode code point"):
        header = f"{10:032b}" + f"{1:032b}" + f"{1:032b}" + f"{0x110000:021b}"
        codec.decode(header)
        
    # Duplicate dictionary phrases
    with pytest.raises(ValueError, match="Duplicate phrase in dictionary"):
        # N=2, both phrases are "A"
        header = f"{10:032b}" + f"{2:032b}" 
        header += f"{1:032b}" + f"{65:021b}" 
        header += f"{1:032b}" + f"{65:021b}"
        codec.decode(header)
        
    # Invalid payload characters
    with pytest.raises(ValueError, match="Invalid character in bitstream payload"):
        # Valid header for N=2, phrases "A", "B"
        header = f"{10:032b}" + f"{2:032b}" 
        header += f"{1:032b}" + f"{65:021b}" 
        header += f"{1:032b}" + f"{66:021b}"
        payload = "02" # width = 1
        codec.decode(header + payload)
        
    # Payload length not divisible by width
    with pytest.raises(ValueError, match="is not a multiple of codeword width"):
        header = f"{10:032b}" + f"{2:032b}" 
        header += f"{1:032b}" + f"{65:021b}" 
        header += f"{1:032b}" + f"{66:021b}"
        # width = 1, but if we do N=3, width=2
        header3 = f"{10:032b}" + f"{3:032b}" 
        header3 += f"{1:032b}" + f"{65:021b}" 
        header3 += f"{1:032b}" + f"{66:021b}"
        header3 += f"{1:032b}" + f"{67:021b}"
        payload = "001" # length 3, indivisible by 2
        codec.decode(header3 + payload)
        
    # Invalid/unmapped codeword
    with pytest.raises(ValueError, match="Invalid codeword"):
        # N=3, width=2. Valid codes are 00, 01, 10. Code 11 is unmapped.
        header3 = f"{10:032b}" + f"{3:032b}" 
        header3 += f"{1:032b}" + f"{65:021b}" 
        header3 += f"{1:032b}" + f"{66:021b}"
        header3 += f"{1:032b}" + f"{67:021b}"
        payload = "11"
        codec.decode(header3 + payload)
        
    # Width == 0 with non-empty payload
    with pytest.raises(ValueError, match="Payload should be empty when width is 0"):
        header = f"{10:032b}" + f"{1:032b}" + f"{1:032b}" + f"{65:021b}"
        payload = "1"
        codec.decode(header + payload)
        
    # Decoded output shorter than original_length
    with pytest.raises(ValueError, match="is shorter than original length"):
        # Set original length to 100, but payload only encodes 1 character
        header = f"{100:032b}" + f"{2:032b}" 
        header += f"{1:032b}" + f"{65:021b}" 
        header += f"{1:032b}" + f"{66:021b}"
        payload = "0"
        codec.decode(header + payload)

# ==============================================================================
# 10. Type validation
# ==============================================================================

def test_encode_decode_type_validation():
    codec = TunstallCodec(10)
    with pytest.raises(TypeError):
        codec.encode(123)
    with pytest.raises(TypeError):
        codec.decode(123)
    with pytest.raises(TypeError):
        codec.encode(None)
    with pytest.raises(TypeError):
        codec.decode(None)

# ==============================================================================
# 11. Determinism
# ==============================================================================

def test_determinism():
    data = "abracadabra"
    
    codec1 = TunstallCodec(8)
    encoded1 = codec1.encode(data)
    
    codec2 = TunstallCodec(8)
    encoded2 = codec2.encode(data)
    
    assert codec1.phrases == codec2.phrases
    assert codec1.phrase_to_code == codec2.phrase_to_code
    assert encoded1 == encoded2

# ==============================================================================
# 12. Structural properties
# ==============================================================================

def test_structural_properties():
    data = "MISSISSIPPI"
    alphabet = set(data)
    codec = TunstallCodec(8)
    codec.fit(data)
    
    # Every phrase must consist only of symbols from the input alphabet
    for phrase in codec.phrases:
        for char in phrase:
            assert char in alphabet
            
    # phrase_to_code and code_to_phrase must be inverse mappings
    for phrase, code in codec.phrase_to_code.items():
        assert codec.code_to_phrase[code] == phrase
        
    # Codewords must be unique
    assert len(set(codec.phrase_to_code.values())) == len(codec.phrase_to_code.values())
    
    # Every codeword must have exactly `width` bits
    if codec.width > 0:
        for code in codec.phrase_to_code.values():
            assert len(code) == codec.width

# ==============================================================================
# 13. Important Tunstall property
# ==============================================================================

def test_tunstall_expansion_rule():
    data = "A B C D E"
    alphabet = set(data)
    K = len(alphabet)
    assert K > 1
    
    for max_dict_size in range(K, K + 20):
        codec = TunstallCodec(max_dict_size)
        codec.fit(data)
        
        N = len(codec.phrases)
        # Expansion starts with K phrases. Each expansion removes 1 and adds K, net change is + (K-1)
        # So N = K + m * (K - 1) for some integer m >= 0
        assert (N - K) % (K - 1) == 0
        assert N <= max_dict_size

# ==============================================================================
# 14. Regression test for the old suffix-padding bug
# ==============================================================================

def test_regression_suffix_padding():
    # Force an alphabet of A and B
    data2 = "AAAB B A" # A is more frequent, B is present.
    codec2 = TunstallCodec(3)
    encoded2 = codec2.encode(data2)
    decoded2 = codec2.decode(encoded2)
    assert decoded2 == data2
    
    # Terminals should just be AA, AB, B (with some probabilities)
    # Reconstructed result doesn't invent padding characters.
