import pytest
from algorithms.lossless.lz77 import LZ77Codec

def test_constructor_validation():
    # Valid
    codec = LZ77Codec(5, 5)
    assert codec.window_size == 5
    assert codec.lookahead_size == 5

    # Invalid types
    with pytest.raises(TypeError):
        LZ77Codec(True, 5)
    with pytest.raises(TypeError):
        LZ77Codec(5, False)
    with pytest.raises(TypeError):
        LZ77Codec(5.0, 5)

    # Invalid values
    with pytest.raises(ValueError):
        LZ77Codec(0, 5)
    with pytest.raises(ValueError):
        LZ77Codec(5, 0)
    with pytest.raises(ValueError):
        LZ77Codec(-1, 5)

def test_empty_input():
    codec = LZ77Codec(5, 5)
    assert codec.encode("") == ""
    assert codec.tokens == []
    
    assert codec.decode("") == ""
    assert codec.tokens == []

@pytest.mark.parametrize("data", [
    "A",
    "AB",
    "AAAAAA",
    "ABABABAB",
    "ABRACADABRA",
    "HELLO WORLD",
    "The quick brown fox jumps over the lazy dog",
    "Punctuation! @#&(*#$)",
    "Moderately repetitive moderately repetitive moderately repetitive"
])
def test_basic_round_trips(data):
    codec = LZ77Codec(window_size=10, lookahead_size=10)
    encoded = codec.encode(data)
    assert codec.decode(encoded) == data

@pytest.mark.parametrize("data", [
    "Hello 世界",
    "😀😀AB",
    "Mixed Unicode \u0394 \u2211 \U0001f914 text",
    "Punctuation! @#&(*#$) + \u4e16\u754c"
])
def test_unicode_round_trips(data):
    codec = LZ77Codec(window_size=10, lookahead_size=10)
    encoded = codec.encode(data)
    assert codec.decode(encoded) == data

def test_literal_token_generation():
    codec = LZ77Codec(5, 5)
    encoded = codec.encode("A")
    assert codec.tokens[0] == (0, 0, "A")
    assert encoded == "<0,0,41>"

def test_serialization():
    codec = LZ77Codec(5, 5)
    encoded = codec.encode("ABABA")
    # A -> <0,0,41>
    # B -> <0,0,42>
    # ABA -> offset 2, len 3, terminal -> <2,3,>
    assert encoded == "<0,0,41><0,0,42><2,3,>"
    assert encoded == encoded.upper().replace('X', 'x')  # Hex must be uppercase except the brackets are not letters

    encoded1 = codec.encode("ABRACADABRA")
    encoded2 = codec.encode("ABRACADABRA")
    assert encoded1 == encoded2

def test_longest_match_behavior():
    # For data where a short match is near and a long match is far
    # e.g., "AB...A...AB"
    # "XY A Z AB ... AB"
    codec = LZ77Codec(10, 10)
    # data: "X" "Y" "A" "Z" "A" "B" "W" "A" "B"
    # index: 0   1   2   3   4   5   6   7   8
    data = "XYZAZABWAB"
    # After W, we are at index 7 ("AB")
    # Matches for "AB":
    # at index 4 (offset 3): length 2 ("AB")
    # at index 2 (offset 5): length 1 ("A", next is "Z")
    codec.encode(data)
    # The token for "AB" at the end should be length 2
    # The last token is a terminal token for "AB"
    last_token = codec.tokens[-1]
    assert last_token[1] == 2  # length 2
    assert last_token[0] == 3  # offset 3

def test_tie_breaking():
    codec = LZ77Codec(20, 10)
    # two matches of same length.
    # index: 01 23 45 67 8 9 10 11
    # string: AB CD AB EF G AB C
    # At index 9 ("AB"), previous matches at index 0 and index 4.
    # index 0 is offset 9. index 4 is offset 5.
    # It should pick offset 5.
    data = "ABCDABEFGAB"
    codec.encode(data)
    # The last token is for "AB" at the end, terminal match length 2.
    last_token = codec.tokens[-1]
    assert last_token[1] == 2
    assert last_token[0] == 5 # nearest match!

def test_overlapping_matches():
    codec = LZ77Codec(10, 10)
    
    encoded = codec.encode("AAAAAA")
    assert codec.decode(encoded) == "AAAAAA"
    # AAAAAA tokens:
    # 1. A -> (0, 0, 'A')
    # 2. AAAAA -> offset 1, length 5, "" -> (1, 5, "")
    assert codec.tokens[-1][1] > codec.tokens[-1][0] # length > offset

    encoded = codec.encode("ABABABAB")
    assert codec.decode(encoded) == "ABABABAB"
    # 1. A
    # 2. B
    # 3. ABABAB -> offset 2, length 6 -> (2, 6, "")
    assert codec.tokens[-1][1] > codec.tokens[-1][0]

@pytest.mark.parametrize("window_size", [1, 2, 4, 8, 16])
def test_window_size_behavior(window_size):
    codec = LZ77Codec(window_size, 10)
    data = "ABRACADABRA" * 3
    encoded = codec.encode(data)
    assert codec.decode(encoded) == data
    for token in codec.tokens:
        assert token[0] <= window_size

@pytest.mark.parametrize("lookahead_size", [1, 2, 3, 4, 8])
def test_lookahead_size_behavior(lookahead_size):
    codec = LZ77Codec(10, lookahead_size)
    data = "ABRACADABRA" * 3
    encoded = codec.encode(data)
    assert codec.decode(encoded) == data
    for token in codec.tokens:
        assert token[1] <= lookahead_size

def test_terminal_match():
    codec = LZ77Codec(10, 10)
    data = "ABABABA"
    encoded = codec.encode(data)
    # Token structure:
    # A
    # B
    # ABABA -> (2, 5, "")
    last_token = codec.tokens[-1]
    assert last_token[2] == ""
    assert last_token[0] > 0
    assert last_token[1] > 0
    assert codec.decode(encoded) == data

def test_token_structure():
    codec = LZ77Codec(10, 10)
    codec.encode("ABRACADABRA")
    assert isinstance(codec.tokens, list)
    for token in codec.tokens:
        assert isinstance(token, tuple)
        assert len(token) == 3
        assert isinstance(token[0], int)
        assert isinstance(token[1], int)
        assert isinstance(token[2], str)
        assert token[2] == "" or len(token[2]) == 1

def test_malformed_decoder_inputs():
    codec = LZ77Codec(10, 10)
    
    with pytest.raises(ValueError, match="must begin with"):
        codec.decode("0,0,41><0,0,42>")
        
    with pytest.raises(ValueError, match="must end with"):
        codec.decode("<0,0,41><0,0,42")
        
    with pytest.raises(ValueError, match="exactly three fields"):
        codec.decode("<0,0>")
        
    with pytest.raises(ValueError, match="must be integers"):
        codec.decode("<X,0,41>")
        
    with pytest.raises(ValueError, match="Negative offset"):
        codec.decode("<-1,0,41>")
        
    with pytest.raises(ValueError, match="Offset 0 is only valid when length is 0"):
        codec.decode("<0,1,41>")
        
    with pytest.raises(ValueError, match="Length 0 is only valid when offset is 0"):
        codec.decode("<1,0,41>")
        
    with pytest.raises(ValueError, match="exceeds configured window size"):
        codec.decode("<0,0,41><11,1,41>")
        
    with pytest.raises(ValueError, match="exceeds configured lookahead size"):
        codec.decode("<0,0,41><1,11,41>")
        
    with pytest.raises(ValueError, match="exceeds current decoded output"):
        codec.decode("<0,0,41><2,1,41>")
        
    with pytest.raises(ValueError, match="must contain only hexadecimal digits"):
        codec.decode("<0,0,4X>")
        
    with pytest.raises(ValueError, match="must contain only hexadecimal digits"):
        codec.decode("<0,0, 41>")
        
    with pytest.raises(ValueError, match="must contain only hexadecimal digits"):
        codec.decode("<0,0,-41>")
        
    with pytest.raises(ValueError, match="must contain only hexadecimal digits"):
        codec.decode("<0,0,+41>")
        
    with pytest.raises(ValueError, match="Invalid Unicode"):
        codec.decode("<0,0,110000>")
        
    with pytest.raises(ValueError, match="Terminal empty symbol encountered before"):
        codec.decode("<0,0,><0,0,41>")
        
    with pytest.raises(ValueError, match="must have offset > 0 and length > 0"):
        codec.decode("<0,0,>")
        
def test_valid_manually_constructed_decode_streams():
    codec = LZ77Codec(10, 10)
    
    # Literal sequence "AB"
    assert codec.decode("<0,0,41><0,0,42>") == "AB"
    
    # Overlapping sequence "AAAAAA"
    assert codec.decode("<0,0,41><1,5,>") == "AAAAAA"

def test_decoder_token_exposure():
    codec = LZ77Codec(10, 10)
    codec.decode("<0,0,41><0,0,42><2,5,>")
    assert codec.tokens == [
        (0, 0, "A"),
        (0, 0, "B"),
        (2, 5, "")
    ]

def test_state_reset():
    codec = LZ77Codec(10, 10)
    codec.encode("AB")
    assert len(codec.tokens) == 2
    
    codec.encode("A")
    assert len(codec.tokens) == 1
    
    codec.decode("<0,0,41><0,0,42>")
    assert len(codec.tokens) == 2
    
    codec.encode("")
    assert len(codec.tokens) == 0

def test_determinism():
    codec1 = LZ77Codec(10, 10)
    codec2 = LZ77Codec(10, 10)
    
    data = "ABRACADABRA"
    encoded1 = codec1.encode(data)
    encoded2 = codec2.encode(data)
    
    assert encoded1 == encoded2
    assert codec1.tokens == codec2.tokens

def test_type_validation():
    codec = LZ77Codec(10, 10)
    
    with pytest.raises(TypeError):
        codec.encode(123)
    with pytest.raises(TypeError):
        codec.encode(True)
        
    with pytest.raises(TypeError):
        codec.decode(123)
    with pytest.raises(TypeError):
        codec.decode(True)

def test_regression_overlapping_decoding():
    codec = LZ77Codec(10, 10)
    # Stream for "ABABAB"
    # A -> <0,0,41>
    # B -> <0,0,42>
    # ABAB -> <2,4,> (length 4 > offset 2)
    decoded = codec.decode("<0,0,41><0,0,42><2,4,>")
    assert decoded == "ABABAB"
