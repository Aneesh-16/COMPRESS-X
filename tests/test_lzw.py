import pytest
from algorithms.lossless.lzw import LZWCodec

# ==================================================
# 1. BASIC ROUND-TRIP TESTS
# ==================================================
@pytest.mark.parametrize("data", [
    "",
    "A",
    "AB",
    "AA",
    "AAAAAA",
    "ABABABA",
    "ABABABAB",
    "ABCABCABC",
    "ABRACADABRA",
    "TOBEORNOTTOBEORTOBEORNOT",
    "Hello World",
    "café",
    "こんにちは",
    "你好世界",
    "😀😀AB",
    "Hello 世界 😀"
])
def test_basic_round_trip(data):
    codec = LZWCodec()
    if not data:
        assert codec.encode(data) == ""
        assert codec.decode("") == ""
    else:
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)
        assert decoded == data

# ==================================================
# 2. EXACT SERIALIZATION TESTS
# ==================================================
@pytest.mark.parametrize("data, expected", [
    ("A", "<INIT:41><CODES:0>"),
    ("AB", "<INIT:41,42><CODES:0,1>"),
    ("ABA", "<INIT:41,42><CODES:0,1,0>"),
    ("😀", "<INIT:1F600><CODES:0>"),
    ("ABABABA", "<INIT:41,42><CODES:0,1,2,4>")
])
def test_exact_serialization(data, expected):
    codec = LZWCodec()
    encoded = codec.encode(data)
    assert encoded == expected

# ==================================================
# 3. ALPHABET TESTS
# ==================================================
def test_alphabet_baba():
    codec = LZWCodec()
    codec.encode("BABA")
    assert codec.alphabet == ["A", "B"]

def test_alphabet_unicode_ordering():
    codec = LZWCodec()
    codec.encode("zAaZ")
    assert codec.alphabet == ["A", "Z", "a", "z"]

def test_alphabet_mixed_unicode():
    codec = LZWCodec()
    codec.encode("a😀A")
    assert codec.alphabet == ["A", "a", "😀"]

# ==================================================
# 4. DICTIONARY GROWTH
# ==================================================
def test_encoder_dictionary_growth():
    codec = LZWCodec()
    codec.encode("ABABABA")
    expected_dict = {
        "A": 0,
        "B": 1,
        "AB": 2,
        "BA": 3,
        "ABA": 4
    }
    assert codec.dictionary == expected_dict
    assert codec.codes == [0, 1, 2, 4]

# ==================================================
# 5. DECODER DICTIONARY
# ==================================================
def test_decoder_dictionary_state():
    codec = LZWCodec()
    codec.decode("<INIT:41,42><CODES:0,1,2,4>")
    expected_dict = {
        0: "A",
        1: "B",
        2: "AB",
        3: "BA",
        4: "ABA"
    }
    assert codec.dictionary == expected_dict
    assert codec.alphabet == ["A", "B"]
    assert codec.codes == [0, 1, 2, 4]

# ==================================================
# 6. KWKWK SPECIAL CASE
# ==================================================
def test_kwkwk_special_case():
    codec = LZWCodec()
    encoded = codec.encode("ABABABA")
    assert encoded == "<INIT:41,42><CODES:0,1,2,4>"
    decoded = codec.decode(encoded)
    assert decoded == "ABABABA"

def test_kwkwk_additional_case():
    codec = LZWCodec()
    encoded = codec.encode("AAAAAA")
    assert encoded == "<INIT:41><CODES:0,1,2>"
    decoded = codec.decode(encoded)
    assert decoded == "AAAAAA"

# ==================================================
# 7. EMPTY INPUT
# ==================================================
def test_empty_input():
    codec = LZWCodec()
    assert codec.encode("") == ""
    assert codec.dictionary == {}
    assert codec.codes == []
    assert codec.alphabet == []

    assert codec.decode("") == ""
    assert codec.dictionary == {}
    assert codec.codes == []
    assert codec.alphabet == []

# ==================================================
# 8. TYPE VALIDATION
# ==================================================
@pytest.mark.parametrize("invalid_input", [None, 123, b"ABC", ["A", "B"]])
def test_encode_type_validation(invalid_input):
    codec = LZWCodec()
    with pytest.raises(TypeError):
        codec.encode(invalid_input)

@pytest.mark.parametrize("invalid_input", [None, 123, b"ABC", ["0", "1"]])
def test_decode_type_validation(invalid_input):
    codec = LZWCodec()
    with pytest.raises(TypeError):
        codec.decode(invalid_input)

# ==================================================
# 9. CODEC REUSE
# ==================================================
def test_codec_reuse():
    codec = LZWCodec()
    
    encoded1 = codec.encode("ABABABA")
    decoded1 = codec.decode(encoded1)
    assert decoded1 == "ABABABA"
    
    encoded2 = codec.encode("AAAAAA")
    decoded2 = codec.decode(encoded2)
    assert decoded2 == "AAAAAA"
    
    encoded3 = codec.encode("Hello 世界 😀")
    decoded3 = codec.decode(encoded3)
    assert decoded3 == "Hello 世界 😀"

# ==================================================
# 10. MALFORMED STRUCTURE
# ==================================================
@pytest.mark.parametrize("malformed", [
    "A",
    "INIT:41><CODES:0>",
    # "<INIT:41><CODES:0>", # This is valid, omitting.
    "<INIT:41><CODES:0>>",
    "<INIT:41><CODES:0>extra",
    "<INIT:41><CODES:0><CODES:1>",
    "<CODES:0><INIT:41>",
    "<INIT:41><INIT:42><CODES:0>",
    "<INIT:41><CODES:0><",
    "<INIT:41<CODES:0>",
    "<INIT:41><CODES:0"
])
def test_decode_malformed_structure(malformed):
    codec = LZWCodec()
    with pytest.raises(ValueError):
        codec.decode(malformed)

# ==================================================
# 11. MALFORMED INIT
# ==================================================
@pytest.mark.parametrize("malformed", [
    "<INIT:><CODES:0>",
    "<INIT:41,,42><CODES:0>",
    "<INIT:41,42,><CODES:0>",
    "<INIT:GG><CODES:0>",
    "<INIT:-1><CODES:0>",
    "<INIT:110000><CODES:0>",
    "<INIT:41,41><CODES:0>",
    "<INIT:42,41><CODES:0>"
])
def test_decode_malformed_init(malformed):
    codec = LZWCodec()
    with pytest.raises(ValueError):
        codec.decode(malformed)

# ==================================================
# 12. MALFORMED CODES
# ==================================================
@pytest.mark.parametrize("malformed", [
    "<INIT:41><CODES:>",
    "<INIT:41,42><CODES:0,>",
    "<INIT:41,42><CODES:0,,1>",
    "<INIT:41,42><CODES:0,A>",
    "<INIT:41,42><CODES:-1>",
    "<INIT:41,42><CODES:2>",
    "<INIT:41,42><CODES:0,3>"
])
def test_decode_malformed_codes(malformed):
    codec = LZWCodec()
    with pytest.raises(ValueError):
        codec.decode(malformed)

# ==================================================
# 13. VALID KWKWK BOUNDARY
# ==================================================
def test_valid_kwkwk_boundary():
    codec = LZWCodec()
    # This should succeed
    decoded = codec.decode("<INIT:41,42><CODES:0,1,2,4>")
    assert decoded == "ABABABA"

def test_invalid_code_boundary():
    codec = LZWCodec()
    # code > next_code should fail
    with pytest.raises(ValueError):
        codec.decode("<INIT:41,42><CODES:0,1,2,5>")

# ==================================================
# 14. UNICODE
# ==================================================
@pytest.mark.parametrize("data", [
    "é",
    "€",
    "こんにちは",
    "你好世界",
    "안녕하세요",
    "😀",
    "😀😃😄😁",
    "Hello 世界 😀",
    "₹₹₹",
    "café résumé"
])
def test_unicode_round_trip(data):
    codec = LZWCodec()
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

    # Verify uppercase hex
    init_block = encoded.split("><")[0]
    hex_values = init_block.replace("<INIT:", "").split(",")
    for h in hex_values:
        assert h.isupper() or h.isdigit()

# ==================================================
# 15. UNICODE BOUNDARY
# ==================================================
def test_unicode_boundary():
    codec = LZWCodec()
    data = chr(0x10FFFF)
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data

def test_unicode_out_of_bounds():
    codec = LZWCodec()
    with pytest.raises(ValueError):
        codec.decode("<INIT:110000><CODES:0>")

# ==================================================
# 16. DETERMINISM
# ==================================================
def test_determinism():
    inputs = ["ABCABC", "Hello World", "こんにちは", "😀😀AB"]
    for data in inputs:
        codec1 = LZWCodec()
        codec2 = LZWCodec()
        
        enc1 = codec1.encode(data)
        enc2 = codec2.encode(data)
        
        assert enc1 == enc2
        assert codec1.alphabet == codec2.alphabet
        assert codec1.codes == codec2.codes

# ==================================================
# 17. PUBLIC STATE
# ==================================================
def test_public_state():
    codec = LZWCodec()
    
    # After encode
    codec.encode("AB")
    assert isinstance(list(codec.dictionary.keys())[0], str)
    assert isinstance(list(codec.dictionary.values())[0], int)
    
    # After decode
    codec.decode("<INIT:41,42><CODES:0,1>")
    assert isinstance(list(codec.dictionary.keys())[0], int)
    assert isinstance(list(codec.dictionary.values())[0], str)
    
    assert codec.alphabet == ["A", "B"]
    assert codec.codes == [0, 1]

# ==================================================
# 18. LZW ALGORITHM CASES
# ==================================================
@pytest.mark.parametrize("data", [
    "TOBEORNOTTOBEORTOBEORNOT",
    "ABRACADABRA",
    "BANANABANDANA",
    "ABCABCABCABC",
    "AAAAABAAAAAB"
])
def test_lzw_algorithm_cases(data):
    codec = LZWCodec()
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert decoded == data
