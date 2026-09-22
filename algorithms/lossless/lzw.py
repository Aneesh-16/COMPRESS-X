"""
LZW Lossless Compression Algorithm.

This module provides the LZWCodec class for encoding and decoding strings
using the Lempel-Ziv-Welch (LZW) algorithm. It builds a continuously growing
phrase dictionary dynamically based on the input data.

The initial alphabet is constructed from the unique characters present in the
input string, sorted by Unicode code point, to support arbitrary Unicode cleanly.

Serialization format:
    <INIT:HEX_1,HEX_2,...><CODES:CODE_1,CODE_2,...>
"""

class LZWCodec:
    """
    LZW Codec for lossless string compression.
    
    Maintains a phrase dictionary based on the unique characters in the input.
    Exposes `tokens`/`codes`, `alphabet`, and `dictionary` state for visualization.
    """

    def __init__(self):
        """Initializes the LZW codec state."""
        self.dictionary = {}
        self.codes = []
        self.alphabet = []

    def encode(self, data: str) -> str:
        """
        Encodes a string using the LZW algorithm.

        Args:
            data (str): The input string to compress.

        Returns:
            str: The deterministic, serialized string of LZW codes and alphabet.

        Raises:
            TypeError: If the input data is not a string.
        """
        if type(data) is not str:
            raise TypeError("Input data must be a string.")

        self.dictionary = {}
        self.codes = []
        self.alphabet = []

        if not data:
            return ""

        # Extract unique characters and sort by Unicode code point
        unique_chars = sorted(list(set(data)))
        self.alphabet = unique_chars
        
        for i, char in enumerate(unique_chars):
            self.dictionary[char] = i
            
        next_code = len(unique_chars)
        
        w = ""
        
        for c in data:
            candidate = w + c
            if candidate in self.dictionary:
                w = candidate
            else:
                self.codes.append(self.dictionary[w])
                self.dictionary[candidate] = next_code
                next_code += 1
                w = c

        if w:
            self.codes.append(self.dictionary[w])
            
        init_hex = [hex(ord(c))[2:].upper() for c in unique_chars]
        init_str = ",".join(init_hex)
        codes_str = ",".join(str(code) for code in self.codes)
        
        return f"<INIT:{init_str}><CODES:{codes_str}>"

    def decode(self, encoded_data: str) -> str:
        """
        Decodes a serialized LZW stream back into the original string.

        Args:
            encoded_data (str): The serialized LZW data.

        Returns:
            str: The reconstructed original string.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If the stream is malformed or invalid.
        """
        if type(encoded_data) is not str:
            raise TypeError("Encoded data must be a string.")

        self.dictionary = {}
        self.codes = []
        self.alphabet = []

        if not encoded_data:
            return ""

        # Validation: Must start with <INIT:
        if not encoded_data.startswith("<INIT:"):
            raise ValueError("Malformed stream: expected '<INIT:' at start.")
            
        init_end_idx = encoded_data.find("><CODES:")
        if init_end_idx == -1:
            raise ValueError("Malformed stream: missing '><CODES:' separator.")
            
        codes_end_idx = encoded_data.find(">", init_end_idx + 8)
        if codes_end_idx == -1:
            raise ValueError("Malformed stream: missing closing '>' for CODES.")
            
        if codes_end_idx != len(encoded_data) - 1:
            raise ValueError("Malformed stream: extra trailing characters.")
            
        init_content = encoded_data[6:init_end_idx]
        codes_content = encoded_data[init_end_idx + 8:codes_end_idx]
        
        if not init_content:
            raise ValueError("Malformed stream: INIT block is empty.")
            
        if not codes_content:
            raise ValueError("Malformed stream: CODES block is empty.")
            
        # Parse INIT
        hex_symbols = init_content.split(',')
        previous_cp = -1
        
        for hex_str in hex_symbols:
            if not hex_str:
                raise ValueError("Malformed stream: empty field in INIT.")
            
            if any(c not in "0123456789ABCDEFabcdef" for c in hex_str):
                raise ValueError("Malformed stream: invalid hex in INIT.")
                
            code_point = int(hex_str, 16)
            if code_point > 0x10FFFF:
                raise ValueError(f"Malformed stream: invalid Unicode code point {code_point}.")
                
            if code_point <= previous_cp:
                if code_point == previous_cp:
                    raise ValueError("Malformed stream: duplicate code points in INIT.")
                raise ValueError("Malformed stream: unsorted code points in INIT.")
                
            previous_cp = code_point
            char = chr(code_point)
            self.alphabet.append(char)
            self.dictionary[len(self.alphabet) - 1] = char
            
        # Parse CODES
        code_strs = codes_content.split(',')
        for c_str in code_strs:
            if not c_str:
                raise ValueError("Malformed stream: empty code field.")
            if not c_str.isdigit():
                raise ValueError("Malformed stream: invalid or negative integer code.")
            self.codes.append(int(c_str))
            
        # Standard LZW Decoding
        next_code = len(self.alphabet)
        first_code = self.codes[0]
        
        if first_code >= len(self.alphabet):
            raise ValueError("Malformed stream: invalid first code.")
            
        previous_phrase = self.dictionary[first_code]
        output = [previous_phrase]
        
        for code in self.codes[1:]:
            if code in self.dictionary:
                current_phrase = self.dictionary[code]
            elif code == next_code:
                # KwKwK case
                current_phrase = previous_phrase + previous_phrase[0]
            else:
                raise ValueError(f"Malformed stream: invalid forward reference {code}.")
                
            output.append(current_phrase)
            self.dictionary[next_code] = previous_phrase + current_phrase[0]
            next_code += 1
            previous_phrase = current_phrase
            
        return "".join(output)
