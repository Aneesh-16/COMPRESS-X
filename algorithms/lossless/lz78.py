"""
LZ78 Lossless Compression Algorithm.

This module provides the LZ78Codec class for encoding and decoding strings
using the Lempel-Ziv 78 algorithm. LZ78 works by building a dynamically
growing phrase dictionary during both encoding and decoding.

Token representation:
    (index, next_symbol)
    where `index` is the dictionary index of the longest previously matched phrase,
    and `next_symbol` is the character appended to it.

Terminal token representation:
    (index, "")
    Used only when the final phrase of the input string exactly matches an existing
    dictionary entry, requiring no new symbols to be added.

Serialization format:
    <index,HEX_SYMBOL>
    For example, 'A' encoded as a new symbol appended to the root (index 0) is <0,41>.
    A terminal token has an empty hex field: <index,>

Dictionary Indexing:
    0 always represents the empty string "".
    New phrases are assigned sequentially increasing indices starting from 1.
"""

class LZ78Codec:
    """
    LZ78 Codec for lossless string compression.
    
    Maintains a continuously growing phrase dictionary.
    Exposes `tokens` and `dictionary` state for visualization and analysis.
    """

    def __init__(self):
        """Initializes the LZ78 codec state."""
        self.dictionary = {0: ""}
        self.tokens = []

    def encode(self, data: str) -> str:
        """
        Encodes a string using the LZ78 algorithm.

        Args:
            data (str): The input string to compress.

        Returns:
            str: The deterministic, serialized string of LZ78 tokens.

        Raises:
            TypeError: If the input data is not a string.
        """
        if type(data) is not str:
            raise TypeError("Input data must be a string.")

        if not data:
            self.dictionary = {0: ""}
            self.tokens = []
            return ""

        # Using phrase -> index for fast O(1) lookups in encoder
        encoder_dict = {"": 0}
        next_index = 1
        current_phrase = ""
        
        self.tokens = []
        serialized_parts = []

        for char in data:
            candidate = current_phrase + char
            if candidate in encoder_dict:
                current_phrase = candidate
            else:
                index = encoder_dict[current_phrase]
                self.tokens.append((index, char))
                encoder_dict[candidate] = next_index
                next_index += 1
                
                hex_symbol = hex(ord(char))[2:].upper()
                serialized_parts.append(f"<{index},{hex_symbol}>")
                current_phrase = ""

        # Handle terminal phrase
        if current_phrase:
            index = encoder_dict[current_phrase]
            self.tokens.append((index, ""))
            serialized_parts.append(f"<{index},>")

        # Populate public dictionary state (index -> phrase)
        self.dictionary = {v: k for k, v in encoder_dict.items()}

        return "".join(serialized_parts)

    def decode(self, encoded_data: str) -> str:
        """
        Decodes a serialized LZ78 stream back into the original string.

        Args:
            encoded_data (str): The serialized LZ78 tokens.

        Returns:
            str: The reconstructed original string.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If the stream is malformed or indices are invalid.
        """
        if type(encoded_data) is not str:
            raise TypeError("Encoded data must be a string.")

        if not encoded_data:
            self.dictionary = {0: ""}
            self.tokens = []
            return ""

        self.dictionary = {0: ""}
        self.tokens = []
        next_index = 1
        output = []

        # Deterministic parser state
        idx = 0
        length = len(encoded_data)

        while idx < length:
            if encoded_data[idx] != '<':
                raise ValueError("Malformed stream: expected '<' at token boundary.")
            
            close_idx = encoded_data.find('>', idx)
            if close_idx == -1:
                raise ValueError("Malformed stream: missing closing '>'.")
            
            token_content = encoded_data[idx + 1:close_idx]
            
            if ',' not in token_content:
                raise ValueError("Malformed stream: missing ',' delimiter in token.")
            
            parts = token_content.split(',')
            if len(parts) != 2:
                raise ValueError("Malformed stream: invalid number of fields in token.")
            
            index_str, hex_str = parts

            if not index_str:
                raise ValueError("Malformed stream: missing index.")
            
            if not index_str.isdigit():
                raise ValueError("Malformed stream: index must be a non-negative integer.")

            dict_index = int(index_str)

            if dict_index < 0:
                raise ValueError("Malformed stream: negative index.")
            
            if dict_index >= next_index:
                raise ValueError(f"Malformed stream: invalid forward reference index {dict_index}.")
            
            if dict_index not in self.dictionary:
                raise ValueError(f"Malformed stream: unknown dictionary reference {dict_index}.")

            is_final_token = (close_idx == length - 1)

            if not hex_str:
                # Terminal token handling
                if not is_final_token:
                    raise ValueError("Malformed stream: terminal empty symbol before final token.")
                
                if dict_index == 0:
                    raise ValueError("Malformed stream: terminal index cannot be 0.")

                self.tokens.append((dict_index, ""))
                phrase = self.dictionary[dict_index]
                output.append(phrase)
                # Terminal token does NOT add a new phrase to the dictionary
            else:
                # Normal token handling
                if any(c not in "0123456789ABCDEFabcdef" for c in hex_str):
                    raise ValueError("Malformed stream: invalid characters in hexadecimal symbol.")
                
                code_point = int(hex_str, 16)
                if code_point > 0x10FFFF:
                    raise ValueError(f"Malformed stream: invalid Unicode code point {code_point}.")
                
                symbol = chr(code_point)
                self.tokens.append((dict_index, symbol))
                phrase = self.dictionary[dict_index] + symbol
                output.append(phrase)
                
                self.dictionary[next_index] = phrase
                next_index += 1

            idx = close_idx + 1

        if idx < length:
            raise ValueError("Malformed stream: extra/trailing characters found.")

        return "".join(output)
