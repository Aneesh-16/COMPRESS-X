"""
LZ77 Lossless Compression Algorithm.

This module implements the classical LZ77 sliding-window compression algorithm.
It encodes strings into a sequence of triples: (offset, length, next_symbol).
Matches can overlap, and tie-breaking selects the nearest match (smallest offset).
"""

from typing import List, Tuple

class LZ77Codec:
    """
    LZ77 Codec for lossless string compression.
    
    The algorithm maintains a search window of past characters and a look-ahead
    buffer of upcoming characters to find repeated sequences.
    """
    
    def __init__(self, window_size: int, lookahead_size: int):
        """
        Initialize the LZ77 codec with specific buffer sizes.
        
        Args:
            window_size: The maximum number of characters to look back (must be > 0).
            lookahead_size: The maximum length of a match to find (must be > 0).
            
        Raises:
            TypeError: If window_size or lookahead_size is not an integer or is a boolean.
            ValueError: If window_size <= 0 or lookahead_size <= 0.
        """
        if type(window_size) is bool or type(lookahead_size) is bool:
            raise TypeError("window_size and lookahead_size must not be booleans.")
            
        if not isinstance(window_size, int) or not isinstance(lookahead_size, int):
            raise TypeError("window_size and lookahead_size must be integers.")
            
        if window_size <= 0 or lookahead_size <= 0:
            raise ValueError("window_size and lookahead_size must be positive integers.")
            
        self.window_size = window_size
        self.lookahead_size = lookahead_size
        # Expose logical tokens for visualization readiness
        self.tokens: List[Tuple[int, int, str]] = []

    def encode(self, data: str) -> str:
        """
        Encode a string into a serialized LZ77 token stream.
        
        Args:
            data: The string to compress.
            
        Returns:
            The serialized LZ77 token stream.
            
        Raises:
            TypeError: If data is not a string or is a boolean.
        """
        if type(data) is bool or not isinstance(data, str):
            raise TypeError("Data must be a string.")
            
        if not data:
            self.tokens = []
            return ""
            
        self.tokens = []
        cursor = 0
        n = len(data)
        serialized_tokens = []
        
        while cursor < n:
            best_offset = 0
            best_length = 0
            
            # Determine the start of our search window bounds
            window_start = max(0, cursor - self.window_size)
            
            # Search for the longest valid match in the window.
            # Iterating offset from 1 upwards means we test the nearest matches first.
            # The strictly greater `>` comparison ensures we only pick a further match
            # if it is strictly longer, satisfying the tie-break requirement (smallest offset).
            max_offset = cursor - window_start
            for offset in range(1, max_offset + 1):
                match_len = 0
                max_match = min(self.lookahead_size, n - cursor)
                
                # Check characters sequentially, which naturally allows overlapping 
                # (match_len >= offset) as long as it repeats the pattern.
                while match_len < max_match:
                    if data[cursor - offset + match_len] == data[cursor + match_len]:
                        match_len += 1
                    else:
                        break
                        
                if match_len > best_length:
                    best_length = match_len
                    best_offset = offset
                    
            if best_length == 0:
                # No match found; output literal
                symbol = data[cursor]
                token = (0, 0, symbol)
                cursor += 1
            else:
                # Match found
                if cursor + best_length == n:
                    # Terminal match reaching EOF, no next symbol available
                    symbol = ""
                    token = (best_offset, best_length, symbol)
                    cursor += best_length
                else:
                    symbol = data[cursor + best_length]
                    token = (best_offset, best_length, symbol)
                    cursor += best_length + 1
                    
            self.tokens.append(token)
            
            # Serialize: <offset,length,hex_code>
            # empty symbol -> empty 3rd field
            if symbol == "":
                hex_sym = ""
            else:
                hex_sym = hex(ord(symbol))[2:].upper()
                
            serialized_tokens.append(f"<{token[0]},{token[1]},{hex_sym}>")
            
        return "".join(serialized_tokens)

    def decode(self, encoded_data: str) -> str:
        """
        Decode a serialized LZ77 token stream back to the original string.
        
        Args:
            encoded_data: The serialized token stream.
            
        Returns:
            The reconstructed original string.
            
        Raises:
            TypeError: If encoded_data is not a string or is a boolean.
            ValueError: If the stream is malformed or contains invalid logic.
        """
        if type(encoded_data) is bool or not isinstance(encoded_data, str):
            raise TypeError("Encoded data must be a string.")
            
        if not encoded_data:
            self.tokens = []
            return ""
            
        self.tokens = []
        output = []
        
        # Split by '<', expecting empty first element
        parts = encoded_data.split("<")
        if len(parts) == 1 or parts[0] != "":
            raise ValueError("Malformed encoded stream: must begin with '<'.")
            
        # Iterate over parts, skipping the first empty element
        for i in range(1, len(parts)):
            part = parts[i]
            
            if not part.endswith(">"):
                raise ValueError("Malformed encoded stream: token must end with '>'.")
                
            content = part[:-1]  # Strip trailing '>'
            fields = content.split(",")
            
            if len(fields) != 3:
                raise ValueError("Malformed token: expected exactly three fields separated by commas.")
                
            try:
                offset = int(fields[0])
                length = int(fields[1])
            except ValueError:
                raise ValueError("Invalid offset or length: must be integers.")
                
            if offset < 0 or length < 0:
                raise ValueError("Negative offset or length are not allowed.")
                
            if offset == 0 and length != 0:
                raise ValueError("Offset 0 is only valid when length is 0.")
                
            if length == 0 and offset != 0:
                raise ValueError("Length 0 is only valid when offset is 0.")
                
            if offset > self.window_size:
                raise ValueError(f"Offset {offset} exceeds configured window size {self.window_size}.")
                
            if length > self.lookahead_size:
                raise ValueError(f"Length {length} exceeds configured lookahead size {self.lookahead_size}.")
                
            if offset > len(output):
                raise ValueError(f"Offset {offset} exceeds current decoded output length {len(output)}.")
                
            sym_hex = fields[2]
            
            if sym_hex == "":
                symbol = ""
                # Terminal token check
                if i != len(parts) - 1:
                    raise ValueError("Terminal empty symbol encountered before the end of the stream.")
                if offset == 0 or length == 0:
                    raise ValueError("Terminal empty symbol must have offset > 0 and length > 0.")
            else:
                if not all(c in "0123456789abcdefABCDEF" for c in sym_hex):
                    raise ValueError("Symbol code must contain only hexadecimal digits.")
                try:
                    code_point = int(sym_hex, 16)
                except ValueError:
                    raise ValueError("Malformed hexadecimal symbol encoding.")
                
                if not (0 <= code_point <= 0x10FFFF):
                    raise ValueError(f"Invalid Unicode code point: {code_point}")
                symbol = chr(code_point)
                    
            self.tokens.append((offset, length, symbol))
            
            # Reconstruction logic accommodating overlapping matches
            if length > 0:
                read_idx = len(output) - offset
                for _ in range(length):
                    output.append(output[read_idx])
                    read_idx += 1
                    
            if symbol != "":
                output.append(symbol)
                
        return "".join(output)
