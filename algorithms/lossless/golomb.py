import math
from typing import Iterable, List

class GolombCodec:
    """
    Implements Golomb lossless coding for non-negative integers.
    
    Golomb coding uses a divisor parameter `m`. For each non-negative integer `n`:
    - The quotient `q = n // m` is encoded in unary (q zeros followed by a one).
    - The remainder `r = n % m` is encoded in truncated binary.
    """
    
    def __init__(self, m: int):
        if not isinstance(m, int) or isinstance(m, bool):
            raise TypeError(f"Golomb parameter m must be an integer, got {type(m).__name__}")
        if m < 1:
            raise ValueError(f"Golomb parameter m must be a positive integer (>= 1), got {m}")
        
        self.m = m
        self.b = math.ceil(math.log2(m)) if m > 1 else 0
        self.cutoff = (1 << self.b) - m if m > 1 else 0

    def encode(self, data: Iterable[int]) -> str:
        """
        Encodes a sequence of non-negative integers into a binary string.
        """
        encoded_bits = []
        for n in data:
            if not isinstance(n, int) or isinstance(n, bool):
                raise TypeError(f"Values must be integers, got {type(n).__name__}")
            if n < 0:
                raise ValueError(f"Values must be non-negative, got {n}")
                
            q = n // self.m
            r = n % self.m
            
            # 1. Unary coding for quotient: q zeros followed by a one
            encoded_bits.append('0' * q + '1')
            
            # 2. Truncated binary coding for remainder
            if self.m > 1:
                if r < self.cutoff:
                    # Encode r in exactly b - 1 bits
                    encoded_bits.append(format(r, f'0{self.b - 1}b'))
                else:
                    # Encode r + cutoff in exactly b bits
                    encoded_bits.append(format(r + self.cutoff, f'0{self.b}b'))
                    
        return "".join(encoded_bits)

    def decode(self, encoded_data: str) -> List[int]:
        """
        Decodes a binary string into a list of non-negative integers.
        """
        if not isinstance(encoded_data, str):
            raise TypeError("Encoded data must be a string")
            
        decoded = []
        n = len(encoded_data)
        i = 0
        
        while i < n:
            # 1. Decode quotient (unary)
            q = 0
            while i < n and encoded_data[i] == '0':
                q += 1
                i += 1
                
            if i >= n:
                raise ValueError("Incomplete unary code (missing terminating '1')")
            
            if encoded_data[i] != '1':
                raise ValueError(f"Invalid character in bitstream: '{encoded_data[i]}'")
                
            i += 1 # Consume the '1'
            
            # 2. Decode remainder (truncated binary)
            r = 0
            if self.m > 1:
                # Read b - 1 bits
                if i + self.b - 1 > n:
                    raise ValueError("Incomplete remainder (EOF while reading truncated binary prefix)")
                
                if self.b - 1 > 0:
                    r_temp_str = encoded_data[i:i + self.b - 1]
                    for char in r_temp_str:
                        if char not in ('0', '1'):
                            raise ValueError(f"Invalid character in bitstream: '{char}'")
                    r_temp = int(r_temp_str, 2)
                    i += self.b - 1
                else:
                    r_temp = 0
                
                if r_temp < self.cutoff:
                    r = r_temp
                else:
                    # Read 1 more bit
                    if i >= n:
                        raise ValueError("Incomplete remainder (EOF while reading final bit)")
                    
                    next_bit = encoded_data[i]
                    if next_bit not in ('0', '1'):
                        raise ValueError(f"Invalid character in bitstream: '{next_bit}'")
                        
                    r = (r_temp << 1 | int(next_bit)) - self.cutoff
                    i += 1
            
            decoded.append(q * self.m + r)
            
        return decoded
