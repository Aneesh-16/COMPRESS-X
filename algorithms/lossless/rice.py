from typing import Iterable

class RiceCodec:
    """
    Rice Coding lossless compression algorithm.
    Rice coding is a special case of Golomb coding where m = 2^k.
    """
    
    def __init__(self, k: int):
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be a non-negative integer.")
        if k < 0:
            raise ValueError("k must be a non-negative integer.")
        self.k = k

    def encode(self, data: Iterable[int]) -> str:
        """
        Encodes an iterable of non-negative integers into a Rice bitstream.
        """
        bits = []
        for n in data:
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError(f"Data values must be non-negative integers. Got {type(n).__name__}: {n}")
            if n < 0:
                raise ValueError(f"Data values must be non-negative integers. Got: {n}")
            
            # Equivalent to n // (2^k)
            q = n >> self.k
            # Equivalent to n % (2^k)
            r = n & ((1 << self.k) - 1)
            
            # Quotient unary: q zeros followed by 1
            bits.append("0" * q + "1")
            
            # Remainder fixed width
            if self.k > 0:
                # Format as fixed width binary string with leading zeros
                bits.append(f"{r:0{self.k}b}")
                
        return "".join(bits)

    def decode(self, encoded_data: str) -> list[int]:
        """
        Decodes a Rice bitstream into a list of non-negative integers.
        """
        if not isinstance(encoded_data, str):
            raise TypeError("encoded_data must be a string.")
            
        result = []
        i = 0
        n_bits = len(encoded_data)
        
        while i < n_bits:
            q = 0
            while i < n_bits and encoded_data[i] == "0":
                q += 1
                i += 1
                
            if i >= n_bits:
                raise ValueError("Incomplete unary quotient at end of stream.")
                
            if encoded_data[i] != "1":
                raise ValueError(f"Invalid character in bitstream: '{encoded_data[i]}'")
                
            # Skip the terminating '1' of the unary quotient
            i += 1
            
            if self.k > 0:
                if i + self.k > n_bits:
                    raise ValueError("Incomplete remainder at end of stream.")
                
                r_str = encoded_data[i:i+self.k]
                for char in r_str:
                    if char not in ("0", "1"):
                        raise ValueError(f"Invalid character in bitstream: '{char}'")
                
                r = int(r_str, 2)
                i += self.k
            else:
                r = 0
                
            value = (q << self.k) | r
            result.append(value)
            
        return result
