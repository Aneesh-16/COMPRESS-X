from collections import Counter
from typing import Dict, List, Tuple

class ShannonFanoCodec:
    """
    Shannon-Fano lossless compression codec.
    
    This class builds a Shannon-Fano code table based on symbol frequencies
    and provides methods to encode and decode strings.
    """
    def __init__(self) -> None:
        self.frequencies: Dict[str, int] = {}
        self.codes: Dict[str, str] = {}
        self.reverse_codes: Dict[str, str] = {}
        
    def fit(self, data: str) -> 'ShannonFanoCodec':
        """
        Fit the model to the provided data to build the code table.
        
        Calculates symbol frequencies, sorts them descending, and recursively
        builds the Shannon-Fano codes.
        
        Args:
            data: The input string to build the model from.
            
        Returns:
            The fitted ShannonFanoCodec instance.
        """
        if not data:
            self.frequencies = {}
            self.codes = {}
            self.reverse_codes = {}
            return self
            
        self.frequencies = dict(Counter(data))
        
        # Sort by frequency (descending) and then by symbol (ascending) for deterministic ties
        sorted_symbols = sorted(self.frequencies.items(), key=lambda item: (-item[1], item[0]))
        
        # Handle the single unique symbol edge case
        if len(sorted_symbols) == 1:
            symbol = sorted_symbols[0][0]
            self.codes = {symbol: "0"}
            self.reverse_codes = {"0": symbol}
            return self
            
        self.codes = {symbol: "" for symbol, _ in sorted_symbols}
        self._build_tree(sorted_symbols)
        self.reverse_codes = {code: symbol for symbol, code in self.codes.items()}
        return self

    def _build_tree(self, symbols: List[Tuple[str, int]]) -> None:
        """
        Recursively partition symbols into two groups with roughly equal frequency sums.
        
        Args:
            symbols: A list of (symbol, frequency) tuples.
        """
        if len(symbols) <= 1:
            return
            
        total_freq = sum(freq for _, freq in symbols)
        
        # Find the split point that minimizes the difference between left and right sums
        best_diff = float('inf')
        split_idx = 1
        current_sum = 0
        
        for i in range(len(symbols) - 1):
            current_sum += symbols[i][1]
            right_sum = total_freq - current_sum
            diff = abs(current_sum - right_sum)
            
            # Keep the first split that gives the minimum difference for determinism
            if diff < best_diff:
                best_diff = diff
                split_idx = i + 1
                
        # Assign '0' to the left partition and '1' to the right partition
        for i in range(split_idx):
            self.codes[symbols[i][0]] += "0"
            
        for i in range(split_idx, len(symbols)):
            self.codes[symbols[i][0]] += "1"
            
        # Recursive calls for both partitions
        self._build_tree(symbols[:split_idx])
        self._build_tree(symbols[split_idx:])

    def encode(self, data: str) -> str:
        """
        Encode the given string using the fitted code table.
        
        Args:
            data: The input string to encode.
            
        Returns:
            The binary string representation of the encoded data.
            
        Raises:
            ValueError: If a symbol is encountered that wasn't in the training data,
                        or if the codec hasn't been fitted.
        """
        if not data:
            return ""
        if not self.codes:
            raise ValueError("Codec must be fitted before encoding.")
        
        try:
            return "".join(self.codes[char] for char in data)
        except KeyError as e:
            raise ValueError(f"Symbol {e} not found in fitted code table.")

    def decode(self, encoded_data: str) -> str:
        """
        Decode the binary string back to the original text.
        
        Args:
            encoded_data: The binary string to decode.
            
        Returns:
            The decoded original string.
            
        Raises:
            ValueError: If the encoded data contains an invalid sequence,
                        or if the codec hasn't been fitted.
        """
        if not encoded_data:
            return ""
        if not self.reverse_codes:
            raise ValueError("Codec must be fitted before decoding.")
            
        decoded = []
        buffer = ""
        for bit in encoded_data:
            buffer += bit
            if buffer in self.reverse_codes:
                decoded.append(self.reverse_codes[buffer])
                buffer = ""
                
        if buffer:
            raise ValueError("Invalid encoded data: trailing bits found.")
            
        return "".join(decoded)
