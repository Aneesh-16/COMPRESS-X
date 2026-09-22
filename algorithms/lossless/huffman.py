import heapq
import itertools
from collections import Counter
from typing import Dict, Optional

class HuffmanNode:
    """
    A node in the Huffman tree.
    
    Can represent either a leaf node (containing a symbol) or an internal node.
    """
    def __init__(self, frequency: int, order_id: int, symbol: Optional[str] = None, 
                 left: Optional['HuffmanNode'] = None, right: Optional['HuffmanNode'] = None):
        # Node construction
        self.frequency = frequency
        self.order_id = order_id
        self.symbol = symbol
        self.left = left
        self.right = right
        
        # Precompute the smallest symbol in this subtree for deterministic tie-breaking.
        # When comparing two nodes with identical frequencies, we fall back to comparing
        # their `min_symbol`. This guarantees stable and reproducible trees regardless of
        # the order in which identical frequencies are processed.
        if symbol is not None:
            self.min_symbol = symbol
        else:
            left_sym = left.min_symbol if left else chr(0x10FFFF)
            right_sym = right.min_symbol if right else chr(0x10FFFF)
            self.min_symbol = min(left_sym, right_sym)
            
    def __lt__(self, other: 'HuffmanNode') -> bool:
        """
        Comparison operator for Priority Queue (heapq).
        """
        # First priority: lowest frequency
        if self.frequency != other.frequency:
            return self.frequency < other.frequency
        # Second priority: lexicographically smallest symbol in subtree
        if self.min_symbol != other.min_symbol:
            return self.min_symbol < other.min_symbol
        # Third priority (final deterministic tie-breaker): unique creation order
        return self.order_id < other.order_id


class HuffmanCodec:
    """
    Huffman lossless compression codec.
    
    This class builds a Huffman code tree based on symbol frequencies
    and provides methods to encode and decode strings.
    """
    def __init__(self) -> None:
        self.frequencies: Dict[str, int] = {}
        self.codes: Dict[str, str] = {}
        self.reverse_codes: Dict[str, str] = {}
        self.tree: Optional[HuffmanNode] = None

    def fit(self, data: str) -> 'HuffmanCodec':
        """
        Fit the model to the provided data.
        
        Builds the frequency table, priority queue, Huffman tree, and code table.
        """
        if not data:
            self.frequencies = {}
            self.codes = {}
            self.reverse_codes = {}
            self.tree = None
            return self

        # 1. Frequency Table
        self.frequencies = dict(Counter(data))
        node_counter = itertools.count()
        
        # Handle the single unique symbol edge case safely
        if len(self.frequencies) == 1:
            symbol = list(self.frequencies.keys())[0]
            self.codes = {symbol: "0"}
            self.reverse_codes = {"0": symbol}
            self.tree = HuffmanNode(frequency=self.frequencies[symbol], order_id=next(node_counter), symbol=symbol)
            return self
            
        # 2. Priority Queue setup
        # Push all symbols as leaf nodes into the min-heap
        pq = []
        for symbol, freq in self.frequencies.items():
            heapq.heappush(pq, HuffmanNode(frequency=freq, order_id=next(node_counter), symbol=symbol))
            
        # 3. Huffman Tree Construction
        while len(pq) > 1:
            # Selecting the two minimum-frequency nodes
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            
            # Combining nodes: Create a new parent node with the sum of frequencies
            parent = HuffmanNode(
                frequency=left.frequency + right.frequency,
                order_id=next(node_counter),
                left=left,
                right=right
            )
            # Push the combined node back into the priority queue
            heapq.heappush(pq, parent)
            
        # The remaining node in the queue is the root of the Huffman tree
        self.tree = pq[0]
        
        # 4. Code Generation
        self.codes = {}
        self._generate_codes(self.tree, "")
        self.reverse_codes = {code: symbol for symbol, code in self.codes.items()}
        return self
        
    def _generate_codes(self, node: HuffmanNode, current_code: str) -> None:
        """
        Assigning binary codes by recursively traversing the tree.
        '0' is appended when going left, '1' when going right.
        """
        if node.symbol is not None:
            self.codes[node.symbol] = current_code
            return
            
        if node.left:
            self._generate_codes(node.left, current_code + "0")
        if node.right:
            self._generate_codes(node.right, current_code + "1")

    def encode(self, data: str) -> str:
        """
        Encode the given string using the fitted code table.
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
        Decode the binary string back to the original text using tree traversal.
        """
        if not encoded_data:
            return ""
        if not self.tree:
            raise ValueError("Codec must be fitted before decoding.")
            
        decoded = []
        current_node = self.tree
        
        # Handle single symbol edge case explicitly
        if len(self.codes) == 1:
            symbol = list(self.codes.keys())[0]
            for bit in encoded_data:
                if bit != "0":
                    raise ValueError(f"Invalid bit '{bit}' for single-symbol code.")
                decoded.append(symbol)
            return "".join(decoded)
        
        # Tree traversal during decoding
        # We start at the root and move left or right based on the bit.
        # When we hit a leaf node (with a symbol), we output the symbol
        # and return back to the root to decode the next character.
        for bit in encoded_data:
            if bit == "0":
                current_node = current_node.left
            elif bit == "1":
                current_node = current_node.right
            else:
                raise ValueError(f"Invalid encoded data: encountered non-binary character '{bit}'.")
                
            if current_node is None:
                raise ValueError("Invalid encoded data: structural error in tree traversal.")
                
            # If we reached a leaf node
            if current_node.symbol is not None:
                decoded.append(current_node.symbol)
                current_node = self.tree # Reset to root for the next symbol
                
        # If we didn't end exactly at the root, the bit sequence is incomplete
        if current_node != self.tree:
            raise ValueError("Invalid encoded data: incomplete sequence at the end.")
            
        return "".join(decoded)
