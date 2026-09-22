from typing import Optional, Dict

class AdaptiveHuffmanNode:
    """
    A node in the Adaptive Huffman tree (FGK algorithm).
    """
    def __init__(self, weight: int, order: int, symbol: Optional[str] = None, is_nyt: bool = False):
        self.weight = weight
        self.order = order
        self.symbol = symbol
        self.is_nyt = is_nyt
        self.left: Optional['AdaptiveHuffmanNode'] = None
        self.right: Optional['AdaptiveHuffmanNode'] = None
        self.parent: Optional['AdaptiveHuffmanNode'] = None


class AdaptiveHuffmanCodec:
    """
    Adaptive Huffman Codec using the FGK (Faller-Gallager-Knuth) algorithm.
    """
    def __init__(self):
        self.root: Optional[AdaptiveHuffmanNode] = None
        self.nyt: Optional[AdaptiveHuffmanNode] = None
        self.symbols: Dict[str, AdaptiveHuffmanNode] = {}
        self.nodes: Dict[int, AdaptiveHuffmanNode] = {}
        
    def _reset(self):
        """Resets the codec to its initial state with a single NYT root node."""
        initial_order = 10000000
        self.nyt = AdaptiveHuffmanNode(weight=0, order=initial_order, is_nyt=True)
        self.root = self.nyt
        self.symbols = {}
        self.nodes = {initial_order: self.nyt}

    def _validate_tree(self):
        """
        Development method: Verifies core structural invariants of the tree.
        """
        if not self.root:
            return
            
        assert self.root.parent is None, "Root has a parent"
        
        visited = set()
        nyt_count = 0
        
        def traverse(node: AdaptiveHuffmanNode):
            nonlocal nyt_count
            assert id(node) not in visited, "Cycle detected"
            visited.add(id(node))
            
            if node.is_nyt:
                nyt_count += 1
                
            if node.left:
                assert node.left.parent == node, "Left child parent mismatch"
                traverse(node.left)
            if node.right:
                assert node.right.parent == node, "Right child parent mismatch"
                traverse(node.right)
                
            if node.left is None and node.right is None:
                assert node.is_nyt or node.symbol is not None, "Leaf node has no symbol and is not NYT"
            else:
                assert node.left is not None and node.right is not None, "Internal node missing a child"
                assert node.symbol is None and not node.is_nyt, "Internal node has symbol or is NYT"
                
        traverse(self.root)
        
        assert nyt_count == 1, f"Expected exactly one NYT node, found {nyt_count}"
        assert len(visited) == len(self.nodes), "Node dictionary mismatch with tree"
        
        for char, leaf in self.symbols.items():
            assert leaf.symbol == char, "Symbol leaf mismatch"
            assert id(leaf) in visited, "Symbol leaf not in tree"
            
        orders = [node.order for node in self.nodes.values()]
        assert len(orders) == len(set(orders)), "Node orders are not unique"

    def _check_sibling_property(self):
        """
        Development method: Verifies the chosen FGK ordering/weight invariant.
        For nodes ordered by increasing order number, node weights must be nondecreasing.
        """
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.order)
        for i in range(len(sorted_nodes) - 1):
            if sorted_nodes[i].weight > sorted_nodes[i+1].weight:
                raise AssertionError(f"Sibling property violated: Order {sorted_nodes[i].order} has weight {sorted_nodes[i].weight}, but higher order {sorted_nodes[i+1].order} has weight {sorted_nodes[i+1].weight}")

    def _get_code(self, node: AdaptiveHuffmanNode) -> str:
        """Returns the binary code for a given node by traversing up to the root."""
        code = []
        curr = node
        while curr.parent is not None:
            if curr.parent.left == curr:
                code.append('0')
            else:
                code.append('1')
            curr = curr.parent
        return ''.join(reversed(code))

    def _is_ancestor(self, ancestor: AdaptiveHuffmanNode, node: AdaptiveHuffmanNode) -> bool:
        """Checks if `ancestor` is an ancestor of `node`."""
        curr = node.parent
        while curr is not None:
            if curr == ancestor:
                return True
            curr = curr.parent
        return False

    def _is_descendant(self, descendant: AdaptiveHuffmanNode, node: AdaptiveHuffmanNode) -> bool:
        """Checks if `descendant` is a descendant of `node`."""
        return self._is_ancestor(ancestor=node, node=descendant)

    def _find_highest_order_in_block(self, weight: int, current: AdaptiveHuffmanNode) -> Optional[AdaptiveHuffmanNode]:
        """
        Finds the node with the highest order among all nodes with the given weight,
        excluding the current node, its ancestors, and its descendants.
        """
        highest = None
        for node in self.nodes.values():
            if node.weight == weight and node != current:
                if not self._is_ancestor(ancestor=node, node=current) and not self._is_descendant(descendant=node, node=current):
                    if highest is None or node.order > highest.order:
                        highest = node
        return highest

    def _swap_positions(self, n1: AdaptiveHuffmanNode, n2: AdaptiveHuffmanNode):
        """
        Swaps two nodes structurally.
        Node order values are exchanged to maintain the FGK position invariant.
        """
        p1 = n1.parent
        p2 = n2.parent
        
        # Determine if n1 and n2 are siblings
        if p1 == p2 and p1 is not None:
            if p1.left == n1:
                p1.left = n2
                p1.right = n1
            else:
                p1.left = n1
                p1.right = n2
            # Parents don't change
        else:
            # Swap children pointers in parents
            if p1 is not None:
                if p1.left == n1:
                    p1.left = n2
                else:
                    p1.right = n2
            else:
                self.root = n2
                    
            if p2 is not None:
                if p2.left == n2:
                    p2.left = n1
                else:
                    p2.right = n1
            else:
                self.root = n1
                    
            # Swap parents
            n1.parent, n2.parent = p2, p1

        # Treat order as a positional property
        n1.order, n2.order = n2.order, n1.order
        self.nodes[n1.order] = n1
        self.nodes[n2.order] = n2

    def _update_tree(self, node: AdaptiveHuffmanNode):
        """
        FGK algorithm update procedure.
        Finds highest eligible node in weight block, swaps structural positions if needed,
        increments weight, and moves to parent.
        """
        curr = node
        while curr is not None:
            highest = self._find_highest_order_in_block(curr.weight, curr)
            
            if highest is not None and highest.order > curr.order:
                self._swap_positions(curr, highest)
            
            curr.weight += 1
            curr = curr.parent
            
            # Call validation methods during development
            # self._validate_tree()
            # self._check_sibling_property()

    def encode(self, data: str) -> str:
        """Encodes string data using Adaptive Huffman (FGK)."""
        if not data:
            return ""
            
        self._reset()
        encoded = []
        
        for char in data:
            if char in self.symbols:
                # Symbol has been seen before
                leaf = self.symbols[char]
                encoded.append(self._get_code(leaf))
                self._update_tree(leaf)
            else:
                # Symbol is new
                encoded.append(self._get_code(self.nyt))
                
                # Encode symbol as 21-bit fixed-width Unicode
                code_point = ord(char)
                if code_point > 0x10FFFF or (0xD800 <= code_point <= 0xDFFF):
                    raise ValueError(f"Invalid Unicode code point for symbol {char}")
                char_code = format(code_point, '021b')
                encoded.append(char_code)
                
                # Split the NYT node
                old_nyt = self.nyt
                old_nyt.is_nyt = False
                
                # Assign valid unique orders according to FGK (highest to lowest)
                new_leaf_order = old_nyt.order - 1
                new_nyt_order = old_nyt.order - 2
                
                new_nyt = AdaptiveHuffmanNode(weight=0, order=new_nyt_order, is_nyt=True)
                new_leaf = AdaptiveHuffmanNode(weight=0, order=new_leaf_order, symbol=char)
                
                new_nyt.parent = old_nyt
                new_leaf.parent = old_nyt
                
                old_nyt.left = new_nyt
                old_nyt.right = new_leaf
                
                self.nyt = new_nyt
                self.symbols[char] = new_leaf
                
                self.nodes[new_nyt_order] = new_nyt
                self.nodes[new_leaf_order] = new_leaf
                
                self._update_tree(new_leaf)
                
        return "".join(encoded)

    def decode(self, encoded_data: str) -> str:
        """Decodes an Adaptive Huffman (FGK) bitstream."""
        if not encoded_data:
            return ""
            
        self._reset()
        decoded = []
        
        i = 0
        while i < len(encoded_data):
            curr = self.root
            
            while curr.left is not None and curr.right is not None:
                if i >= len(encoded_data):
                    raise ValueError("Incomplete encoded symbol at end of stream")
                    
                bit = encoded_data[i]
                i += 1
                if bit == '0':
                    curr = curr.left
                elif bit == '1':
                    curr = curr.right
                else:
                    raise ValueError(f"Invalid binary character '{bit}' at index {i-1}")
                    
            if curr.is_nyt:
                if i + 21 > len(encoded_data):
                    raise ValueError("Truncated NYT symbol data")
                
                char_bits = encoded_data[i:i+21]
                i += 21
                
                if not all(c in '01' for c in char_bits):
                    raise ValueError("Invalid binary character in NYT data")
                    
                code_point = int(char_bits, 2)
                if code_point > 0x10FFFF or (0xD800 <= code_point <= 0xDFFF):
                    raise ValueError("Invalid Unicode code point in stream")
                char = chr(code_point)
                decoded.append(char)
                
                old_nyt = self.nyt
                old_nyt.is_nyt = False
                
                new_leaf_order = old_nyt.order - 1
                new_nyt_order = old_nyt.order - 2
                
                new_nyt = AdaptiveHuffmanNode(weight=0, order=new_nyt_order, is_nyt=True)
                new_leaf = AdaptiveHuffmanNode(weight=0, order=new_leaf_order, symbol=char)
                
                new_nyt.parent = old_nyt
                new_leaf.parent = old_nyt
                
                old_nyt.left = new_nyt
                old_nyt.right = new_leaf
                
                self.nyt = new_nyt
                self.symbols[char] = new_leaf
                
                self.nodes[new_nyt_order] = new_nyt
                self.nodes[new_leaf_order] = new_leaf
                
                self._update_tree(new_leaf)
                
            else:
                char = curr.symbol
                decoded.append(char)
                self._update_tree(curr)
                
        return "".join(decoded)
