import math
from collections import Counter

class TunstallCodec:
    """
    Tunstall Coding lossless compression algorithm.
    A variable-to-fixed length compression scheme.
    """

    def __init__(self, dictionary_size: int):
        if not isinstance(dictionary_size, int) or isinstance(dictionary_size, bool):
            raise TypeError("dictionary_size must be an integer.")
        if dictionary_size <= 0:
            raise ValueError("dictionary_size must be strictly positive.")
        self.dictionary_size = dictionary_size
        
        self.phrases = []
        self.phrase_to_code = {}
        self.code_to_phrase = {}
        self.phrase_by_first = {}
        self.width = 0

    def fit(self, data: str):
        """
        Builds the Tunstall phrase dictionary based on the input data probabilities.
        """
        if not isinstance(data, str):
            raise TypeError("Data must be a string.")
            
        if not data:
            self.phrases = []
            self.phrase_to_code = {}
            self.code_to_phrase = {}
            self.phrase_by_first = {}
            self.width = 0
            return

        freqs = Counter(data)
        total = sum(freqs.values())
        probs = {char: count / total for char, count in freqs.items()}
        alphabet = sorted(probs.keys())
        
        K = len(alphabet)
        if K > self.dictionary_size:
            raise ValueError(f"Dictionary size {self.dictionary_size} is too small for alphabet size {K}.")
            
        terminals = [(probs[char], char) for char in alphabet]
        
        if K > 1:
            while True:
                if len(terminals) + K - 1 > self.dictionary_size:
                    break
                    
                # Sort by highest probability first. 
                # Tie-break lexicographically (ascending).
                terminals.sort(key=lambda x: (-x[0], x[1]))
                
                best_prob, best_phrase = terminals.pop(0)
                
                for char in alphabet:
                    new_phrase = best_phrase + char
                    new_prob = best_prob * probs[char]
                    terminals.append((new_prob, new_phrase))
                    
        self.phrases = sorted([phrase for prob, phrase in terminals])
        N = len(self.phrases)
        
        if N <= 1:
            self.width = 0
        else:
            self.width = math.ceil(math.log2(N))
            
        self.phrase_to_code = {}
        self.code_to_phrase = {}
        self.phrase_by_first = {}
        
        for i, phrase in enumerate(self.phrases):
            code = f"{i:0{self.width}b}" if self.width > 0 else ""
            self.phrase_to_code[phrase] = code
            self.code_to_phrase[code] = phrase
            self.phrase_by_first.setdefault(phrase[0], []).append(phrase)

    def encode(self, data: str) -> str:
        """
        Encodes a string into a Tunstall bitstream.
        """
        if not isinstance(data, str):
            raise TypeError("Data must be a string.")
            
        if not data:
            return ""
            
        self.fit(data)
        
        header = []
        # 32-bit original length
        header.append(f"{len(data):032b}")
        
        N = len(self.phrases)
        header.append(f"{N:032b}")
        
        for phrase in self.phrases:
            header.append(f"{len(phrase):032b}")
            for char in phrase:
                header.append(f"{ord(char):021b}")
                
        header_str = "".join(header)
        
        payload = []
        i = 0
        
        while i < len(data):
            matched = False
            candidates = self.phrase_by_first.get(data[i], [])
            for phrase in candidates:
                if data.startswith(phrase, i):
                    payload.append(self.phrase_to_code[phrase])
                    i += len(phrase)
                    matched = True
                    break
                    
            if not matched:
                # End of string suffix that doesn't match a full terminal.
                # Because the dictionary is a complete prefix tree over the alphabet,
                # the suffix must be a prefix of at least one terminal phrase.
                suffix = data[i:]
                for phrase in candidates:
                    if phrase.startswith(suffix):
                        payload.append(self.phrase_to_code[phrase])
                        matched = True
                        break
                
                if not matched:
                    raise ValueError("Invalid suffix: characters not in alphabet.")
                break
                
        return header_str + "".join(payload)

    def decode(self, encoded_data: str) -> str:
        """
        Decodes a Tunstall bitstream back into the original string.
        """
        if not isinstance(encoded_data, str):
            raise TypeError("Encoded data must be a string.")
            
        if not encoded_data:
            return ""
            
        if len(encoded_data) < 64:
            raise ValueError("Truncated bitstream: missing header.")
            
        original_length_str = encoded_data[:32]
        N_str = encoded_data[32:64]
        
        if set(original_length_str) - {'0', '1'} or set(N_str) - {'0', '1'}:
            raise ValueError("Invalid character in bitstream header.")
            
        original_length = int(original_length_str, 2)
        if original_length == 0:
            raise ValueError("Non-empty stream declared original length of 0.")
        if original_length < 0:
            raise ValueError("Original length cannot be negative.")
            
        N = int(N_str, 2)
        if N <= 0:
            raise ValueError("Dictionary must contain at least one phrase.")
            
        if 64 + N * 32 > len(encoded_data):
            raise ValueError("Bitstream too short to contain declared dictionary metadata.")
        
        idx = 64
        phrases = []
        seen_phrases = set()
        
        for _ in range(N):
            if idx + 32 > len(encoded_data):
                raise ValueError("Truncated bitstream while reading phrase length.")
                
            p_len = int(encoded_data[idx:idx+32], 2)
            if p_len <= 0:
                raise ValueError("Phrase length must be strictly positive.")
                
            idx += 32
            
            chars = []
            for _ in range(p_len):
                if idx + 21 > len(encoded_data):
                    raise ValueError("Truncated bitstream while reading phrase characters.")
                char_code = int(encoded_data[idx:idx+21], 2)
                if char_code > 0x10FFFF:
                    raise ValueError(f"Invalid Unicode code point: {char_code}")
                chars.append(chr(char_code))
                idx += 21
                
            phrase = "".join(chars)
            if phrase in seen_phrases:
                raise ValueError(f"Duplicate phrase in dictionary: {phrase}")
            seen_phrases.add(phrase)
            phrases.append(phrase)
            
        if N <= 1:
            width = 0
        else:
            width = math.ceil(math.log2(N))
            
        if N > 2 ** width:
            raise ValueError("Dictionary size N exceeds representable fixed-width code space.")
            
        code_to_phrase = {}
        for i, phrase in enumerate(phrases):
            code = f"{i:0{width}b}" if width > 0 else ""
            code_to_phrase[code] = phrase
            
        payload = encoded_data[idx:]
        if set(payload) - {'0', '1'}:
            raise ValueError("Invalid character in bitstream payload.")
            
        if width > 0:
            if len(payload) % width != 0:
                raise ValueError(f"Payload length {len(payload)} is not a multiple of codeword width {width}.")
                
            decoded_phrases = []
            for i in range(0, len(payload), width):
                code = payload[i:i+width]
                if code not in code_to_phrase:
                    raise ValueError(f"Invalid codeword: {code}")
                decoded_phrases.append(code_to_phrase[code])
                
            decoded_str = "".join(decoded_phrases)
        else:
            if len(payload) > 0:
                raise ValueError("Payload should be empty when width is 0.")
            if N == 1:
                phrase = phrases[0]
                repeats = math.ceil(original_length / len(phrase))
                decoded_str = phrase * repeats
            else:
                decoded_str = ""

        if len(decoded_str) < original_length:
            raise ValueError(f"Decoded string length ({len(decoded_str)}) is shorter than original length ({original_length}).")

        return decoded_str[:original_length]
