import math
from collections import Counter
from typing import Any, Dict, Iterable, TypedDict

class EntropyResult(TypedDict):
    entropy: float
    total_symbols: int
    unique_symbols: int
    frequencies: Dict[Any, int]
    probabilities: Dict[Any, float]

def calculate_entropy(data: Iterable[Any]) -> EntropyResult:
    """
    Calculate the Shannon entropy of the given data.
    
    The function computes the probability of each unique symbol in the data
    and uses the formula H(X) = -Σ p(x) log2(p(x)) to find the entropy.
    
    Args:
        data: An iterable of hashable symbols (e.g., string, bytes, list).
        
    Returns:
        EntropyResult containing the calculated entropy, total symbol count,
        number of unique symbols, and dictionaries of frequencies and probabilities.
    """
    frequencies = dict(Counter(data))
    total_symbols = sum(frequencies.values())
    
    # Handle empty input safely
    if total_symbols == 0:
        return {
            "entropy": 0.0,
            "total_symbols": 0,
            "unique_symbols": 0,
            "frequencies": {},
            "probabilities": {}
        }
    
    probabilities = {}
    entropy = 0.0
    
    for symbol, count in frequencies.items():
        prob = count / total_symbols
        probabilities[symbol] = prob
        # H(X) = -Σ p(x) log2(p(x))
        entropy -= prob * math.log2(prob)
        
    return {
        "entropy": entropy,
        "total_symbols": total_symbols,
        "unique_symbols": len(frequencies),
        "frequencies": frequencies,
        "probabilities": probabilities
    }
