import math
from typing import Any, Iterable, TypedDict
from analysis.entropy import calculate_entropy

class RedundancyResult(TypedDict):
    entropy: float
    maximum_entropy: float
    redundancy: float
    total_symbols: int
    unique_symbols: int

def calculate_redundancy(data: Iterable[Any]) -> RedundancyResult:
    """
    Calculate the relative data redundancy of the given data.
    
    Uses the previously implemented Shannon entropy to find the redundancy:
    H_max = log2(N)
    R = 1 - H(X) / H_max
    where N is the number of unique symbols and H(X) is the entropy.
    
    Args:
        data: An iterable of hashable symbols (e.g., string, bytes, list).
        
    Returns:
        RedundancyResult containing the actual entropy, maximum possible entropy,
        calculated relative redundancy, and symbol counts.
    """
    entropy_result = calculate_entropy(data)
    
    entropy = entropy_result["entropy"]
    total_symbols = entropy_result["total_symbols"]
    unique_symbols = entropy_result["unique_symbols"]
    
    # Handle empty input or single unique symbol safely to avoid division by zero
    if unique_symbols <= 1:
        return {
            "entropy": 0.0,
            "maximum_entropy": 0.0,
            "redundancy": 0.0,
            "total_symbols": total_symbols,
            "unique_symbols": unique_symbols
        }
    
    maximum_entropy = math.log2(unique_symbols)
    redundancy = 1.0 - (entropy / maximum_entropy)
    
    return {
        "entropy": entropy,
        "maximum_entropy": maximum_entropy,
        "redundancy": redundancy,
        "total_symbols": total_symbols,
        "unique_symbols": unique_symbols
    }
