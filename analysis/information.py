import math

def calculate_information_value(probability: float) -> float:
    """
    Calculate the self-information (or surprisal) of an event.
    
    Self-information measures the amount of information associated with an event.
    It is defined mathematically as I(x) = -log2(P(x)). Events with lower
    probabilities carry more information (are more surprising) when they occur.
    
    Args:
        probability: A float between 0 and 1 representing the event's probability.
        
    Returns:
        The self-information value in bits.
        
    Raises:
        ValueError: If probability is less than 0, greater than 1, or exactly 0
                    (since information for an impossible event is undefined).
    """
    if probability < 0.0 or probability > 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    
    if probability == 0.0:
        raise ValueError("Information value for an impossible event (probability 0) is undefined.")
        
    if probability == 1.0:
        return 0.0
        
    return -math.log2(probability)
