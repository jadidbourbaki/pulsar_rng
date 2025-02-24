import numpy as np
from collections import Counter
import os

def calculate_min_entropy_from_file(file_path):
    """
    Read a binary array from a text file and calculate its min-entropy.
    
    The file should contain only 0s and 1s (can be separated by spaces, commas, newlines, etc.)
    
    Args:
        file_path (str): Path to the text file containing binary data
        
    Returns:
        float: The min-entropy value
    """
    # Read the file
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        raise IOError(f"Error reading file: {e}")
    
    # Extract binary digits (ignore whitespace and other characters)
    binary_array = []
    for char in content:
        if char == '0':
            binary_array.append(0)
        elif char == '1':
            binary_array.append(1)
    
    # Validate that we have data
    if len(binary_array) == 0:
        raise ValueError("No binary data (0s and 1s) found in the file")
    
    # Count occurrences of each value
    counts = Counter(binary_array)
    
    # Calculate probabilities
    n = len(binary_array)
    probabilities = {value: count / n for value, count in counts.items()}
    
    # Find the maximum probability
    p_max = max(probabilities.values())
    
    # Calculate min-entropy
    min_entropy = -np.log2(p_max)
    
    print(f"Min entropy {min_entropy}, Length: {len(binary_array)}, Counts {counts}")

def run_ent(output_bin):
    os.system(f"ent {output_bin}")

