import pathlib
import hashlib
import pint.logging
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import logging
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from ui_utils import *
from dataset_util import *
from analysis_util import *

DATA_FOLDER = pathlib.Path("data")
PLOT_DATA_FOLDER = pathlib.Path("plot_data")

class Quantifier(Enum):
    THRESHOLD = "threshold"
    GRAY_CODING = "gray_coding"
    SHA512 = "sha512"

    def __str__(self):
        return self.value

QUANTIFIER = Quantifier.SHA512

class Debiasing(Enum):
    NONE = "none"
    XOR = "xor"
    VON_NEUMANN = "von_neumann"
    SHAKE256 = "shake256"

    def __str__(self):
        return self.value

DEBIASING_METHOD = Debiasing.SHAKE256
DATASET = Dataset.EPTA
VERBOSE = False
PLOT_NORMALIZED = False

class PulsarTimingResiduals:
    def __init__(self, toas, model, residuals):
        self.toas = toas
        self.model = model
        self.residuals = residuals

@with_progress_indicator
def get_pulsar_timing_residuals(pulsar_name):
    # Create a progress bar context
    with tqdm(total=4, desc=f"Analyzing pulsar {pulsar_name}", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        # Step 1: Get parameter file
        print("Getting parameter file")
        par_file = get_par_file(pulsar_name, DATASET)
        pbar.update(1)
        
        # Step 2: Get timing file
        print("Getting timing file")
        tim_file = get_tim_file(pulsar_name, DATASET)
        pbar.update(1)
        
        # Step 3: Generate model
        print("Generating model")
        model = get_model(parfile=par_file)
        pbar.update(1)
        
        # Step 4: Process TOAs and calculate residuals
        print("Processing TOAs and calculating residuals")
        toas = get_TOAs(timfile=tim_file, planets=True)
        residuals = Residuals(toas, model).time_resids.value
        pbar.update(1)
        
    return PulsarTimingResiduals(toas, model, residuals)

# Normalize residuals to [0,1]
def normalize_residuals(pulsar_residuals):
    return (pulsar_residuals - np.min(pulsar_residuals)) / (np.max(pulsar_residuals) - np.min(pulsar_residuals))

# Thresholding method
def apply_threshold_to_residuals(pulsar_residuals, threshold=0.5):
    return (pulsar_residuals > threshold).astype(int)

# Convert integer to Gray code
def int_to_gray(n):
    return n ^ (n >> 1)

# Convert residuals to Gray-coded bitstream
def residuals_to_gray_bits(residuals, num_bits=8):
    max_val = 2**num_bits - 1
    scaled_residuals = np.clip((residuals * max_val).astype(int), 0, max_val)
    gray_coded = np.vectorize(int_to_gray)(scaled_residuals)

    bit_arrays = np.array(
        [list(np.binary_repr(x, width=num_bits)) for x in gray_coded], dtype=int
    )
    return bit_arrays.flatten()

# Expand bitstream using SHA-512 in counter mode
def hash_expand(bitstream, target_size=10**6):
    hex_data = "".join(map(str, bitstream))
    expanded_bits = []
    
    counter = 0
    while len(expanded_bits) < target_size:
        hash_input = (hex_data + str(counter)).encode()
        digest = hashlib.sha512(hash_input).digest()
        expanded_bits.extend(np.unpackbits(np.frombuffer(digest, dtype=np.uint8)))
        counter += 1
    
    return np.array(expanded_bits[:target_size])

def shake256_debiasing(random_bits):
    byte_data = np.packbits(random_bits)
    hasher = hashlib.shake_256()
    hasher.update(byte_data)
    hash_bytes = hasher.digest(len(byte_data))  # Preserve original length
    return np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))

# XOR Debiasing
def xor_debiasing(random_bits):
    return np.bitwise_xor(random_bits[:-1], random_bits[1:])

# Von Neumann Debiasing
def von_neumann_debiasing(random_bits):
    debiased_bits = []
    for i in range(0, len(random_bits) - 1, 2):
        b1, b2 = random_bits[i], random_bits[i + 1]
        if b1 == 0 and b2 == 1:
            debiased_bits.append(0)
        elif b1 == 1 and b2 == 0:
            debiased_bits.append(1)
    return np.array(debiased_bits)

# Save bitstreams
def save_random_bits_to_files(random_bits, pulsar_name):
    output_file_ascii = DATA_FOLDER / f"{pulsar_name}_{DEBIASING_METHOD.value}_{QUANTIFIER.value}_{DATASET.value}.ascii"
    with open(output_file_ascii, "w") as f:
        f.write("".join(map(str, random_bits)))
    print(f"ASCII bitstream saved to {output_file_ascii}")

    output_file_bin = DATA_FOLDER / f"{pulsar_name}_{DEBIASING_METHOD.value}_{QUANTIFIER.value}_{DATASET.value}.bin"
    with open(output_file_bin, "wb") as f:
        byte_array = np.packbits(random_bits)
        f.write(byte_array)
    print(f"Binary bitstream saved to {output_file_bin}")

    return output_file_ascii, output_file_bin

# Generate random numbers
def generate_rng_files(pulsar_name):
    pulsar_residuals = get_pulsar_timing_residuals(pulsar_name).residuals
    normalized_residuals = normalize_residuals(pulsar_residuals)

    # Choose quantification method
    if QUANTIFIER == Quantifier.THRESHOLD:
        random_bits = apply_threshold_to_residuals(normalized_residuals)
    elif QUANTIFIER == Quantifier.GRAY_CODING:
        random_bits = residuals_to_gray_bits(normalized_residuals)
    elif QUANTIFIER == Quantifier.SHA512:
        raw_bits = residuals_to_gray_bits(normalized_residuals)
        random_bits = hash_expand(raw_bits)
    else:
        raise ValueError("Invalid quantification method selected.")

    # Choose debiasing method
    if DEBIASING_METHOD == Debiasing.XOR:
        debiased_bits = xor_debiasing(random_bits)
    elif DEBIASING_METHOD == Debiasing.VON_NEUMANN:
        debiased_bits = von_neumann_debiasing(random_bits)
    elif DEBIASING_METHOD == Debiasing.SHAKE256:
        debiased_bits = shake256_debiasing(random_bits)
    else:
        debiased_bits = random_bits  # No debiasing

    return save_random_bits_to_files(debiased_bits, pulsar_name)

def plot_residuals(pulsar_name, toas, residuals, plot_color):
    """
    Create a publication-quality plot of pulsar timing residuals with both MJD and Year axes
    
    Parameters:
    -----------
    pulsar_name : str
        Name of the pulsar
    toas : pint.toa.TOAs
        TOA object containing the time measurements
    residuals : array
        Timing residuals in microseconds
    """

    if PLOT_NORMALIZED:
        residuals = normalize_residuals(residuals)

    # Get MJD times
    mjd_times = toas.get_mjds().value
    
    # Convert MJD to decimal year
    # Using January 1, 2000 (MJD 51544.5) as reference
    years = 2000.0 + (mjd_times - 51544.5)/365.25
    
    # Create figure with two x-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot residuals vs MJD
    ax1.scatter(mjd_times, residuals, color=plot_color, alpha=0.6, s=20)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Increase font sizes
    fontsize = 18  # Adjust as needed
    ax1.set_xlabel('Time (MJD)', fontsize=fontsize)

    if PLOT_NORMALIZED:
        ax1.set_ylabel('Residuals (normalized)', fontsize=fontsize)
    else:
        ax1.set_ylabel('Residuals (μs)', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax1.tick_params(axis='both', which='minor', labelsize=fontsize - 4)
    
    # Create secondary x-axis with years
    ax2 = ax1.twiny()
    ax2.scatter(years, residuals, alpha=0)  # Invisible scatter plot to set limits
    ax2.set_xlabel('Year', fontsize=fontsize)
    ax2.tick_params(axis='x', which='major', labelsize=fontsize - 2)
    ax2.tick_params(axis='x', which='minor', labelsize=fontsize - 4)

    if PLOT_NORMALIZED:
        plt.axhline(y=0.5, color='red', linestyle='--')
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()

    if PLOT_NORMALIZED:
        save_file = f'{PLOT_DATA_FOLDER}/{pulsar_name}_{DATASET.value}_normalized_residuals.pdf'
    else:
        save_file = f'{PLOT_DATA_FOLDER}/{pulsar_name}_{DATASET.value}_residuals.pdf'

    plt.savefig(save_file, dpi=600, bbox_inches='tight')


def print_info():
    print("Experiment settings:")
    print(f"Dataset = {DATASET}")
    print(f"Quantifier = {QUANTIFIER}")
    print(f"Debiasing Method = {DEBIASING_METHOD}")

# Main function
def main():
    global DATASET
    global QUANTIFIER
    global DEBIASING_METHOD
    global VERBOSE
    global PLOT_NORMALIZED

    print(f"Detected {len(EPTA_PULSARS)} pulsars in EPTA dataset.")
    print(f"Detected {len(NANOGRAV_PULSARS)} pulsars in NANOGrav dataset.")

    parser = argparse.ArgumentParser(description="Generate random bits from pulsar timing residuals.")
    parser.add_argument('command', help="Command to execute", choices=["plot", "list", "rng"])
    parser.add_argument('-l', '--list', help="List all pulsars in dataset", action='store_true')
    parser.add_argument('-i', '--index', help="Index of the pulsar.", default=0, type=int, required=False)
    parser.add_argument('-n', '--name', help="Name of the Pulsar", default="", type=str, required=False)
    parser.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), help="Dataset to use", default=Dataset.NANOGRAV.value, required=False)
    parser.add_argument('-q', '--quantifier', type=Quantifier, choices=list(Quantifier), help="Quantifier to use", default=Quantifier.SHA512.value, required=False)
    parser.add_argument('-dm', '--debiasing-method', type=Debiasing, choices=list(Debiasing), help="Debiasing method to use", default=Debiasing.SHAKE256.value, required=False)
    parser.add_argument('-pc', '--plot-color', type=str, default='navy', help='Matplotlib plot color (for the plot command)', required=False)
    parser.add_argument('-e', '--ent', action='store_true', help="Run the ent tool at the end", required=False)
    parser.add_argument('-me', '--min-entropy', action='store_true', help="Calculate min entropy", required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output', required=False)
    parser.add_argument('-pn', '--plot-normalized', action='store_true', help="Plot the normalized values", required=False)

    args = parser.parse_args()

    QUANTIFIER = args.quantifier
    DEBIASING_METHOD = args.debiasing_method
    VERBOSE = args.verbose
    PLOT_NORMALIZED = args.plot_normalized

    if not VERBOSE:
        pint.logging.setup(level="ERROR")

    DATASET = Dataset(args.dataset)

    if DATASET == Dataset.EPTA:
        pulsar_name = EPTA_PULSARS[args.index]
        pulsars = EPTA_PULSARS
    elif DATASET == Dataset.NANOGRAV:
        pulsar_name = NANOGRAV_PULSARS[args.index]
        pulsars = NANOGRAV_PULSARS

    if args.command == "list":
        print(f"*** Pulsars in dataset {DATASET} ***")
        i = 0
        for pulsar in pulsars:
            print(i, "-", pulsar)
            i += 1

        print("*** done ***")
        return

    if len(args.name) != 0:
        pulsar_name = args.name

    if args.command == "plot":
        print_info()
        residuals = get_pulsar_timing_residuals(pulsar_name)
        plot_residuals(pulsar_name, residuals.toas, residuals.residuals, args.plot_color)
        return

    if args.command == "rng":    
        print_info()
        print(f"Generating files for pulsar {pulsar_name} in dataset {DATASET}")
        output_ascii, output_bin = generate_rng_files(pulsar_name)

        if args.ent:
            run_ent(output_bin)

        if args.min_entropy:
            calculate_min_entropy_from_file(output_ascii)
        
        return
    
    parser.print_usage()

if __name__ == '__main__':
    main()