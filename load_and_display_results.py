"""Loads and prints data from current training iteration"""

import os
import json
import glob

import numpy as np

def load_and_display_results(directory='./out'):
    """Load and print data from specified directory."""
    pattern_best = os.path.join(directory, '*best*noisy_cem*.npy')
    pattern_mean = os.path.join(directory, '*means*noisy_cem*.npy')
    files_best = glob.glob(pattern_best)
    files_mean = glob.glob(pattern_mean)
    if not files_best and not files_mean:
        print("No files found.")
        return None

    for file in files_best:
        print("=" * 20)
        print(f"file {file}")
        print("=" * 20)
        data = dict(np.load(file, allow_pickle=True).item())
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
        print(json.dumps(data, indent=2))
        print("=" * 20)
        print("=" * 20)
        print()

    for file in files_mean:
        print("=" * 20)
        print(f"file {file}")
        print("=" * 20)
        print(np.load(file, allow_pickle=True))
        print("=" * 20)
        print("=" * 20)
        print()

if __name__ == "__main__":
    load_and_display_results()
