"""Loads and prints data from current training iteration"""

import os
import json
import glob

import numpy as np

def load_and_display_results(directory='./out'):
    """Load and print data from specified directory."""
    pattern_data = os.path.join(directory, 'data*noisy_cem*.npy')

    files_data = glob.glob(pattern_data)

    if not files_data:
        print("No files found.")
        return

    for file in files_data:
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

if __name__ == "__main__":
    load_and_display_results()
