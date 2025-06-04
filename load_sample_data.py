import numpy as np

def load_best_sample_with_metadata():
    """Load the best sample with full metadata."""
    data = np.load('./out/best_sample_with_metadata.npy', allow_pickle=True).item()
    print(data)

if __name__ == "__main__":
    load_best_sample_with_metadata()