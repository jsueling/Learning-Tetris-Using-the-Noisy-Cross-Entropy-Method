import numpy as np

def load_best_weight_sample():
    """Load the best sample with full metadata."""
    data = np.load('./out/best_elite_vector_data.npy', allow_pickle=True).item()
    print(data)

if __name__ == "__main__":
    load_best_weight_sample()
