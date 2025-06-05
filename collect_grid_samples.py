import multiprocessing as mp
import numpy as np

import tetris_env

# 'score': 37156.666666666664, 'iteration': 21
BEST_BCTS_WEIGHT_VECTOR = [
    -19.2856487, -2.03977975, 10.05600815,
    21.53822127, 7.6310916, 8.13881565,
    0.47503094, 31.50869033
]

def collect_single_batch(_):
    """
    Collects a single batch of samples from the Tetris environment.
    """
    return tetris_env.simulation_data_collection(BEST_BCTS_WEIGHT_VECTOR)

def collect_grid_samples():
    """
    Collects sample data (flattened grid + one-hot encoded piece)
    from Tetris simulations using multiprocessing.
    """

    n_processes = mp.cpu_count() - 1
    all_samples = []

    save_counter = 0
    iteration_number = 0
    while save_counter < 10:

        with mp.Pool(processes=n_processes) as pool:
            batch_simulation_samples = pool.map(
                collect_single_batch,
                range(n_processes)
            )

        for simulation_samples in batch_simulation_samples:
            for single_grid_sample in simulation_samples:
                all_samples.append(single_grid_sample)

        if len(all_samples) > 5000:
            np.save(f'./out/vae_samples_batch_{iteration_number}.npy', np.array(all_samples))
            all_samples = []  # Clear memory
            save_counter += 1

        iteration_number += 1

if __name__ == "__main__":
    collect_grid_samples()
