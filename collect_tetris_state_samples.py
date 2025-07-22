"""This script collects state samples from Tetris using an optimised weight vector."""

import multiprocessing as mp
import numpy as np

from tqdm import tqdm

from tetris_env import Tetris, simulation_data_collection, evaluate_best_move

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
    return simulation_data_collection(BEST_BCTS_WEIGHT_VECTOR)

def collect_state_samples():
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
            np.save(f'./out/tetris_state_samples_batch_{iteration_number}.npy', all_samples)
            all_samples = []  # Clear memory
            save_counter += 1

        iteration_number += 1

# This was the best performing weight vector which was obtained from the following run:
# ../out/data_uniform_noisy_cem_multivariate_4.npy

#   "best_elite_vector": [
#     23.26748362686174,
#     -5.31287222716384,
#     18.49261716922195,
#     37.257612578570445,
#     21.01654349243818,
#     22.26255254597566,
#     0.36536045532569705,
#     58.99509032260189
#   ],
#   "best_score": 412892.8,
#   "best_iteration_index": 55

UPDATED_BEST_BCTS_WEIGHT_VECTOR = [
    23.26748362686174,
    -5.31287222716384,
    18.49261716922195,
    37.257612578570445,
    21.01654349243818,
    22.26255254597566,
    0.36536045532569705,
    58.99509032260189
]

class TetrisSample:
    """
    Represents a sample from the Tetris environment.
    n_step_rewards are an approximation of value function of
    a pre-optimised evaluation function.
    """
    def __init__(self, grid, n_step_rewards):
        self.grid = grid
        self.n_step_rewards = n_step_rewards

def collect_n_step_helper(env_copy: Tetris, n_steps, helper_seed):
    """
    Collects a single n-step sample from the Tetris environment.
    """
    np.random.seed(helper_seed)
    rewards = []
    done = False
    while not done and len(rewards) < n_steps:
        # Given the best weight vector, return the best action
        action = evaluate_best_move(
            UPDATED_BEST_BCTS_WEIGHT_VECTOR,
            env_copy.grid,
            env_copy.current_tetromino.type
        )
        done, reward = env_copy.step(action)
        rewards.append(reward)
    # Pad with zeros if the episode ends before n_steps
    return rewards + (n_steps - len(rewards)) * [0]

def collect_n_step_sample(
    env: Tetris,
    n_steps,
    step_count
):
    """
    Samples parallel n-step rewards from the Tetris environment,
    each with different random seeds.
    """
    num_parallel_samples = 10
    with mp.Pool(processes=num_parallel_samples) as pool:
        n_step_rewards = pool.starmap(
            collect_n_step_helper,
            [
                (env.copy(), n_steps, (step_count + i) % RANDOM_SEED_LIMIT)
                for i in range(num_parallel_samples)
            ]
        )
    copy_grid = np.zeros((20, 10), dtype=int)
    np.copyto(copy_grid, env.grid)
    return TetrisSample(
        grid=copy_grid,
        n_step_rewards=n_step_rewards
    )

def collect_n_step_samples(total_sample_target=50000, n_steps=125):
    """
    Collect samples from the Tetris environment,
    using an optimised weight vector as a controller.
    """
    sample_interval = 200
    collected_samples = []
    step_count = 0
    env = Tetris()
    with tqdm(total=total_sample_target, desc="Collecting samples") as pbar:
        while len(collected_samples) < total_sample_target:
            env.reset()
            done = False
            while not done and len(collected_samples) < total_sample_target:
                action = evaluate_best_move(
                    UPDATED_BEST_BCTS_WEIGHT_VECTOR,
                    env.grid,
                    env.current_tetromino.type
                )
                done, _ = env.step(action)
                step_count += 1
                if step_count > 0 and step_count % sample_interval == 0:
                    collected_samples.append(collect_n_step_sample(env, n_steps, step_count))
                    pbar.update()
    np.save(
        f'./out/{short_num(total_sample_target)}_tetris_n_step_samples_{n_steps}_steps.npy',
        collected_samples
    )

def short_num(num):
    """
    Shortens a number for better readability.
    """
    if num >= 1e9:
        return f"{int(num / 1e9)}b"
    elif num >= 1e6:
        return f"{int(num / 1e6)}m"
    elif num >= 1e3:
        return f"{int(num / 1e3)}k"
    else:
        return str(num)

if __name__ == "__main__":
    RANDOM_SEED = 25
    RANDOM_SEED_LIMIT = 2 ** 32
    np.random.seed(RANDOM_SEED)
    # collect_state_samples()
    collect_n_step_samples()
