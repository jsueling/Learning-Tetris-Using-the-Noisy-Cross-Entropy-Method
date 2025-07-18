"""This module implements the Cross-Entropy Method (CEM) with constant noise"""

import multiprocessing as mp
import math
import os
import random

import numpy as np
from scipy import stats
from tqdm import tqdm

import tetris_env

def evaluate_sample(input_data):
    """Helper function to evaluate a single sample in parallel."""
    sample, seed, tetromino_randomisation_scheme = input_data
    return tetris_env.simulation(
        sample,
        seed=seed, # Ensure reproducibility across separate processes
        tetromino_randomisation_scheme=tetromino_randomisation_scheme
    )

def constant_noisy_cem_multivariate(
        alpha,
        iteration_count,
        rho,
        noise,
        weight_vector_size=8,
        n_processes=None,
        seed=None,
        tetromino_randomisation_scheme=None
    ):
    """
    Optimises a weight vector which maximises Tetris score using 
    the cross-entropy method with constant noise. Samples vectors from a
    multivariate normal distribution.
    Parameters:
    - alpha: Learning rate for the CMA-ES algorithm.
    - iteration_count: Number of iterations to run the simulation.
    - rho: Fraction of vectors that are selected for the next generation.
    - noise: Constant noise value to add to the covariance matrix.
    - weight_vector_size: Dependent on which feature set is used
    - n_processes: Number of processes to use for parallel evaluation
    - seed: Seed randomness for reproducibility.
    - tetromino_randomisation_scheme: Scheme for randomising tetromino generation.
    """

    if tetromino_randomisation_scheme not in ["uniform", "bag"]:
        raise ValueError("Tetromino randomisation scheme must be either 'uniform' or 'bag'.")
    if seed is None:
        raise ValueError("Seed must be provided for reproducibility.")

    # Seed randomness in main loop
    np.random.seed(seed)
    random.seed(seed)

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    # Initialisation
    mean_0 = np.array([0] * weight_vector_size)
    var_0 = 100
    cov_0 = np.diag([var_0] * weight_vector_size)

    # Ensure output directory exists
    os.makedirs('./out', exist_ok=True)
    checkpoint_file = (
        f'./out/data_{tetromino_randomisation_scheme}_noisy_cem_multivariate_{seed}.npy'
    )

    # Attempt to load previous state if it exists and continue from where
    # the last iteration left off.
    if os.path.exists(checkpoint_file):

        loaded_data = np.load(checkpoint_file, allow_pickle=True).item()

        mean_prev = loaded_data['mean_prev']
        cov_prev = loaded_data['cov_prev']
        elite_mean_avg_scores_log = loaded_data['elite_mean_avg_scores_log']

        best_score_elite_mean = loaded_data['best_score']
        best_scoring_vector_data = {
            "best_elite_vector": loaded_data['best_elite_vector'],
            "best_score": loaded_data['best_score'],
            "best_iteration_index": loaded_data['best_iteration_index']
        }

        # Continue from where the last iteration left off
        last_iteration_index = loaded_data['last_iteration_index']
        start_index = last_iteration_index + 1
    else:
        # If no previous data exists, start fresh
        mean_prev = mean_0
        cov_prev = cov_0
        elite_mean_avg_scores_log = []

        best_score_elite_mean = -np.inf
        best_scoring_vector_data = {}

        start_index = 0

    # The number of sampled vectors per generation
    n_sampled_vectors = 100

    # Create a constant noise matrix along the diagonal
    matrix_noise = np.diag([noise] * weight_vector_size)

    for iteration_index in tqdm(
        range(start_index, iteration_count),
        initial=start_index,
        total=iteration_count
    ):

        # Create the distribution for this generation
        distribution = stats.multivariate_normal(
            mean=mean_prev,
            cov=cov_prev
        )

        sample_vectors = distribution.rvs(
            size=n_sampled_vectors,
            random_state=np.random.default_rng(
                iteration_index * (100 ** 0) + (seed+1) * (100 ** 1)
            )
        )

        # Preprocess the vectors to include seed and tetromino randomisation scheme
        # The seed is converted to base 100 to ensure uniqueness across iterations, samples
        # and experiments since max(seed, iteration_index, sample_index) < 100.
        # The first and second sets of evaluation_samples have seeds separated
        # by a digit in base 100 (100 ** 3).
        sample_vectors_mp_input = [
            (
                vector,
                (seed+1) * (100 ** 3) + iteration_index * (100 ** 1) + sample_index * (100 ** 0),
                tetromino_randomisation_scheme
            ) for sample_index, vector in enumerate(sample_vectors)
        ]

        # Using multiprocessing to evaluate samples in parallel
        with mp.Pool(processes=n_processes) as pool:
            sample_evaluation_scores = pool.map(evaluate_sample, sample_vectors_mp_input)

        # Calculate the top k (rho * N) best vectors
        k = math.floor(n_sampled_vectors * rho)

        # Evaluate and sort the samples based on their scores
        ranked_sample_indices = sorted(
            range(n_sampled_vectors),
            key=lambda sample_index: sample_evaluation_scores[sample_index],
            reverse=True
        )

        # Select the top k indices
        top_k_indices = ranked_sample_indices[:k]
        # Select the top k elite vectors
        elite_vectors = sample_vectors[top_k_indices]

        # New parameter estimation using MLE

        elite_mean_vector = np.mean(elite_vectors, axis=0)

        # Among the best samples, captures individual feature spread on diagonal
        # and inter-feature relationships on the off-diagonal.
        cov_next = np.cov(elite_vectors, rowvar=False)

        mean_next = elite_mean_vector
        # Update the mean and covariance for the next generation
        mean_prev = alpha * mean_next + (1 - alpha) * mean_prev
        # Add constant noise
        cov_prev = (alpha ** 2 * cov_next + (1 - alpha) ** 2 * cov_prev) + matrix_noise

        # Preprocess elite mean vector for parallel evaluation
        elite_mean_vector_mp_input = [
            (
                elite_mean_vector,
                seed * (100 ** 2) + iteration_index * (100 ** 1) + sample_index * (100 ** 0),
                tetromino_randomisation_scheme
            ) for sample_index in range(30)
        ]

        # Run 30 simulations in parallel with the best sample
        with mp.Pool(processes=n_processes) as pool:
            elite_vector_scores = pool.map(evaluate_sample, elite_mean_vector_mp_input)

        # Avg score of 30 Tetris simulations using elite mean vector of the current generation
        avg_score_elite_mean = np.mean(elite_vector_scores)

        if avg_score_elite_mean > best_score_elite_mean:

            best_score_elite_mean = avg_score_elite_mean

            best_scoring_vector_data = {
                'best_elite_vector': elite_mean_vector,
                'best_score': best_score_elite_mean,
                'best_iteration_index': iteration_index
            }

        elite_mean_avg_scores_log.append(avg_score_elite_mean)

        checkpoint_data = {
            'last_iteration_index': iteration_index,
            'mean_prev': mean_prev,
            'cov_prev': cov_prev,
            'elite_mean_avg_scores_log': elite_mean_avg_scores_log,
            **best_scoring_vector_data
        }

        np.save(checkpoint_file, checkpoint_data)

def constant_noisy_cem_univariate(
        iteration_count,
        rho,
        noise,
        weight_vector_size=8,
        n_processes=None,
        seed=None,
        tetromino_randomisation_scheme=None
    ):
    """
    Optimises a weight vector which maximises Tetris score using the
    cross-entropy method with constant noise. Samples vectors from a
    univariate normal distribution (each feature is sampled independently)
    and assumes a learning rate of 1.0. It can be inferred this is the method used
    in Thierry and Scherrer's BCTS paper since the authors mention CMA-ES with constant
    noise as an extension of the CEM algorithm described:
    https://inria.hal.science/inria-00418930/document
    Parameters:
    - iteration_count: Number of iterations to run the simulation.
    - rho: Fraction of vectors that are selected for the next generation.
    - noise: Constant noise value to add to the variance of each feature.
    - weight_vector_size: Dependent on which feature set is used
    - n_processes: Number of processes to use for parallel evaluation
    - seed: Seed randomness for reproducibility.
    - tetromino_randomisation_scheme: Scheme for randomising tetromino generation.
    """

    if tetromino_randomisation_scheme not in ["uniform", "bag"]:
        raise ValueError("Tetromino randomisation scheme must be either 'uniform' or 'bag'.")
    if seed is None:
        raise ValueError("Seed must be provided for reproducibility.")

    # Seed randomness in main loop
    np.random.seed(seed)
    random.seed(seed)

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    # Initialisation
    mean_0 = np.array([0] * weight_vector_size)
    var_0 = np.array([100] * weight_vector_size)

    # Ensure output directory exists
    os.makedirs('./out', exist_ok=True)
    checkpoint_file = (
        f'./out/data_{tetromino_randomisation_scheme}_noisy_cem_univariate_{seed}.npy'
    )

    # Attempt to load previous state if it exists and continue from where
    # the last iteration left off.
    if os.path.exists(checkpoint_file):

        loaded_data = np.load(checkpoint_file, allow_pickle=True).item()

        mean_prev = loaded_data['mean_prev']
        var_prev = loaded_data['var_prev']
        elite_mean_avg_scores_log = loaded_data['elite_mean_avg_scores_log']

        best_score_elite_mean = loaded_data['best_score']
        best_scoring_vector_data = {
            "best_elite_vector": loaded_data['best_elite_vector'],
            "best_score": loaded_data['best_score'],
            "best_iteration_index": loaded_data['best_iteration_index']
        }

        last_iteration_index = loaded_data['last_iteration_index']
        start_index = last_iteration_index + 1
    else:
        # If no previous data exists, start fresh
        mean_prev = mean_0
        var_prev = var_0
        elite_mean_avg_scores_log = []

        # Best score and associated data of elite mean vector across all iterations
        best_score_elite_mean = -np.inf
        best_scoring_vector_data = {}

        start_index = 0

    # The number of sampled vectors per generation
    n_sampled_vectors = 100
    # Create a constant noise vector to be added
    # to the variance of each feature at each iteration
    constant_noise = np.array([noise] * weight_vector_size)

    for iteration_index in tqdm(
        range(start_index, iteration_count),
        initial=start_index,
        total=iteration_count
    ):

        # Sample vectors from a univariate normal distribution
        sample_vectors = np.random.normal(
            loc=mean_prev,
            scale=np.sqrt(var_prev),
            size=(n_sampled_vectors, weight_vector_size)
        )

        # Preprocess the vectors to include seed and tetromino randomisation scheme
        # The seed is converted to base 100 to ensure uniqueness across iterations, samples
        # and experiments since max(seed, iteration_index, sample_index) < 100.
        # The first and second sets of evaluation_samples have seeds separated
        # by a digit in base 100 (100 ** 3).
        sample_vectors_mp_input = [
            (
                vector,
                (seed+1) * (100 ** 3) + iteration_index * (100 ** 1) + sample_index * (100 ** 0),
                tetromino_randomisation_scheme
            ) for sample_index, vector in enumerate(sample_vectors)
        ]

        # Using multiprocessing to evaluate samples in parallel
        with mp.Pool(processes=n_processes) as pool:
            sample_evaluation_scores = pool.map(evaluate_sample, sample_vectors_mp_input)

        # Calculate the top k (rho * N) best vectors
        k = math.floor(n_sampled_vectors * rho)

        # Evaluate and sort the samples based on their scores
        ranked_sample_indices = sorted(
            range(n_sampled_vectors),
            key=lambda sample_index: sample_evaluation_scores[sample_index],
            reverse=True
        )

        # Select the top k indices
        top_k_indices = ranked_sample_indices[:k]
        # Select the top k elite vectors
        elite_vectors = sample_vectors[top_k_indices]

        # New parameter estimation using MLE

        elite_mean_vector = np.mean(elite_vectors, axis=0)

        mean_next = elite_mean_vector
        # σ² ← (variance of the selected vectors) + Zt
        var_next = elite_vectors.var(axis=0, ddof=1) + constant_noise

        # Update the mean and variance for the next generation
        mean_prev = mean_next
        var_prev = var_next

        # Preprocess elite mean vector for parallel evaluation
        elite_mean_vector_mp_input = [
            (
                elite_mean_vector,
                seed * (100 ** 2) + iteration_index * (100 ** 1) + sample_index * (100 ** 0),
                tetromino_randomisation_scheme
            ) for sample_index in range(30)
        ]

        # Run 30 simulations in parallel with the elite mean vector
        with mp.Pool(processes=n_processes) as pool:
            elite_vector_scores = pool.map(evaluate_sample, elite_mean_vector_mp_input)

        # Avg score of 30 Tetris simulations using elite mean vector of the current generation
        avg_score_elite_mean = np.mean(elite_vector_scores)

        if avg_score_elite_mean > best_score_elite_mean:

            best_score_elite_mean = avg_score_elite_mean

            best_scoring_vector_data = {
                'best_elite_vector': elite_mean_vector,
                'best_score': best_score_elite_mean,
                'best_iteration_index': iteration_index
            }

        elite_mean_avg_scores_log.append(avg_score_elite_mean)

        checkpoint_data = {
            'last_iteration_index': iteration_index,
            'mean_prev': mean_prev,
            'var_prev': var_prev,
            'elite_mean_avg_scores_log': elite_mean_avg_scores_log,
            **best_scoring_vector_data
        }

        np.save(checkpoint_file, checkpoint_data)
