import multiprocessing as mp
import math
import os

import numpy as np
from scipy import stats
from tqdm import tqdm

import tetris_env

def evaluate_sample(sample):
    """Helper function to evaluate a single sample."""
    return tetris_env.simulation(sample)

def simulation_CE_const_noise(
        alpha,
        iteration_count,
        rho,
        noise,
        weight_vector_size=8,
        n_processes=None
    ):
    """
    Optimises a weight vector which maximises Tetris score using 
    the cross-entropy method with constant noise.
    Parameters:
    - alpha: Learning rate for the CMA-ES algorithm.
    - iteration_count: Number of iterations to run the simulation.
    - rho: Fraction of vectors that are selected for the next generation.
    - noise: Constant noise value to add to the covariance matrix.
    - weight_vector_size: Dependent on which feature set is used
    - n_processes: Number of processes to use for parallel evaluation
    """

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    # Initialisation
    var_0 = 100
    mean_0 = [0] * weight_vector_size
    cov_0 = np.diag([var_0] * weight_vector_size)

    mean_prev = np.array(mean_0)
    cov_prev = cov_0

    best_score_elite_mean = -np.inf
    os.makedirs('./out', exist_ok=True)  # Ensure output directory exists

    # The number of sampled vectors per generation
    n_sampled_vectors = 100

    # Create a constant noise matrix along the diagonal
    matrix_noise = np.diag([noise] * weight_vector_size)

    elite_mean_avg_scores_log = []

    for iteration_index in tqdm(range(iteration_count)):

        # Create the distribution for this generation
        distribution = stats.multivariate_normal(
            mean=mean_prev,
            cov=cov_prev
        )

        sample_vectors = distribution.rvs(size=n_sampled_vectors)

        # Using multiprocessing to evaluate samples in parallel
        with mp.Pool(processes=n_processes) as pool:
            sample_evaluation_scores = pool.map(evaluate_sample, sample_vectors)

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

        # Run 30 simulations in parallel with the best sample
        with mp.Pool(processes=n_processes) as pool:
            elite_vector_scores = pool.map(evaluate_sample, [elite_mean_vector for _ in range(30)])

        # Avg score of 30 Tetris simulations using elite mean vector of the current generation
        avg_score_elite_mean = np.mean(elite_vector_scores)

        if avg_score_elite_mean > best_score_elite_mean:

            best_score_elite_mean = avg_score_elite_mean

            best_data = {
                'best_elite_vector': elite_mean_vector,
                'score': avg_score_elite_mean,
                'iteration': iteration_index + 1
            }

            np.save('./out/best_elite_vector_data.npy', best_data)

        elite_mean_avg_scores_log.append(avg_score_elite_mean)

        # Overwrites each iteration, maintaining all previous scores in real time
        np.save(
            './out/simulation_CE_const_noise_scores.npy',
            elite_mean_avg_scores_log
        )

def constant_noisy_cem_no_covariance(
        iteration_count,
        rho,
        noise,
        weight_vector_size=8,
        n_processes=None
    ):
    """
    Optimises a weight vector which maximises Tetris score using the
    cross-entropy method with constant noise. Does not use covariance matrix and
    assumes a learning rate of 1.0 as inferred from the paper since the authors
    mention CMA-ES with constant noise as an extension of the CEM algorithm described:
    https://inria.hal.science/inria-00418930/document
    Parameters:
    - iteration_count: Number of iterations to run the simulation.
    - rho: Fraction of vectors that are selected for the next generation.
    - noise: Constant noise value to add to the variance of each feature.
    - weight_vector_size: Dependent on which feature set is used
    - n_processes: Number of processes to use for parallel evaluation
    """

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    # Initialisation
    var_0 = [100] * weight_vector_size
    mean_0 = [0] * weight_vector_size

    mean_prev = np.array(mean_0)
    var_prev = np.array(var_0)

    best_score_elite_mean = -np.inf
    os.makedirs('./out', exist_ok=True)  # Ensure output directory exists

    # The number of sampled vectors per generation
    n_sampled_vectors = 100
    # Create a constant noise vector to be added
    # to the variance of each feature at each iteration
    constant_noise = np.array([noise] * weight_vector_size)

    elite_mean_avg_scores_log = []

    for iteration_index in tqdm(range(iteration_count)):

        # Sample vectors from a univariate normal distribution
        sample_vectors = np.random.normal(
            loc=mean_prev,
            scale=np.sqrt(var_prev),
            size=(n_sampled_vectors, weight_vector_size)
        )

        # Using multiprocessing to evaluate samples in parallel
        with mp.Pool(processes=n_processes) as pool:
            sample_evaluation_scores = pool.map(evaluate_sample, sample_vectors)

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

        # Run 30 simulations in parallel with the elite mean vector
        with mp.Pool(processes=n_processes) as pool:
            elite_vector_scores = pool.map(evaluate_sample, [elite_mean_vector for _ in range(30)])

        # Avg score of 30 Tetris simulations using elite mean vector of the current generation
        avg_score_elite_mean = np.mean(elite_vector_scores)

        if avg_score_elite_mean > best_score_elite_mean:

            best_score_elite_mean = avg_score_elite_mean

            best_data = {
                'best_elite_vector': elite_mean_vector,
                'score': avg_score_elite_mean,
                'iteration': iteration_index + 1
            }

            np.save('./out/best_elite_vector_data.npy', best_data)

        elite_mean_avg_scores_log.append(avg_score_elite_mean)

        np.save(
            './out/constant_noisy_cem_no_covariance_scores.npy',
            elite_mean_avg_scores_log
        )

def simulation_CE_deacr_noise(alpha, N_iteration,rho,a,b): #alpha : taux d'actualistion 
                                   #N_mean: nombre de simulation par vecteur
                                   #N_iteration : nombre d'iterations
                                   #rho : the fraction of verctors that are selected
                                   #retourne L_plot : le score maximal par itération
                                   #noise : value of the constant noise to add
                                   #a,b : params of the decreasing noise, a=5 , b=100 in the paper

    # Initialisation
    mu0 = [0]*21
    sigma0 = np.diag([100]*21)
    V0 = (mu0, sigma0)
    parameters = [V0]
    t=1

    L_plot=[]
    L_norm=[]
    for j in range (N_iteration):


        # Create the distribution
        distribution = stats.multivariate_normal(parameters[t-1][0], parameters[t-1][1])
        

        # Evaluate each parameter pool
        N = 100
        sample_list = []
        sample_score= []

        for i in range(N):
            
            sample = distribution.rvs() #vecteur de paramètre W


            sample_score.append(tetris_env.simulation(sample))
            sample_list.append(sample)

        # Keeping the rho*N bests vectors
        k=math.floor(N*rho)

        indices=sorted(range(len(sample_score)), key=lambda i: sample_score[i], reverse=True)[:k]
        sample_high = [sample_list[i] for i in indices]
        best_sample=sample_list[indices[0]]


        # New parameter estimation using MLE


        mean = np.mean(sample_high, axis = 0)
        cov =  np.cov(sample_high, rowvar = False)
        res = (mean, cov)

        L_norm.append(np.linalg.norm(cov))
        #add noise 
        noise = max(0, a-N/b)
        matrix_noise = np.diag([noise]*21)

        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                        alpha ** 2 * np.array(res[1]) + (1 - alpha) ** 2 * np.array(parameters[-1][1])+matrix_noise))    

 #calcul de la moyenne du meilleur vecteur sur 30 parties
        L_mean=[sample_score[indices[0]]] #liste des scores des 30 simulations
        for k in range (29):
            L_mean.append(tetris_env.simulation(best_sample))

        print(np.mean(L_mean))
        L_plot.append(L_mean)
        t+=1
        print(L_plot,L_norm,mean)
    return(L_plot, L_norm,mean)


     
