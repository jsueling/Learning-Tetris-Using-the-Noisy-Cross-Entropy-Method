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

def simulation_CE_const_noise(alpha, N_iteration,rho,noise, n_processes=None): #alpha : taux d'actualistion 
                               #N_iteration : nombre d'iterations
                               #rho : the fraction of verctors that are selected
                               #noise : value of the constant noise to add 

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    weight_vector_size = 8  # Size of the weight vector for Tetris feature set

    # Initialisation
    mu0 = [0] * weight_vector_size
    sigma0 = np.diag([100] * weight_vector_size)
    V0 = (mu0, sigma0)
    parameters = [V0]

    # L_plot = []

    highest_avg_score = -np.inf # Track best performing sample across iterations
    os.makedirs('./out', exist_ok=True)  # Ensure output directory exists

    for i in tqdm(range(N_iteration)):


        # Create the distribution
        distribution = stats.multivariate_normal(parameters[i][0], parameters[i][1])

        # Evaluate each parameter pool
        N = 100
        sample_list = [distribution.rvs() for _ in range(N)]

        # Using multiprocessing to evaluate samples in parallel
        with mp.Pool(processes=n_processes) as pool:
            sample_score = pool.map(evaluate_sample, sample_list)

        # Keeping the rho*N bests vectors
        k=math.floor(N*rho)

        indices=sorted(range(len(sample_score)), key=lambda x: sample_score[x], reverse=True)[:k]
        sample_high = [sample_list[x] for x in indices]
        best_sample=sample_list[indices[0]]


        # New parameter estimation using MLE

        # Element-wise addition of vectors, then element-wise division by the number of vectors
        mean = np.mean(sample_high, axis = 0) # Avg of best vectors

        cov =  np.cov(sample_high, rowvar = False)
        res = (mean, cov)


        #add noise

        matrix_noise = np.diag([noise] * weight_vector_size)

        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                        alpha ** 2 * np.array(res[1]) + (1 - alpha) ** 2 * np.array(parameters[-1][1]) + matrix_noise))

        # Run 30 simulations in parallel with the best sample
        with mp.Pool(processes=n_processes) as pool:
            L_mean = pool.map(evaluate_sample, [best_sample for _ in range(30)])

        # Avg score of 30 simulations using the 1st best-scoring vector of the current generation
        avg_score_best_sample = np.mean(L_mean)
        # print(avg_score_best_sample)

        if avg_score_best_sample > highest_avg_score:

            highest_avg_score = avg_score_best_sample

            best_data = {
                'sample': best_sample,
                'score': highest_avg_score,
                'iteration': i+1
            }

            np.save('./out/best_sample_with_metadata.npy', best_data)

        # L_plot.append(L_mean)

        # L_plot is a list of lists containing the scores of the 30 simulations
        # of the best performing vector for each iteration
        # mean is an element-wise avg of the best sample vectors in the current iteration
        # print(L_plot, mean)

    # return L_plot, mean



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


     
