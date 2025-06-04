from CE_method_with_noise import simulation_CE_const_noise

import tetris_env

def main():
    # Parameters
    alpha = 0.8  # Learning rate
    N_iteration = 100  # Number of iterations
    rho = 0.1  # Fraction of vectors that are selected for the next iteration
    noise = 4  # Constant noise to add

    # BCTS_best_vector = [-6.24158494, -1.74254374, 8.42026978, 13.7807557, 1.47415752, 5.03357062, 3.26164688, 11.38943883]
    # tetris_env.simulation_gif(BCTS_m_best_vector, num_moves=100)

    # Run the simulation
    simulation_CE_const_noise(alpha, N_iteration, rho, noise)

if __name__ == "__main__":
    main()
