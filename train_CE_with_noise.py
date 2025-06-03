from CE_method_with_noise import simulation_CE_const_noise

def main():
    # Parameters
    alpha = 0.8  # Learning rate
    N_iteration = 100  # Number of iterations
    rho = 0.1  # Fraction of vectors that are selected for the next iteration
    noise = 4  # Constant noise to add

    # Run the simulation
    simulation_CE_const_noise(alpha, N_iteration, rho, noise)

if __name__ == "__main__":
    main()