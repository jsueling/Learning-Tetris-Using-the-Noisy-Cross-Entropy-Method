"""Training script for the cross-entropy method with constant noise."""

from CE_method_with_noise import simulation_CE_const_noise

if __name__ == "__main__":

    ALPHA = 1.0  # Learning rate
    N_ITERATION = 100000
    RHO = 0.1  # Fraction of vectors that are selected for the next iteration
    NOISE = 4.0  # Constant noise to add

    # Run the simulation
    simulation_CE_const_noise(
        alpha=ALPHA,
        N_iteration=N_ITERATION,
        rho=RHO,
        noise=NOISE
    )
