"""Training script for the cross-entropy method with constant noise."""

from CE_method_with_noise import simulation_CE_const_noise, constant_noisy_cem_no_covariance

if __name__ == "__main__":

    ALPHA = 1.0  # Learning rate
    TOTAL_ITERATIONS = 100000
    RHO = 0.1  # Fraction of vectors that are selected for the next iteration
    NOISE = 4.0  # Constant noise to add

    # Run the simulation

    # simulation_CE_const_noise(
    #     alpha=ALPHA,
    #     iteration_count=TOTAL_ITERATIONS,
    #     rho=RHO,
    #     noise=NOISE,
    #     weight_vector_size=8, # BCTS feature set size
    # )

    constant_noisy_cem_no_covariance(
        iteration_count=TOTAL_ITERATIONS,
        rho=RHO,
        noise=NOISE,
        weight_vector_size=8,  # BCTS feature set size
    )
