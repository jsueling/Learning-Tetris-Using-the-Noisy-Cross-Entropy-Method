"""Training script for the cross-entropy method with constant noise."""

from CE_method_with_noise import constant_noisy_cem_multivariate, constant_noisy_cem_univariate

if __name__ == "__main__":

    ALPHA = 1.0  # Learning rate
    TOTAL_ITERATIONS = 100
    RHO = 0.1  # Fraction of vectors that are selected for the next iteration
    NOISE = 4.0  # Constant noise to add

    for i in range(0, 5):
        constant_noisy_cem_univariate(
            iteration_count=TOTAL_ITERATIONS,
            rho=RHO,
            noise=NOISE,
            weight_vector_size=8,
            seed=i,
            tetromino_randomisation_scheme="uniform"
        )

    for i in range(0, 5):
        constant_noisy_cem_multivariate(
            alpha=ALPHA,
            iteration_count=TOTAL_ITERATIONS,
            rho=RHO,
            noise=NOISE,
            weight_vector_size=8,
            seed=i,
            tetromino_randomisation_scheme="uniform"
        )

    for i in range(5, 10):
        constant_noisy_cem_univariate(
            iteration_count=TOTAL_ITERATIONS,
            rho=RHO,
            noise=NOISE,
            weight_vector_size=8,
            seed=i,
            tetromino_randomisation_scheme="bag"
        )

    for i in range(5, 10):
        constant_noisy_cem_multivariate(
            alpha=ALPHA,
            iteration_count=TOTAL_ITERATIONS,
            rho=RHO,
            noise=NOISE,
            weight_vector_size=8,
            seed=i,
            tetromino_randomisation_scheme="bag"
        )
