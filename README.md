# Learning Tetris Using the Noisy Cross-Entropy Method

This project is a fork of [corentinpla/Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method](https://github.com/corentinpla/Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method). It implements and analyses the training of a Tetris-playing agent using the Noisy Cross-Entropy Method (CEM).

The agent learns an optimal weight vector for a set of grid features, aiming to maximise the number of lines cleared. This implementation explores several variations of the CEM algorithm and Tetris environment.

## Key Features

*   **Noisy Cross-Entropy Method**: Implements CEM with constant noise, which helps prevent premature convergence and encourages exploration.
*   **Univariate vs. Multivariate Distributions**: Compares two approaches for sampling weight vectors:
    *   **Univariate**: Each feature weight is sampled independently from a normal distribution.
    *   **Multivariate**: The entire weight vector is sampled from a multivariate normal distribution, capturing correlations between features.
*   **Tetromino Generation Schemes**: Analyses the impact of different piece generation schemes on learning speed and performance:
    *   **Uniform**: Each of the 7 unique Tetrominoes has an equal and independent chance of appearing next.
    *   **Bag (7-Bag)**: Tetrominoes are drawn without replacement from a "bag" containing one of each piece. A new bag is created when the old one is empty.
*   **BCTS Feature Set**: Uses the advanced feature set from Thiery and Scherrer's paper, "Building Controllers for Tetris" (BCTS) [1].
*   **Checkpointing**: Automatically saves and resumes training progress, allowing long-running experiments to be stopped and restarted.
*   **Parallelised Evaluation**: Leverages Python's `multiprocessing` to significantly speed up the evaluation of sampled weight vectors.
*   **Reproducibility**: Includes a `Dockerfile` for a consistent and isolated development environment.

## Project Structure

```
.
├── constant_noisy_cem.py  # Core implementation of the Noisy CEM algorithm
├── tetris_env.py          # Tetris game environment and logic
├── bcts_feature_set.py    # BCTS feature set
├── train_noisy_cem.py     # Main script to run training experiments
├── plot_results.py        # Script to plot and visualise training results
├── collect_tetris_state_samples.py # Utility to collect game data with a trained agent
├── Dockerfile             # Docker configuration for reproducible setup
└── pyproject.toml         # Project dependencies
```

## Setup
1. **Clone the repository:**
    ```sh
    git clone https://github.com/jsueling/Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method.git
    cd Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method
    ```
2.  **Install [Poetry](https://python-poetry.org/docs/#installation).**

3.  **Install dependencies:**
    ```sh
    poetry install
    ```

## Usage
### Docker (Recommended)


1.  **Build the Docker image:**
    ```sh
    docker build -t tetris-cem .
    ```

2.  **Run the container with a volume mount for the output:**
    
    This command starts an interactive shell inside the container and maps the `./out` directory to your local machine so you can access the results.
    ```sh
    docker run -it -v "$(pwd)/out:/app/out" tetris-cem
    ```

### Running the Experiments

1.  **Start Training:**
    
    Execute the training script. It will run experiments for both univariate and multivariate models using both "uniform" and "bag" piece generation schemes across different seeds, saving the results in the `./out` directory.
    ```sh
    poetry run python train_noisy_cem.py
    ```
2. **Display Results:**

    Fetches saved data from `./out` and prints to the terminal.
    ```sh
    poetry run python load_and_display_results.py
    ```

3.  **Plot Results:**
    
    Once training is complete or has produced some data, run the plotting script to visualise the learning curves.
    ```sh
    poetry run python plot_results.py
    ```

## References
[1] [Building Controllers for Tetris](https://inria.hal.science/inria-00418954/document), Christophe Thiery, Bruno Scherrer

## Dependencies

*   `numpy`
*   `scipy`
*   `matplotlib`
*   `tqdm`
*   `imageio`