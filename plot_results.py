"""Plots results of the noisy cross-entropy method across different agent variants."""
import os
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from collect_tetris_state_samples import short_num

directory = '../out'

univariate_uniform = os.path.join(directory, '*uniform*univariate*.npy')
multivariate_uniform = os.path.join(directory, '*uniform*multivariate*.npy')
univariate_bag = os.path.join(directory, '*bag*univariate*.npy')
multivariate_bag = os.path.join(directory, '*bag*multivariate*.npy')

files_uu = glob.glob(univariate_uniform)
files_mu = glob.glob(multivariate_uniform)
files_ub = glob.glob(univariate_bag)
files_mb = glob.glob(multivariate_bag)

uniform_runs = defaultdict(list)
bag_runs = defaultdict(list)

for f in files_uu:
    data = np.load(f, allow_pickle=True).item()
    uniform_runs["univariate_uniform"].append(data["elite_mean_avg_scores_log"])
for f in files_mu:
    data = np.load(f, allow_pickle=True).item()
    uniform_runs["multivariate_uniform"].append(data["elite_mean_avg_scores_log"])

plt.figure(figsize=(10, 6))

# for f in sorted(files_ub):
#     data = np.load(f, allow_pickle=True).item()
#     iterations = np.arange(len(data["elite_mean_avg_scores_log"]))
#     plt.plot(
#         iterations,
#         data["elite_mean_avg_scores_log"],
#         alpha=0.6,
#         label=f"univariate bag seed {f[-5]}"
#     )

# for f in sorted(files_mb):
#     data = np.load(f, allow_pickle=True).item()
#     iterations = np.arange(len(data["elite_mean_avg_scores_log"]))
#     plt.plot(
#         iterations,
#         data["elite_mean_avg_scores_log"],
#         alpha=0.6,
#         label=f"multivariate bag seed {f[-5]}"
#     )

for agent_type, runs in uniform_runs.items():
    min_len = min(len(run) for run in runs)
    truncated = np.array([run[:min_len] for run in runs])
    mean_scores = np.mean(truncated, axis=0)
    sem_scores = np.std(truncated, axis=0) / np.sqrt(truncated.shape[0])
    iterations = np.arange(min_len)
    plt.plot(iterations, mean_scores, label=agent_type)
    plt.fill_between(iterations, mean_scores - sem_scores, mean_scores + sem_scores, alpha=0.2)

# for agent_type, runs in bag_runs.items():
#     if not runs:
#         continue
#     for i, run in enumerate(runs):
#         iterations = np.arange(len(run))
#         plt.plot(iterations, run, alpha=0.6, label=f"{agent_type} seed {i+1}")
plt.xticks(np.arange(0, 5))

yticks = plt.yticks()[0]
ytick_labels = [short_num(y) for y in yticks]
plt.yticks(yticks, ytick_labels)
plt.ylim(bottom=0)
plt.xlim(left=0)

plt.xlabel('Iteration')
plt.ylabel('Avg lines cleared by elite mean vector')
# plt.title('Elite Mean Avg Scores by Agent Type')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
