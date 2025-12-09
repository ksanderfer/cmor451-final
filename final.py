from policies import run_experiment
import numpy as np
from utils import plot_time_avg_queue, plot_single_simulation, run_experiments, print_confidence_intervals

policies = [1, 2, 3]
exp_num = [1, 2, 3, 4]
run_time = 100_000
n_reps = 50

if __name__ == "__main__":    
    # Plot average customers per queue to determine warmup interval
    plot_single_simulation(policies, exp_num, run_time)

    # Looks like 4,000 hours is a good warmup time.
    warmup_time = 4_000

    # Run all experiments (100,000 hours, 50 replications)
    results = run_experiments(policies, experiments, run_time=10_000, n_reps=5, warmup_time=warmup_time)

    # Confidence Intervals
    alpha = 0.05 
    print_confidence_intervals(policies, experiments, results, alpha)

