import numpy as np
import math
from scipy.stats import t as student_t
import matplotlib.pyplot as plt



def summarize_queue_log(queue_log, warmup_time):
    times = np.array([t for t, _ in queue_log], dtype=float)
    qs    = np.array([q for _, q in queue_log], dtype=float)

    mask = times >= warmup_time
    if not np.any(mask):
        raise ValueError("Warm-up time too large")
    
    # return a single vector length 3
    return qs[mask].mean(axis=0)





def ci_from_queue_logs(queue_logs, warmup_time, alpha=0.05, rel_floor=1e-12):
    import math
    import numpy as np
    from scipy.stats import t as student_t

    n_policies = len(queue_logs)
    results = {}

    for pol_idx in range(n_policies):
        reps = queue_logs[pol_idx]
        n = len(reps)

        rep_metrics = np.array(
            [summarize_queue_log(run_log, warmup_time) for run_log in reps]
        )  # shape = (n,3)

        means = rep_metrics.mean(axis=0)
        s2    = rep_metrics.var(axis=0, ddof=1)
        s     = np.sqrt(s2)
        tcrit = student_t.ppf(1 - alpha/2, n - 1)

        policy_result = {}
        for q_idx, q_name in enumerate(["q0", "q1", "q2"]):
            delta = tcrit * s[q_idx] / math.sqrt(n)
            mean_q = means[q_idx]
            ci_low = mean_q - delta
            ci_high = mean_q + delta
            rel_hw = delta / max(abs(mean_q), rel_floor)

            policy_result[q_name] = {
                "mean": float(mean_q),
                "ci": (float(ci_low), float(ci_high)),
                "rel_halfwidth": float(rel_hw),
                "variance": float(s2[q_idx]),
                "delta": float(delta),
            }

        results[pol_idx] = policy_result

    return results



def plot_queue_lengths(queue_log, warmup_time=0.0, exp_num="", policy_num=""):
    # Extract arrays
    times = np.array([t for (t, _) in queue_log], dtype=float)
    qs    = np.array([q for (_, q) in queue_log], dtype=float)   # shape (N,3)

    # Apply warm-up filter
    mask = (times >= warmup_time)

    times = times[mask]
    qs    = qs[mask]

    # Split into the 3 queues
    q1 = qs[:, 0]
    q2 = qs[:, 1]
    q3 = qs[:, 2]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(times, q1, label='Queue 1')
    plt.plot(times, q2, label='Queue 2')
    plt.plot(times, q3, label='Queue 3')

    plt.xlabel('Time')
    plt.ylabel('Queue length')
    plt.title(f'Policy {policy_num}: Queue lengths over time (experiment {exp_num}), warm-up = {warmup_time}')
    plt.legend()
    plt.grid(True)
    plt.show()




