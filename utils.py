import numpy as np
import math
from scipy.stats import t as student_t
import matplotlib.pyplot as plt
from policies import run_experiment



def summarize_queue_log(queue_log, warmup_time):
    times = np.array([t for t, _ in queue_log], dtype=float)
    qs    = np.array([q for _, q in queue_log], dtype=float)

    mask = times >= warmup_time
    if not np.any(mask):
        raise ValueError("Warm-up time too large")
    
    # return a single vector length 3
    return qs[mask].mean(axis=0)





def ci_from_queue_logs(queue_logs, warmup_time, alpha=0.05, rel_floor=1e-12):
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




def time_avg_from_queue_log(queue_log, warmup_time=0.0):
    if not queue_log:
        return [], [], [], []

    #  sorted by time
    queue_log = sorted(queue_log, key=lambda x: x[0])

    area = np.zeros(3) 
    times = []
    avg_q1 = []
    avg_q2 = []
    avg_q3 = []

    t_prev, q_prev = queue_log[0]
    q_prev = np.asarray(q_prev, float)

    for t, q in queue_log[1:]:
        q = np.asarray(q, float)
        dt = t - t_prev
        if dt < 0:
            raise RuntimeError("Non-monotone times in queue_log")

        area += q_prev * dt

        t_prev = t
        q_prev = q

        if t > 0:
            avg = area / t
            if t >= warmup_time:
                times.append(t)
                avg_q1.append(avg[0])
                avg_q2.append(avg[1])
                avg_q3.append(avg[2])

    return times, avg_q1, avg_q2, avg_q3


def plot_time_avg_queue(queue_log, warmup_time=0.0, exp_num=None, policy_num=None):
    times, q1, q2, q3 = time_avg_from_queue_log(queue_log, warmup_time=warmup_time)

    plt.figure()
    plt.plot(times, q1, label="Queue 1")
    plt.plot(times, q2, label="Queue 2")
    plt.plot(times, q3, label="Queue 3")

    plt.xlabel("Time")
    plt.ylabel("Time-average queue length")
    title = "Time-average queue length"
    if policy_num is not None and exp_num is not None:
        title += f" (Policy {policy_num}, Exp {exp_num})"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def ci_1d(samples, alpha=0.05):
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    mean = samples.mean()

    if n < 2:
        return {
            "mean": mean,
            "ci": (mean, mean),
            "halfwidth": 0.0,
            "rel_halfwidth": 0.0,
        }

    s = samples.std(ddof=1)
    if s == 0:
        return {
            "mean": mean,
            "ci": (mean, mean),
            "halfwidth": 0.0,
            "rel_halfwidth": 0.0,
        }

    tcrit = student_t.ppf(1 - alpha/2, n - 1)
    hw = tcrit * s / np.sqrt(n)
    rel_hw = 0.0 if mean == 0 else hw / abs(mean)

    return {
        "mean": mean,
        "ci": (mean - hw, mean + hw),
        "halfwidth": hw,
        "rel_halfwidth": rel_hw,
    }


def plot_single_simulation(policies, exp_num, run_time):
    queue_logs = []

    # NOTE: run_experiment returns queue_lengths_over_time, profit_per_customer, frac_over_30s, avg_queue_len

    # Run single experiment for each policy
    for i in range(len(policies)): 
        pol_results = []
        for j in range(len(exp_num)): 
            pol_results.append(run_experiment(policies[i], exp_num[j], run_time)[0])
        queue_logs.append(pol_results)

    for pol_idx, pol in enumerate(policies):
        for exp_idx, exp in enumerate(exp_num):
            log = queue_logs[pol_idx][exp_idx] 
            plot_time_avg_queue(log, warmup_time=0.0,
                                exp_num=exp, policy_num=pol)




def run_experiments(policies, experiments, run_time=100_000, n_reps=50, warmup_time=4_000):
    results = {}
    for pol in policies:
        results[pol] = {}
        for exp in experiments:
            queue_logs = []
            profits = []
            fracs_over30 = []
            avg_queues = []

            for r in range(n_reps):
                qlog, profit_per_customer, frac_over_30s, avg_queue_len = run_experiment(
                    pol, exp, run_time, warmup_time
                )
                queue_logs.append(qlog)
                profits.append(profit_per_customer)
                fracs_over30.append(frac_over_30s)
                avg_queues.append(avg_queue_len) 

            results[pol][exp] = {
                "queue_logs": queue_logs,
                "profits": np.array(profits),
                "frac_over_30s": np.array(fracs_over30),
                "avg_queues": np.array(avg_queues), 
            }
    
    return results



def print_confidence_intervals(policies, experiments, results, alpha=0.05):
    for pol in policies:
        for exp in experiments:
            res = results[pol][exp]
            profits = res["profits"]
            fracs_over30 = res["frac_over_30s"]
            avg_queues = res["avg_queues"]

            ci_profit = ci_1d(profits, alpha=alpha)
            ci_frac_q1 = ci_1d(fracs_over30[:, 0], alpha=alpha)
            ci_frac_q2 = ci_1d(fracs_over30[:, 1], alpha=alpha)
            ci_frac_q3 = ci_1d(fracs_over30[:, 2], alpha=alpha)
            ci_q1 = ci_1d(avg_queues[:, 0], alpha=alpha)
            ci_q2 = ci_1d(avg_queues[:, 1], alpha=alpha)
            ci_q3 = ci_1d(avg_queues[:, 2], alpha=alpha)

            print(f"\nPolicy {pol}, Experiment {exp}")
            print(f"  Profit per customer:      mean={ci_profit['mean']:.3f}, "
                f"CI={ci_profit['ci']}, rel_hw={ci_profit['rel_halfwidth']:.3%}")
            print(f"  Fraction wait >30s for queue 1:       mean={ci_frac_q1['mean']:.3f}, "
                f"CI={ci_frac_q1['ci']}, rel_hw={ci_frac_q1['rel_halfwidth']:.3%}")
            print(f"  Fraction wait >30s for queue 2:       mean={ci_frac_q2['mean']:.3f}, "
                f"CI={ci_frac_q2['ci']}, rel_hw={ci_frac_q2['rel_halfwidth']:.3%}")
            print(f"  Fraction wait >30s for queue 3:       mean={ci_frac_q3['mean']:.3f}, "
                f"CI={ci_frac_q3['ci']}, rel_hw={ci_frac_q3['rel_halfwidth']:.3%}")
            print(f"  Avg queue 1 length:                   mean={ci_q1['mean']:.3f}, "
                f"CI={ci_q1['ci']}, rel_hw={ci_q1['rel_halfwidth']:.3%}")
            print(f"  Avg queue 2 length:                   mean={ci_q2['mean']:.3f}, "
                f"CI={ci_q2['ci']}, rel_hw={ci_q2['rel_halfwidth']:.3%}")
            print(f"  Avg queue 3 length:                   mean={ci_q3['mean']:.3f}, "
                f"CI={ci_q3['ci']}, rel_hw={ci_q3['rel_halfwidth']:.3%}")
