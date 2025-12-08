import numpy as np
from math import sqrt
from scipy.stats import t

ALPHA = 0.05                # 95% per-metric
MIN_PILOT = 40              # don't evaluate until we have enough runs
TARGET_REL = 0.01           # 1% relative error

END_TIME = 1_000_000 # minutse
NP_SEED = 67

ARRIVAL_TIME_MEANS = [1, .5, .8] # mean interarrival in mins for cust type i

SERVICE_TIME_MEANS = [5, 8, 12] # mean service time in mins for server type i 

STAFFING_LEVELS = {
    1:[6,4,9],
    2:[7,4,9],
    3:[6,5,9],
    4:[6,4,10],
} # experiment num : staff levels 

# c_j := server j
WAITING_COSTS = [3, 2, 1]

# r_{ij} := revenue for cust of type i served by server type j
REVENUE = [
    [70, None, None],
    [64, 38, None],
    [None, 46, 60]
]

np.random.seed(NP_SEED)

#ALL USE INDEX + 1
def next_arr_time(queue_idx):
    return np.random.exponential(ARRIVAL_TIME_MEANS[queue_idx])

def next_dep_time(cust_type_idx):
    return np.random.exponential(SERVICE_TIME_MEANS[cust_type_idx])


def get_wait_cost(i):
    return WAITING_COSTS[i]

def get_revenue(i,j):
    return REVENUE[i][j]

def calculate_cost(dt, queues):
    cost = 0
    for q in range(len(queues)):
        for _ in range(len(queues[q])):
            cost += get_wait_cost(q) * dt 
    return cost

def get_min_dep(deps):
    min_time = np.inf
    min_indices = (None, None)
    for g, row in enumerate(deps):
        for s, t_dep in enumerate(row):
            if t_dep < min_time:
                min_time = t_dep
                min_indices = (g, s)
    return min_time, min_indices


results_avgtime = []
results_avgnum = []
results_numserved = []
results_util = []
results_lastdepart = []

results = [results_avgtime, results_avgnum, results_numserved, results_util, results_lastdepart]

num_itrs = 0

def run_sim_policy1(max_time):
    for exp_num in (1,2,3,4):


        total_cost = 0

        time = 0
        arrivals = 0
        departures = 0 
        n = [0,0,0]               # num in system

        ta = []
        for i in range(3):
            ta.append(next_arr_time(i)) # list of first arrival for each cust type
            
        td = [
            [np.inf]*STAFFING_LEVELS[exp_num][0],
            [np.inf]*STAFFING_LEVELS[exp_num][1],
            [np.inf]*STAFFING_LEVELS[exp_num][2]
        ]   # list of first dep for each type; since none are here, all inf
        arrival_data = [
            [[]]*STAFFING_LEVELS[exp_num][0],
            [[]]*STAFFING_LEVELS[exp_num][1],
            [[]]*STAFFING_LEVELS[exp_num][2]
        ]
        
        server_cust_type = [
            [0]*STAFFING_LEVELS[exp_num][0],
            [0]*STAFFING_LEVELS[exp_num][1],
            [0]*STAFFING_LEVELS[exp_num][2]
        ] # i = server type; j = {0 if server not busy; else n where n is job type} 


        queues = [
            [], # customer type 1
            [], # customer type 2
            [], # customer type 3
        ]
        running_avg = [0,0,0]
        num_over_30 = 0
        revenue = 0

        while time < END_TIME or min(n) > 0:
            if min(ta) <= get_min_dep(td)[0] and min(ta) <= END_TIME:
                curr_arr = min(ta)
                arr_idx = ta.index(curr_arr)
                total_cost += calculate_cost(curr_arr - time, queues)
                for i in (0,1,2):
                    running_avg[i] += len(queues[i]) * (curr_arr - time)
                time = curr_arr

                server_available_flag = False
                # Primary policy: check server type j first
                if len(queues[arr_idx])==0:
                    for j in range(len(server_cust_type[arr_idx])):
                        if server_cust_type[arr_idx][j] == 0: #start serving
                            server_available_flag = True
                            server_cust_type[arr_idx][j] = arr_idx + 1
                            td[arr_idx][j] = next_dep_time(arr_idx) + time
                            break

                    if server_available_flag == False and arr_idx != 1:
                        # Try servr type j - 1
                        for j in range(len(server_cust_type[arr_idx - 1])):
                            if server_cust_type[arr_idx - 1][j] == 0:
                                server_cust_type[arr_idx - 1][j] = arr_idx + 1 
                                td[arr_idx - 1][j] = next_dep_time(arr_idx) + time
                                server_available_flag = True
                                break
                else:
                    queues[arr_idx].append(time)
                
                
                
                arrivals += 1
                n[arr_idx] += 1
                ta[arr_idx] = time + next_arr_time(arr_idx)
                arrival_data.append(time)

            elif get_min_dep(td)[0] < min(ta) and get_min_dep(td)[0] <= END_TIME:
                curr_dep = get_min_dep(td)[0]
                dep_job = get_min_dep(td)[1][0]
                dep_server = get_min_dep(td)[1][1]
                total_cost += calculate_cost(curr_dep - time, queues)
                for i in (0,1,2):
                    running_avg[i] += len(queues[i]) * (curr_dep - time)


                #Implement policy
                server_cust_type[dep_server][dep_job] = 0
                td[dep_server][dep_job] = np.inf
                if queues[dep_job]:
                    if time - queues[dep_job].pop() > 30:
                        num_over_30 +=1
                    server_cust_type[dep_job][dep_server] = dep_job + 1
                    td[dep_job][dep_server] = next_dep_time(dep_server)
                elif queues[min(dep_job+1, 2)]:
                    if time - queues[min(dep_job+1, 2)].pop() > 30:
                        num_over_30 += 1 
                    server_cust_type[min(dep_job+1, 2)][dep_server] = dep_job + 1
                    td[min(dep_job+1, 2)][dep_server] = next_dep_time(dep_server)

                
                time = curr_dep
                departures += 1
                revenue += get_revenue(dep_job, dep_server)

            elif min(ta) > END_TIME and get_min_dep(td)[0] > END_TIME:
                for i in (0,1,2):
                    running_avg[i] += len(queues[i]) * (END_TIME - time)
                time = END_TIME
                departures += 1

        

        # Analyze Results:
        
        print(f"------------------Results for Experiment Number {exp_num}:--------------------")
        profit = revenue - total_cost
        avg_profit = profit / departures
        print(f"Avg profit per customer: {avg_profit}")
        
        frac_wait_30 = num_over_30 / departures
        print(f"Fraction of customers who wait more than 30s: {frac_wait_30} ")
        
        for avg in range(len(running_avg)):
            print(f"Average length of queue {avg}: {running_avg[avg]}")
        

run_sim_policy1(100)