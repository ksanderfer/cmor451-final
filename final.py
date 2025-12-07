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

def next_dep_time(service_idx):
    return np.random.exponential(ARRIVAL_TIME_MEANS[service_idx])

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

def get_min_dep(deps:list[list]):
    min = np.inf
    for i in deps:
        for j in i:
            if j < min:
                min = j
                min_idx = (i,j) 
    return (min, min_idx)

results_avgtime = []
results_avgnum = []
results_numserved = []
results_util = []
results_lastdepart = []

results = [results_avgtime, results_avgnum, results_numserved, results_util, results_lastdepart]

num_itrs = 0

def run_sim_policy1(max_time):
    for exp_num in (1,2,3,4):

        num_itrs+=1

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
        server_util = 0
        time_past = 0
        num_over_30 = 0

        while time < END_TIME or n > 0:
            if min(ta) <= get_min_dep(td)[0] and min(ta) <= END_TIME:
                curr_arr = min(ta)
                arr_idx = ta.index(curr_arr)
                total_cost += calculate_cost(curr_arr - time, queues)
                time = curr_arr

                server_available_flag = False
                # Primary policy: check server type j first
                if len(queues[arr_idx])==0:
                    for j in range(server_cust_type[arr_idx]):
                        if server_cust_type[arr_idx][j] == 0: #start serving
                            server_available_flag = True
                            server_cust_type[arr_idx][j] = arr_idx + 1
                            td[arr_idx][j] = next_dep_time(arr_idx) + time
                            break

                    if server_available_flag == False and arr_idx != 1:
                        # Try servr type j - 1
                        for j in range(server_cust_type[arr_idx - 1]):
                            if server_cust_type[arr_idx - 1][j] == 0:
                                server_cust_type[arr_idx - 1][j] = arr_idx + 1 
                                td[arr_idx - 1][j] = next_dep_time(arr_idx) + time
                                server_available_flag = True
                                break
                else:
                    queues[arr_idx].append(time)
                
                running_avg[arr_idx] += max(n[arr_idx]-1, 0) * (curr_arr - time)
                
                arrivals += 1
                n[arr_idx] += 1
                ta[arr_idx] = time + next_arr_time(arr_idx)
                arrival_data.append(time)

            elif get_min_dep(td)[0] < min(ta) and get_min_dep(td)[0] <= END_TIME:
                curr_dep = get_min_dep(td)[0]
                dep_job = get_min_dep(td)[1][0]
                dep_server = get_min_dep(td)[1][1]
                total_cost += calculate_cost(curr_dep - time, queues)


                #Implement policy
                server_cust_type[dep_server][dep_job] = 0
                if queues[dep_job]:
                    if time - queues[dep_job].pop() > 30:
                        num_over_30 +=1
                    server_cust_type[dep_job][dep_server] = dep_job + 1
                elif queues[min(dep_job+1, 2)]:
                    if time - queues[min(dep_job+1, 2)].pop() > 30:
                        num_over_30 += 1 
                    server_cust_type[min(dep_job+1, 2)][dep_server] = dep_job + 1

                running_avg[dep_idx] += max(n[dep_idx]-1, 0) * (curr_dep - time)
                time = curr_dep
                n[dep_idx] -= 1
                departures += 1
                if n[dep_idx] == 0:
                    td[dep_idx] = np.inf
                else:
                    td[dep_idx] = time + next_dep_time(dep_idx)
                departure_data.append(time)

            elif min(ta, td) > END_TIME and n > 0:
                running_avg += max(n-1, 0) * (td - time)
                if n > 0:
                    server_util += 1 * (td - time)
                time = td
                n -= 1
                departures += 1
                if n > 0:
                    td = time + dep_time()
                departure_data.append(time)

            elif min(ta, td) > END_TIME and n == 0:
                time_past = max(time - END_TIME, 0)
                time = END_TIME + time_past

        time_past = max(time - END_TIME, 0.0)
        total_time = time 
        avg_Lq = running_avg / total_time
        util   = server_util / total_time

        # Analyze Results:
        #print(f"---Results for simulation {exp_num+1}---")
        # Avg time in system
        elapsed_time = 0
        for i in range(len(departure_data)):
            elapsed_time += departure_data[i] - arrival_data[i]
        avg_time = elapsed_time / len(departure_data)
        #print(f"Average time in system: {avg_time:.2f}")

        results_avgtime.append(avg_time)

        # Avg num in queue
        #print(f"Average number waiting in line: {avg_Lq:.2f}")
        results_avgnum.append(avg_Lq)

        # Number served per day
        #print(f"Number of customers served: {len(arrival_data)}")
        results_numserved.append(len(arrival_data))

        # Server utilization
        #print(f"Server utilization: {util:.2f}")
        results_util.append(util)

        # Last departure time
        #print(f"Last departure time: {departure_data[len(departure_data)-1]:.2f}")
        results_lastdepart.append(departure_data[len(departure_data)-1])

        #print("---------------------------------------------")

        if num_itrs < MIN_PILOT:
            continue

        all_sub1 = True
        intervals = []
        halflengths = []
        errors = []
        points = []

        alpha_prime = ALPHA  
        conf = 1 - alpha_prime

        for result in [results_avgtime, results_avgnum, results_numserved, results_util, results_lastdepart]:
            n = len(result)
            xbar = float(np.mean(result))
            s    = float(np.std(result, ddof=1))
            tcrit = t.ppf(1 - alpha_prime/2, df=n-1)
            se = s / sqrt(n)
            h  = tcrit * se
            ci = (xbar - h, xbar + h)

            intervals.append(ci)
            halflengths.append(h)

            # relative error check (use abs mean)
            relerr = h / max(abs(xbar), 1e-12)
            errors.append(relerr)
            if relerr > TARGET_REL:
                all_sub1 = False

        if all_sub1:
            print(f"Number of iterations: {num_itrs}")
            print("CIs (95% per metric):")
            for ci in intervals:
                print(ci)
                print("Point: ", (ci[1]+ci[0])/2)
            print("Half-widths:", halflengths)
            print("Errors: ", errors)
            break

