import numpy as np

np.random.seed(67)

def next_interarrival(cust_type):
    if cust_type == 1:
        return np.random.exponential(1)
    elif cust_type == 2: 
        return np.random.exponential(0.5)
    elif cust_type == 3:
        return np.random.exponential(0.8)
    else:
        raise RuntimeError(f"Invalid customer type in next_interarrival: {cust_type}")
    
def next_service_time(cust_type):
    if cust_type == 1:
        return np.random.exponential(5)
    elif cust_type == 2: 
        return np.random.exponential(8)
    elif cust_type == 3:
        return np.random.exponential(12)
    else:
        raise RuntimeError(f"Invalid customer type in next_service_time: {cust_type}")

def waiting_cost(dt, queues):
    cost = 0
    c = 3  # cost coefficient for each queue type
    assert len(queues) == 3
    for queue in queues:
        cost += dt * len(queue) * c
        c -= 1
    return cost

REVENUE_MAT = [
    [70,  None, None],
    [64,  38,   None],
    [None, 46,  60]
]

def find_soonest_departure_time(departure_times: list[list]):
    assert len(departure_times) == 3
    soonest_departure_time = np.inf
    for server_type in departure_times:
        for server in server_type:
            if server < soonest_departure_time:
                soonest_departure_time = server
    return soonest_departure_time

def find_soonest_departure_idx(departure_times: list[list]):
    assert len(departure_times) == 3
    soonest_departure_time = np.inf
    soonest_departure_idx = [0, 0]
    for server_type_idx in range(len(departure_times)):
        for server_idx in range(len(departure_times[server_type_idx])):
            if departure_times[server_type_idx][server_idx] < soonest_departure_time:
                soonest_departure_time = departure_times[server_type_idx][server_idx]
                soonest_departure_idx = [server_type_idx, server_idx]
    return soonest_departure_idx

def revenue(cust_type, server_type):
    assert (cust_type in (1, 2, 3) and server_type in (1, 2, 3))
    val = REVENUE_MAT[cust_type - 1][server_type - 1]
    if val is None:
        raise RuntimeError(
            f"Invalid revenue combo: cust_type={cust_type}, server_type={server_type}"
        )
    return val

def increment_avg_queue_len(dt, run_length, queues, avg_queue_len):
    assert len(queues) == 3 and len(avg_queue_len) == 3 and run_length >= dt
    for queue_idx in range(len(queues)):
        avg_queue_len[queue_idx] += (dt / run_length) * len(queues[queue_idx])

def get_applicable_server_types(cust_type):
    # return 0-based server-type indices
    if cust_type == 1:
        # type 1 customer → server type 1
        return [0]
    elif cust_type == 2: 
        # type 2 customer → server type 2, then 1
        return [1, 0]
    elif cust_type == 3:
        # type 3 customer → server type 3, then 2
        return [2, 1]
    else:
        raise RuntimeError(f"Invalid customer type in get_applicable_server_types: {cust_type}")

def get_applicable_cust_types(server_type):
    # server_type is 1-based here; return 0-based customer-type indices
    if server_type == 1:
        # server type 1 → customer types 1, 2
        return [0, 1]
    elif server_type == 2: 
        # server type 2 → customer types 2, 3
        return [1, 2]
    elif server_type == 3:
        # server type 3 → customer type 3 only
        return [2]
    else:
        raise RuntimeError(f"Invalid server type in get_applicable_cust_types: {server_type}")

STAFF_PER_EXPERIMENT = [
    [6, 4, 9],
    [7, 4, 9],
    [6, 5, 9],
    [6, 4, 10]
]

def run_policy_1(exp_num, run_length):

    queue_lengths_over_time = []

    staff_levels = STAFF_PER_EXPERIMENT[exp_num - 1]

    # server_statuses[g][s] = 0 if free, else customer type (1,2,3) being served
    server_statuses = []
    for num_staff in staff_levels:
        server_statuses.append([0] * num_staff)

    # departure_times mirrors server_statuses
    departure_times = []
    for num_staff in staff_levels:
        departure_times.append([np.inf] * num_staff)

    # queues[i] = list of arrival times for customer type i+1
    queues = [
        [],
        [],
        []
    ]

    # initial arrival times for each customer type
    next_arrivals = [
        next_interarrival(1),
        next_interarrival(2),
        next_interarrival(3)
    ]

    profit              = 0.0
    num_departed        = 0
    system_time         = 0.0
    num_waited_over_30s = 0
    avg_queue_len       = [0.0, 0.0, 0.0]

    while system_time < run_length:

        queue_lengths_over_time.append((system_time, [len(queues[0]), len(queues[1]), len(queues[2])]))


        next_arrival_time = min(next_arrivals)
        next_departure_time = find_soonest_departure_time(departure_times)

        # don't simulate past run_length
        new_time = min(next_arrival_time, next_departure_time, run_length)
        dt = new_time - system_time
        if dt < 0:
            raise RuntimeError("Negative dt encountered.")
        system_time = new_time

        # accumulate waiting cost and queue stats
        profit -= waiting_cost(dt, queues)
        increment_avg_queue_len(dt, run_length, queues, avg_queue_len)
        

        if system_time >= run_length:
            break

        if next_arrival_time <= next_departure_time:
            #ARRIVAL
            cust_type_idx = next_arrivals.index(next_arrival_time)
            cust_type = cust_type_idx + 1

            applicable_server_types = get_applicable_server_types(cust_type)
            server_is_found = False

            for server_type in applicable_server_types:
                for server_idx in range(len(server_statuses[server_type])):
                    if server_statuses[server_type][server_idx] == 0:
                        # assign immediately
                        server_statuses[server_type][server_idx] = cust_type
                        departure_times[server_type][server_idx] = (
                            system_time + next_service_time(cust_type)
                        )
                        server_is_found = True
                        break
                if server_is_found:
                    break

            if not server_is_found:
                # customer joins the queue for their type
                queues[cust_type - 1].append(system_time)

            # schedule next arrival for this customer type
            next_arrivals[cust_type_idx] = system_time + next_interarrival(cust_type)

        else:
            # DEPARTURE 
            free_server_idx = find_soonest_departure_idx(departure_times)
            g, s = free_server_idx  # g = server type index (0,1,2); s = server index within that type
            free_server_type = g + 1  # convert to 1-based for revenue and routing

            cust_type = server_statuses[g][s]
            if cust_type == 0:
                raise RuntimeError("Departure from idle server encountered.")
            num_departed += 1

            # gain revenue
            profit += revenue(cust_type, free_server_type)

            # server becomes idle
            server_statuses[g][s] = 0
            departure_times[g][s] = np.inf

            # try to pull next waiting customer, if any
            applicable_cust_types = get_applicable_cust_types(free_server_type)

            for cust_type_idx in applicable_cust_types:
                if queues[cust_type_idx]:
                    arrival_time = queues[cust_type_idx].pop(0)
                    if (system_time - arrival_time) > 0.5:  # 0.5 minutes = 30s
                        num_waited_over_30s += 1
                    new_cust_type = cust_type_idx + 1
                    server_statuses[g][s] = new_cust_type
                    departure_times[g][s] = system_time + next_service_time(new_cust_type)
                    break

    # REPORT RESULTS 
    print(f"-------------------Results for experiment {exp_num}:----------------")

    profit_per_customer = profit / num_departed
    print(f"Average profit per customer: {profit_per_customer}")

    frac_over_30s = num_waited_over_30s / num_departed
    print(f"Fraction of customers waiting over 30s: {frac_over_30s}")

    for i, queue_len in enumerate(avg_queue_len, start=1):
        print(f"Average length of queue {i}: {queue_len}")
    print()
    
    return queue_lengths_over_time