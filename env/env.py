import numpy as np
from .graph import generate_grid, get_shortest_path, remove_node, shortest_hop_distance


class IIoTNetwork:

    def __init__(
        self,
        N,
        M,
        F_D,
        F_E,
        B_E,
        P_E,
        coverage,
        sigma2,
        R_E2E,
        lambda_I,
        alpha,
        beta,
        recv=3,
        seed=None,
    ):
        self.N = N  # Number of edge servers
        self.M = M  # Number of devices
        self.F_D = F_D  # Local device computation capacity
        self.F_E = F_E  # Edge server computation capacity
        self.B_E = B_E  # Bandwidth per device
        self.P_E = P_E  # Transmission power
        self.coverage = coverage  # edge server coverage
        self.sigma2 = sigma2  # Noise power
        self.R_E2E = R_E2E  # Wired connection rate
        self.lambda_I = lambda_I  # Interruption sensitivity
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.recv = recv

        self.random_state = np.random.RandomState(seed)

        self.state_dim = 3 + self.N + self.N + 1 + self.N
        self.action_dim = self.N + 1

    def init_topology(self):
        # generate hexagonal grid topology
        self.adjacency_list, self.positions = generate_grid(self.N)

        # randomly assign devices to edge servers
        # self.device_assignments = self.random_state.randint(0, self.N, self.M)
        self.device_assignments = self.random_state.randint(0, self.N, self.M)
        self.shortest_distances = shortest_hop_distance(self.adjacency_list)

    def update_device_assignments(self):
        self.device_assignments = self.random_state.randint(0, self.N, self.M)

    def update_channel(self):
        self.device_distances = self.random_state.rand(self.M) * self.coverage

        # Path loss model in dB
        PL_dB = 128.1 + 37.6 * np.log10(self.device_distances / 1000)

        # Convert path loss from dB to linear scale
        PL_linear = 10 ** (-PL_dB / 10)

        # Convert P_E (43 dBm) to Watts
        P_E_Watts = 10 ** ((self.P_E - 30) / 10)  # 43 dBm → 19.95 W

        # Convert sigma^2 (-173 dBm) to Watts
        sigma2_Watts = 10 ** ((self.sigma2 - 30) / 10)  # -173 dBm → 5.01e-21 W

        self.channel_status = np.log2(1 + P_E_Watts * PL_linear / sigma2_Watts)

        # Compute channel capacity using Shannon's formula
        R = self.B_E * self.channel_status  # Mbps

        # Update the data rate
        self.data_rate = R  # Mbps

    def generate_task(self):
        # generate tasks size in bits for each device
        self.si = self.random_state.randint(5 * 10**6, 8 * 10**6, self.M)

        # generate per bit tasks computation requirement in cycles for each device
        self.fi = self.random_state.randint(100, 150, self.M)

        # generate task completion deadline for each device
        self.ri = self.random_state.uniform(0.8, 1.0, self.M)

        # generate required computation cycles for each device
        self.ci = self.si * self.fi

    def generate_interruption_threshold(self):
        U = self.random_state.uniform(0, 1, self.N)
        self.interruption_threshold = np.log(U * 100 + 1) / self.lambda_I / 100

    def reset(self):
        self.init_topology()  # generate grid topology and assign devices to edge servers
        self.update_channel()  # update channel capacity
        self.generate_task()  # generate tasks for devices
        self.availability = np.ones(self.N)  # availability of edge servers
        self.recovery_time = np.zeros(self.N)  # recovery time of edge servers
        self.generate_interruption_threshold()  # generate interruption threshold
        obs, masks = self.observation()
        return obs, masks

    def observation(self):
        # Generate state observation
        observation = np.zeros((self.M, self.state_dim))
        for i in range(self.M):
            observation[i] = np.concatenate(
                [
                    [self.si[i] / 10**6],
                    [self.fi[i] / 1500],
                    [self.ri[i]],
                    self.availability,
                    np.where(
                        np.isinf(self.shortest_distances[self.device_assignments[i]]),
                        -1,
                        self.shortest_distances[self.device_assignments[i]],
                    ),
                    np.where(np.arange(self.N) == self.device_assignments[i], 1, 0),
                    [self.channel_status[i]],
                ]
            )
        # Generate action mask
        action_mask = np.zeros((self.M, self.N + 1))
        for i in range(self.M):
            for j in range(self.N):
                if self.availability[j] == 0:
                    action_mask[i, j + 1] = 1
                if (
                    get_shortest_path(
                        self.adjacency_list, self.device_assignments[i], j
                    )
                    is None
                ):
                    action_mask[i, j + 1] = 1

        return observation, action_mask

    def step(self, actions):

        device_counts = np.zeros(self.N)

        # # action masking: change to local computation if edge server is not available
        # for i in range(self.M):
        #     if actions[i] > 0:
        #         target_edge = actions[i] - 1
        #         device_counts[target_edge] += 1
        #         if self.availability[target_edge] == 0:
        #             actions[i] = 0  # computation node is not available
        #         if (
        #             get_shortest_path(
        #                 self.adjacency_list, self.device_assignments[i], target_edge
        #             )
        #             is None
        #         ):
        #             actions[i] = 0  # no path between device and edge server

        #         # add then overload bandwidth and computation capacity here

        # compute the allocated computation capacity to devices
        allocated_capacity = np.zeros(self.N)

        for i in range(self.M):
            if actions[i] > 0:
                target_edge = actions[i] - 1
                device_counts[target_edge] += 1

        for i in range(self.N):
            allocated_capacity[i] = min(5 * 10**9, self.F_E / max(1, device_counts[i]))
            # allocated_capacity[i] = self.F_E / max(1, device_counts[i])

        task_done = np.zeros(self.M, dtype=int)  # 0: task not done, 1: task done
        task_finished = np.zeros(self.M)  # task completion time
        task_arrival = np.zeros(self.M)  # task arrival time (at edge server)
        interruption_penalty = np.zeros(self.M)  # interruption penalty
        deadline_penalty = np.zeros(self.M)  # deadline penalty

        # find shortest route from and edge server to another
        list_routes = []
        for i in range(self.M):
            temp = get_shortest_path(
                self.adjacency_list,
                self.device_assignments[i],
                actions[i] - 1,
            )
            list_routes.append(temp if temp is not None else [])

        # Step 1: Calculate the expected delay for each device
        for i in range(self.M):
            target_edge = actions[i] - 1
            if actions[i] == 0:  # local computation
                task_finished[i] = self.ci[i] / self.F_D
                task_done[i] = 1
            elif (
                target_edge == self.device_assignments[i]
            ):  # offload to local edge server
                task_arrival[i] = (
                    self.si[i] / self.data_rate[i]
                )  # wireless transmission time from device to local edge server
                task_finished[i] = (
                    self.ci[i] / allocated_capacity[target_edge] + task_arrival[i]
                )
            else:  # offload to remote edge server
                hop_distance = self.shortest_distances[
                    self.device_assignments[i], target_edge
                ]
                task_arrival[i] = (
                    self.si[i]
                    / self.data_rate[
                        i
                    ]  # wireless transmission time from device to local edge server
                    + self.si[i]
                    * hop_distance
                    / self.R_E2E  # wired transmission time from local edge server to remote edge server
                )

                # print(self.si[i] * hop_distance / self.R_E2E)
                task_finished[i] = (
                    self.ci[i] / allocated_capacity[target_edge] + task_arrival[i]
                )

        while not all(task_done):
            task_execution_info = [[] for _ in range(self.N)]

            for i in range(self.M):
                if not task_done[i]:
                    target_edge = actions[i] - 1
                    task_execution_info[target_edge].append(
                        (task_arrival[i], allocated_capacity[target_edge] / self.F_E)
                    )
                    task_execution_info[target_edge].append(
                        (task_finished[i], -allocated_capacity[target_edge] / self.F_E)
                    )

            interruption_event = []
            for edge in range(self.N):  # for each edge server that can be interrupted
                if self.availability[edge] and self.interruption_threshold[edge] <= 1:
                    current_load = 0
                    last_time = 0
                    threshold = self.interruption_threshold[edge]
                    execution_info_sorted = sorted(
                        task_execution_info[edge], key=lambda x: x[0]
                    )

                    for time, load in execution_info_sorted:
                        current_load += load
                        last_time = time
                        if current_load >= threshold:
                            interruption_event.append((edge, last_time, current_load))
                            break

            if not interruption_event:
                for i in range(self.M):
                    if not task_done[i]:
                        task_done[i] = 1
                break

            sorted_interruption_event = sorted(interruption_event, key=lambda x: x[1])
            interrupted_edge = sorted_interruption_event[0][0]
            served_duration = sorted_interruption_event[0][1]

            new_adjacency_list = remove_node(self.adjacency_list, interrupted_edge)
            new_shortest_distances = shortest_hop_distance(new_adjacency_list)

            self.availability[interrupted_edge] = 0
            self.recovery_time[interrupted_edge] = self.recv

            for i in range(self.M):
                if not task_done[i]:
                    target_edge = actions[i] - 1
                    if (
                        target_edge == interrupted_edge
                    ):  # if the interrupted edge server is the target edge server
                        interruption_penalty[i] = 1
                        if (
                            task_finished[i] < served_duration
                        ):  # if the task is finished before the interruption
                            task_done[i] = 1
                        else:
                            task_finished[i] = (
                                served_duration + self.ci[i] / self.F_D
                            )  # switch to local computation
                            task_done[i] = 1
                    if target_edge != interrupted_edge:
                        old_routes = list_routes[
                            i
                        ]  # take the current wired transmission path
                        if (
                            interrupted_edge in old_routes
                        ):  # if the interrupted edge server is in the old path
                            if (
                                self.si[i]
                                * self.shortest_distances[
                                    self.device_assignments[i], interrupted_edge
                                ]
                                / self.R_E2E
                                > served_duration
                            ):  # if the data transmission is not finished
                                pre_interruption_edge = old_routes[
                                    old_routes.index(interrupted_edge) - 1
                                ]  # take the edge server before the interrupted edge server
                                if (
                                    get_shortest_path(
                                        new_adjacency_list,
                                        pre_interruption_edge,
                                        target_edge,
                                    )
                                    is None
                                ):  # if there is no path between the pre-interruption edge server and the target edge server
                                    task_finished[i] = (
                                        served_duration + self.ci[i] / self.F_D
                                    )  # switch to local computation
                                    task_done[i] = 1
                                else:  # if there is a path between the pre-interruption edge server and the target edge server
                                    task_arrival[i] = (
                                        served_duration
                                        + self.si[i]
                                        / self.data_rate[
                                            i
                                        ]  # wireless transmission time from device to local edge server
                                        + self.si[i]
                                        * new_shortest_distances[
                                            pre_interruption_edge, target_edge
                                        ]
                                        / self.R_E2E  # wired transmission time from local edge server to remote edge server
                                    )
                                    task_finished[i] = (
                                        task_arrival[i]
                                        + self.ci[i] / allocated_capacity[target_edge]
                                    )  # update the task completion time
                                    old_routes = old_routes[
                                        : old_routes.index(interrupted_edge)
                                    ] + get_shortest_path(
                                        new_adjacency_list,
                                        pre_interruption_edge,
                                        target_edge,
                                    )
                                    list_routes[i] = old_routes

            self.adjacency_list = new_adjacency_list
            self.shortest_distances = new_shortest_distances

        # step the recovery time of edge servers
        for i in range(self.N):
            if self.availability[i] == 0:
                if self.recovery_time[i] == 0:
                    self.availability[i] = 1
                else:
                    self.recovery_time[i] -= 1

        for i in range(self.M):
            if task_finished[i] > self.ri[i]:
                deadline_penalty[i] = abs(task_finished[i] - self.ri[i])

        avg_delay = np.mean(task_finished)

        rewards = -self.alpha * (task_finished) - self.beta * interruption_penalty + 70

        rewards = [rewards[i] for i in range(self.M)]

        availability_ratio = self.availability.sum() / self.N

        self.update_device_assignments()
        self.update_channel()
        self.generate_interruption_threshold()
        self.generate_task()
        observations, action_masks = self.observation()
        return (
            observations,
            action_masks,
            rewards,
            False,
            {
                "avg_delay": avg_delay,
                "availability_ratio": availability_ratio,
                "interruption_penalty": interruption_penalty / self.M,
            },
        )
