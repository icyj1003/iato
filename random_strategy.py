import numpy as np
from env.env import IIoTNetwork

T = 3000  # number of time slots
N = 5  # 10, 15, 20
M = 10  # 20, 30, 40
F_D = 2 * 10**9  # computation capacity of devices
F_E = 25 * 10**9  # computation capacity of edge servers
B_E = 2 * 10**6  # bandwidth per device: 20 MHz
P_E = 35  # transmission power: 35 dBm
coverage = 500  # edge server coverage: 500 m
sigma2 = -174  # noise power: -174 dBm/Hz
R_E2E = 10000 * 10**6  # wired connection rate: 150 Mbps
lambda_I = 1e-3  # interruption sensitivity parameter: 1e-3, 3e-3, 5e-3, 7e-3
alpha = 1
beta = 0.1

env = IIoTNetwork(
    N, M, T, F_D, F_E, B_E, P_E, coverage, sigma2, R_E2E, lambda_I, alpha, beta
)

# evaluate for 3000 time slots with on device computation only
delay = []
env.reset()
for i in range(T):
    obs, rewards, done, info = env.step(np.zeros(M, dtype=int))
    delay.append(info["avg_delay"])

print(f"Average delay: {np.mean(delay)}")

# evaluate for 3000 time slots with random strategy
delay = []
env.reset()
for i in range(T):
    obs, rewards, done, info = env.step(np.random.randint(0, N, M))
    delay.append(info["avg_delay"])

print(f"Average delay: {np.mean(delay)}")
