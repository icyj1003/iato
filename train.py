from env.env import IIoTNetwork
from maa2c.agent import MAA2C

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

agent = MAA2C(
    env,
    M,
    env.state_dim,
    env.action_dim,
    hidden_dim=32,
    lr=0.001,
    gamma=0.99,
    tau=0.01,
    # device="cuda:0",
    device="cpu",
    batch_size=64,
    T=T,
    memory_size=10000,
)

agent.train(200)
