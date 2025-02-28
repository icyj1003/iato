import argparse

import numpy as np

from env.env import IIoTNetwork

parser = argparse.ArgumentParser(
    description="Train MAA2C agent in IIoTNetwork environment"
)
parser.add_argument("--T", type=int, default=3000, help="number of time slots")
parser.add_argument("--N", type=int, default=5, help="number of devices")
parser.add_argument("--M", type=int, default=10, help="number of edge servers")
parser.add_argument(
    "--F_D", type=float, default=2e9, help="computation capacity of devices"
)
parser.add_argument(
    "--F_E", type=float, default=25e9, help="computation capacity of edge servers"
)
parser.add_argument("--B_E", type=float, default=2e6, help="bandwidth per device")
parser.add_argument("--P_E", type=float, default=35, help="transmission power")
parser.add_argument("--coverage", type=float, default=500, help="edge server coverage")
parser.add_argument("--sigma2", type=float, default=-174, help="noise power")
parser.add_argument("--R_E2E", type=float, default=150e6, help="wired connection rate")
parser.add_argument(
    "--lambda_I", type=float, default=1e-3, help="interruption sensitivity parameter"
)
parser.add_argument("--alpha", type=float, default=100, help="alpha parameter")
parser.add_argument("--beta", type=float, default=5, help="beta parameter")
parser.add_argument("--episodes", type=int, default=100, help="number of episodes")
parser.add_argument("--device", type=str, default="cpu", help="device to train on")
parser.add_argument("--actor_lr", type=float, default=1e-5, help="actor learning rate")
parser.add_argument(
    "--critic_lr", type=float, default=1e-2, help="critic learning rate"
)
parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--memory_size", type=int, default=100000, help="memory size")
parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
parser.add_argument("--seed", type=int, default=0, help="random seed")


if __name__ == "__main__":

    args = parser.parse_args()

    T = args.T
    N = args.N
    M = args.M
    F_D = args.F_D
    F_E = args.F_E
    B_E = args.B_E
    P_E = args.P_E
    coverage = args.coverage
    sigma2 = args.sigma2
    R_E2E = args.R_E2E
    lambda_I = args.lambda_I
    alpha = args.alpha
    beta = args.beta
    episodes = args.episodes
    device = args.device
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    gamma = args.gamma
    batch_size = args.batch_size
    memory_size = args.memory_size
    hidden_dim = args.hidden_dim
    seed = args.seed

    env = IIoTNetwork(
        N,
        M,
        T,
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
        seed,
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
