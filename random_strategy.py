import argparse

import numpy as np

from env.env import IIoTNetwork

parser = argparse.ArgumentParser(
    description="Train MAA2C agent in IIoTNetwork environment"
)
parser.add_argument("--N", type=int, default=5, help="number of devices")
parser.add_argument("--M", type=int, default=20, help="number of edge servers")
parser.add_argument(
    "--F_D", type=float, default=1e9, help="computation capacity of devices"
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
    "--lambda_I", type=float, default=1e-2, help="interruption sensitivity parameter"
)
parser.add_argument("--alpha", type=float, default=10, help="alpha parameter")
parser.add_argument("--beta", type=float, default=1, help="beta parameter")
parser.add_argument("--episodes", type=int, default=200, help="number of episodes")
parser.add_argument("--device", type=str, default="cpu", help="device to train on")
parser.add_argument("--actor_lr", type=float, default=1e-4, help="actor learning rate")
parser.add_argument(
    "--critic_lr", type=float, default=1e-2, help="critic learning rate"
)
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--memory_size", type=int, default=10000, help="memory size")
parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--recv", type=int, default=1, help="Recovery time")

if __name__ == "__main__":

    args = parser.parse_args()

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
    recv = args.recv

    env = IIoTNetwork(
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
        recv,
        seed,
    )

    def eval(self, strategy="random"):
        print(f"Evaluating with {strategy} strategy")
        delay = []
        availabilities = []
        obs, masks = self.reset()
        for i in range(3000):
            if strategy == "random":
                logits = np.random.rand(M, N + 1)
                logits = logits + masks * -1e9
                actions = np.argmax(logits, axis=1)
            elif strategy == "on_device":
                actions = np.zeros(M)
            elif strategy == "on_local":
                actions = env.device_assignments + 1
                for i in range(M):
                    if masks[i, actions[i] - 1] == 1:
                        actions[i] = 0

            obs, masks, rewards, done, info = self.step(actions)
            delay.append(info["avg_delay"])
            availabilities.append(info["availability_ratio"])

        print(f"Average delay: {np.mean(delay)}")
        print(f"Average availability: {np.mean(availabilities)}")

    eval(env, "random")
    eval(env, "on_device")
    eval(env, "on_local")
