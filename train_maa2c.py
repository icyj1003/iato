import argparse
import datetime
import json
import os

from env.env import IIoTNetwork
from maa2c.agent import MAA2C

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
parser.add_argument("--recv", type=int, default=0, help="Recovery time")


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
    recv = args.recv

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
        recv,
        seed,
    )

    agent = MAA2C(
        env,
        M,
        env.state_dim,
        env.action_dim,
        hidden_dim=hidden_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        device=device,
        batch_size=batch_size,
        T=T,
        memory_size=memory_size,
        grad_clip=0.5,
    )

    agent.train(num_episodes=episodes)

    now = datetime.datetime.now()

    path = os.path.join(
        "models", f"N={N}_M={M}_lambdaI={lambda_I}_{now.strftime('%b%d_%H-%M-%S')}"
    )

    os.makedirs(path, exist_ok=True)

    # save args
    with open(os.path.join(path, "args.txt"), "w") as f:
        f.write(str(args))

    # save agents
    agent.save(path)

    # eval agens
    results = agent.eval()

    # save results as json
    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump(results, f)
