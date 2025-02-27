import numpy as np
import torch
from .net import Actor, Critic
from .memory import ReplayMemory
from tqdm import tqdm
import tensorboardX


class MAA2C:
    def __init__(
        self,
        env,
        num_agents,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=0.001,
        gamma=0.99,
        tau=0.01,
        device="cpu",
        batch_size=64,
        T=3000,
        memory_size=10000,
    ):
        self.num_agents = num_agents
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.batch_size = batch_size
        self.T = T
        self.writer = tensorboardX.SummaryWriter()
        self.steps = 0

        # Create a separate actor for each agent
        self.actors = [
            Actor(state_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_agents)
        ]

        # Shared critic for all agents
        self.critic = Critic(state_dim * num_agents, hidden_dim).to(device)

        # Optimizers for actors and critic
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Actors parameters are shared across all agents
        for i in range(1, self.num_agents):
            self.actors[i] = self.actors[0]
            self.actor_optimizers[i] = self.actor_optimizers[0]

        self.memory = ReplayMemory(
            10000
        )  # You can implement a shared experience buffer if needed

    def select_action(self, states):
        self.steps += 1
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = []
        log_probs = []
        for i in range(self.num_agents):
            # Each agent has its own policy
            action, log_prob = self.actors[i].act(states[i])
            actions.append(action)
            log_probs.append(log_prob)
        return actions, log_probs

    def update(self, states, actions, rewards, next_states, dones):
        # convernt to tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(
            self.device
        )
        dones = torch.tensor(np.array(dones), dtype=torch.float).to(self.device)

        # Reshape next_states to a joint representation
        joint_next_states = next_states.reshape(
            next_states.shape[0], -1
        )  # Shape: (batch_size, num_agents * state_dim)
        next_states_values = (
            self.critic(joint_next_states).squeeze(-1).detach()
        )  # Detach to prevent gradients

        # Compute the target values using Bellman equation
        target_values = (
            rewards.mean(dim=1) + (1 - dones) * self.gamma * next_states_values
        )

        # Reshape states similarly for critic input
        joint_states = states.reshape(
            states.shape[0], -1
        )  # Shape: (batch_size, num_agents * state_dim)
        values = self.critic(joint_states).squeeze(-1)  # Critic's current estimation

        # Compute advantage
        advantages = (
            target_values - values
        ).detach()  # Detach to prevent affecting critic updates

        # Update Critic (shared across all agents)
        critic_loss = torch.mean((values - target_values) ** 2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.writer.add_scalar("loss/critic", critic_loss, self.steps)

        # Update Actors
        for i in range(self.num_agents):
            agent_state = states[:, i, :]
            # Use the log probability and advantage to calculate the actor loss

            logits = self.actors[i](agent_state)

            dist = torch.distributions.Categorical(logits=logits)

            log_probs = dist.log_prob(actions[:, i])

            actor_loss = -torch.mean(log_probs * advantages.unsqueeze(-1))

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=1.0)
            self.actor_optimizers[i].step()
            self.writer.add_scalar(f"loss/actor_{i}", actor_loss, self.steps)

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            states = self.env.reset()
            done = False
            delay = []
            with tqdm(total=self.T) as pbar:
                while not done:
                    actions, log_probs = self.select_action(
                        states
                    )  # Select actions for all agents
                    next_states, rewards, dones, infos = self.env.step(
                        actions
                    )  # Take actions in environment

                    # Store experience in memory
                    self.memory.add(states, actions, rewards, next_states, dones)

                    # If enough experiences are collected, start training
                    if len(self.memory) > self.batch_size:
                        batch = self.memory.sample(self.batch_size)
                        self.update(*batch)

                    # Move to the next state
                    states = next_states
                    done = dones
                    delay.append(infos["avg_delay"])
                    pbar.update(1)

            self.writer.add_scalar("delay/episode", np.mean(delay), episode)
            print(f"Episode {episode} completed, average delay: {np.mean(delay)}")
