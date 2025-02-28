import numpy as np
import tensorboardX
import torch
from tqdm import tqdm

from .memory import ReplayMemory
from .net import Actor, Critic


class MAA2C:
    def __init__(
        self,
        env,
        num_agents,
        state_dim,
        action_dim,
        hidden_dim=64,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        device="cpu",
        batch_size=64,
        T=3000,
        memory_size=10000,
        grad_clip=5.0,
        learn_every=10,
    ):
        self.num_agents = num_agents
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.T = T
        self.writer = tensorboardX.SummaryWriter()
        self.steps = 0
        self.memory_size = memory_size
        self.grad_clip = grad_clip
        self.learn_every = learn_every

        # Create a separate actor for each agent
        self.actors = [
            Actor(state_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_agents)
        ]

        # Shared critic for all agents
        self.critics = [
            Critic(num_agents * state_dim, hidden_dim).to(device)
            for _ in range(num_agents)
        ]

        # Optimizers for actors and critic
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=critic_lr)
            for critic in self.critics
        ]

        # Actors parameters are shared across all agents
        for i in range(1, self.num_agents):
            self.actors[i] = self.actors[0]
            self.actor_optimizers[i] = self.actor_optimizers[0]

        # Critic parameters are shared across all agents
        for i in range(1, self.num_agents):
            self.critics[i] = self.critics[0]
            self.critic_optimizers[i] = self.critic_optimizers[0]

        # Memory for experience replay
        self.memory = ReplayMemory(self.memory_size)

    def select_action(self, states):
        # Select actions for all agents
        self.steps += 1

        # Convert states to tensors
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

        # Update the critic and actor networks for each agent
        for i in range(self.num_agents):

            ## Update Critic ##

            # Use the joint states to get the value of the current state
            joint_next_states = next_states.reshape(next_states.shape[0], -1)

            # Get the value of the next state
            next_states_values = self.critics[i](joint_next_states).squeeze(-1).detach()

            # Calculate the target value using the Bellman equation
            target_values = (
                rewards[:, i] + (1 - dones) * self.gamma * next_states_values
            )

            # Get the value of the current state
            joint_states = states.reshape(states.shape[0], -1)
            values = self.critics[i](joint_states).squeeze(-1)

            # Calculate the advantage
            advantages = (target_values - values).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Calculate the critic loss using Huber loss
            critic_loss = torch.nn.functional.smooth_l1_loss(values, target_values)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.critics[i].parameters(), max_norm=self.grad_clip
            )

            self.critic_optimizers[i].step()

            # Log the critic loss
            self.writer.add_scalar(f"loss/critic_{i}", critic_loss, self.steps)

            ## Update Actor ##

            # Get the current state of the agent
            agent_state = states[:, i, :]

            # Get the logits from the actor network
            logits = self.actors[i](agent_state)

            # Create a distribution from the logits
            dist = torch.distributions.Categorical(logits=logits)

            # Get the log probabilities of the actions
            log_probs = dist.log_prob(actions[:, i])

            # Calculate the actor loss
            entropy_loss = dist.entropy().mean()  # Entropy regularization
            actor_loss = (
                -torch.mean(log_probs * advantages) - 0.001 * entropy_loss
            )  # Add entropy term

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actors[i].parameters(), max_norm=self.grad_clip
            )

            self.actor_optimizers[i].step()

            # Log the actor loss
            self.writer.add_scalar(f"loss/actor_{i}", actor_loss, self.steps)

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            states = self.env.reset()
            done = False
            delay = []
            with tqdm(total=self.T) as pbar:  # trainng for T time slots in each episode
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

                        if self.steps % self.learn_every == 0:
                            self.update(*batch)

                    # Move to the next state
                    states = next_states
                    done = dones
                    delay.append(infos["avg_delay"])
                    pbar.update(1)

            self.writer.add_scalar("delay/episode", np.mean(delay), episode)
            print(f"Episode {episode} completed, average delay: {np.mean(delay)}")

    def save(self, path):
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), f"{path}/actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), f"{path}/critic_{i}.pth")

    def eval(self):
        states = self.env.reset()
        done = False
        delay = []
        with tqdm(total=self.T) as pbar:
            while not done:
                actions, _ = self.select_action(states)
                next_states, rewards, dones, infos = self.env.step(actions)
                states = next_states
                done = dones
                delay.append(infos["avg_delay"])
                pbar.update(1)

        print(f"Average delay: {np.mean(delay)}")
        return np.mean(delay)
