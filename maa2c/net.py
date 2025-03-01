import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(
            hidden_dim, action_dim
        )  # Output is the logits for a categorical distribution
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, state, mask):

        x = self.fc1(state)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.fc3(x)

        x = x + mask * -1e9  # Mask out invalid actions
        logits = F.softmax(x, dim=-1)  # Mask out invalid actions
        return logits  # Return logits for Categorical distribution

    def act(self, state, mask):
        """
        Sample an action from the policy distribution (Categorical).
        """
        logits = self.forward(state, mask)
        dist = torch.distributions.Categorical(probs=logits)
        action = dist.sample()  # Sample an action
        return action, dist.log_prob(action)  # Return the action and log-probability

    def get_action_prob(self, state, mask):
        """
        Get the probability of each action (for logging or debugging purposes).
        """
        logits = self.forward(state, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.probs


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(
            hidden_dim, 1
        )  # Output is a single scalar value (state-action value)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action

        x = self.fc1(x)
        # x = self.norm1(x)
        x = F.relu(x)

        x = self.fc2(x)
        # x = self.norm2(x)
        x = F.relu(x)

        value = self.fc3(x)

        return value  # State-action value estimation
