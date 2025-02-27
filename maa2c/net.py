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

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits  # Return logits for Categorical distribution

    def act(self, state):
        """
        Sample an action from the policy distribution (Categorical).
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()  # Sample an action
        return action, dist.log_prob(action)  # Return the action and log-probability

    def get_action_prob(self, state):
        """
        Get the probability of each action (for logging or debugging purposes).
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.probs


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(
            hidden_dim, 1
        )  # Output is a single scalar value (state-value)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value  # State-value estimation
