import random
from collections import namedtuple


Experience = namedtuple(
    "Experience", ("states", "masks", "actions", "rewards", "next_states", "dones")
)


class ReplayMemory(object):
    """
    Replay memory buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, mask, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(
            state, mask, action, reward, next_state, done
        )
        self.position = (self.position + 1) % self.capacity

    def add(self, states, masks, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(
                    states, masks, actions, rewards, next_states, dones
                ):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, masks, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, masks, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
