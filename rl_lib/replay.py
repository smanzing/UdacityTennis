import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, random_seed):
        """
        @param action_size: dimension of each action
        @param buffer_size: maximum size of buffer
        @param batch_size: size of each training batch
        @param random_seed: random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(random_seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_indices=None):
        """Randomly sample a batch of experiences from memory."""
        # experiences = random.choices(self.memory, k=self.batch_size)
        if batch_indices is None:
            batch_indices = np.random.choice(len(self.memory), size=self.batch_size)

        states = torch.from_numpy(
            np.vstack([self.memory[i].state for i in batch_indices if self.memory[i] is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([self.memory[i].action for i in batch_indices if self.memory[i] is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([self.memory[i].reward for i in batch_indices if self.memory[i] is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([self.memory[i].next_state for i in batch_indices if self.memory[i] is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([self.memory[i].done for i in batch_indices if self.memory[i] is not None]).astype(
                np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def is_size_of_memory_sufficient_to_draw_batch(self) -> bool:
        return len(self.memory) > self.batch_size
