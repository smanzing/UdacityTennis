import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, normalize_batch_first_layers, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            normalize_batch_first_layers (bool): use batch normalization
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.normalize_batch_first_layers = normalize_batch_first_layers

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.bn = nn.BatchNorm1d(fc1_units)

        self._initialize_weights()

    def forward(self, state):
        x = F.relu(self.fc1(state))

        if self.normalize_batch_first_layers:
            x = self.bn(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

    def _initialize_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, normalize_batch_first_layers, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            normalize_batch_first_layers (bool): use batch normalization
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.normalize_batch_first_layers = normalize_batch_first_layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn = nn.BatchNorm1d(fc1_units)

        self._initialize_weights()

    def forward(self, state, action):
        x1 = F.relu(self.fc1(state))
        if self.normalize_batch_first_layers:
            x1 = self.bn(x1)
        x = torch.cat((x1, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def _initialize_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


