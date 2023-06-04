from rl_lib.agent import DDPGAgent
from rl_lib.replay import ReplayBuffer
from typing import List
import numpy as np


class MultiAgent:
    def __init__(self, num_agents: int, agents: List[DDPGAgent], replay_buffers: List[ReplayBuffer]):
        """
        @param num_agents number of agents in the environment
        @param agents list of DDPGAgents
        @param replay_buffers list of ReplayBuffer for each DDPGAgent
        """
        self.num_agents = num_agents
        self.agents = agents
        self.replay_buffers = replay_buffers

    def reset(self):
        for a in self.agents:
            a.reset()

    def act(self, states, add_noise=True, clip=True):
        all_actions = list()
        for i in range(self.num_agents):
            action = self.agents[i].act(states[(i,), :], add_noise, clip)
            all_actions.append(action)
        all_actions = np.vstack(all_actions)
        return all_actions

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for i in range(self.num_agents):
            self.replay_buffers[i].add(states[(i,), :],
                                       actions[(i,), :],
                                       rewards[i],
                                       next_states[(i,), :],
                                       dones[i])

        for i in range(self.num_agents):
            self.agents[i].step(self.replay_buffers)

    def save_networks(self, filename):
        for i in range(self.num_agents):
            self.agents[i].save_networks(filename)
