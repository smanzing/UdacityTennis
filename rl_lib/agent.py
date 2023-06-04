import numpy as np
import random

from rl_lib.model import Actor, Critic
from rl_lib.replay import ReplayBuffer
from rl_lib.noise import OUNoise

import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Batch:
    def __init__(self):
        states = None
        actions = None
        rewards = None
        next_states = None
        dones = None
        actions_local = None
        actions_next_states = None


class DDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, agent_id: int, seed: int, nr_agents: int,
                 gamma: float = 0.99, tau: float = 1e-3, learning_rate_actor: float = 1e-4, learning_rate_critic=1e-3,
                 weight_decay_critic=10e-2, update_n_times: int = 1, normalize_batch_first_layers: bool = False):
        """
        @param state_size: dimension of each state
        @param action_size: dimension of each action
        @param agent_id: ID of agent in environment
        @param seed:  random seed
        @param seed:  number of agents
        @param gamma: discount factor
        @param tau: for soft update of target parameters
        @param learning_rate_actor: learning rate
        @param learning_rate_critic: learning rate
        @param weight_decay_critic: weight decay
        @param update_n_times: how often to run the learning step
        @param normalize_batch_first_layers: use batch normalization for networks after first layer
        """

        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.nr_agents = nr_agents
        self.seed = random.seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.update_n_times = update_n_times

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.weight_decay_critic = weight_decay_critic

        # Networks
        self.actor_network_local = Actor(state_size, action_size, seed, normalize_batch_first_layers).to(device)
        self.actor_network_target = Actor(state_size, action_size, seed, normalize_batch_first_layers).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network_local.parameters(), lr=self.learning_rate_actor)

        self.critic_network_local = Critic(state_size * self.nr_agents, action_size * self.nr_agents, seed,
                                           normalize_batch_first_layers).to(device)
        self.critic_network_target = Critic(state_size * self.nr_agents, action_size * self.nr_agents, seed,
                                            normalize_batch_first_layers).to(device)
        self.critic_optimizer = optim.Adam(self.critic_network_local.parameters(), lr=self.learning_rate_critic,
                                           weight_decay=self.weight_decay_critic)

        self._copy_weights(self.actor_network_local, self.actor_network_target)
        self._copy_weights(self.critic_network_local, self.critic_network_target)

        # Process noise
        self.noise = OUNoise(self.action_size, self.seed)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True, clip=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_network_local.eval()
        with torch.no_grad():
            action = self.actor_network_local(state).cpu().data.numpy()
        self.actor_network_local.train()
        if add_noise:
            action += self.noise.sample()
        if clip:
            action = np.clip(action, -1, 1)
        return action

    def save_networks(self, filename):
        torch.save(self.critic_network_local.state_dict(), filename + str(self.agent_id) + '_critic.pth')
        torch.save(self.actor_network_local.state_dict(), filename + str(self.agent_id) + '_actor.pth')

    def step(self, replay_buffers: List[ReplayBuffer]):
        """Update value parameters using a sampled batch of experience tuples."""

        # all replay buffers have same size, so we just take the first one
        if not replay_buffers[self.agent_id].is_size_of_memory_sufficient_to_draw_batch():
            return

        for i in range(self.update_n_times):
            batch_indices = np.random.choice(len(replay_buffers[self.agent_id]),
                                             size=replay_buffers[self.agent_id].batch_size)
            own_batch = self._prepare_own_batch(replay_buffers, batch_indices)
            combined_batch = self._prepare_combined_batch(replay_buffers, batch_indices)
            self._update_critic(own_batch, combined_batch)
            self._update_actor(combined_batch)

            self._soft_update(self.actor_network_local, self.actor_network_target)
            self._soft_update(self.critic_network_local, self.critic_network_target)

    def _prepare_combined_batch(self, replay_buffers: List[ReplayBuffer], batch_indices):
        states, actions, rewards, next_states, dones = replay_buffers[0].sample(batch_indices)
        actions_local = self.actor_network_local(states)
        actions_next_states = self.actor_network_target(next_states)

        for i in range(1, len(replay_buffers)):
            state, action, reward, next_state, done = replay_buffers[i].sample(batch_indices)
            action_local = self.actor_network_local(state)
            action_next_state = self.actor_network_target(next_state)

            states = torch.hstack((states, state))
            actions = torch.hstack((actions, action))
            rewards = torch.hstack((rewards, reward))
            next_states = torch.hstack((next_states, next_state))
            dones = torch.hstack((dones, done))
            actions_local = torch.hstack((actions_local, action_local))
            actions_next_states = torch.hstack((actions_next_states, action_next_state))

        combined_batch = Batch()
        combined_batch.states = states
        combined_batch.actions = actions
        combined_batch.rewards = torch.sum(rewards, dim=1)
        combined_batch.next_states = next_states
        combined_batch.dones = dones
        combined_batch.actions_local = actions_local
        combined_batch.actions_next_states = actions_next_states
        return combined_batch

    def _prepare_own_batch(self, replay_buffers: List[ReplayBuffer], batch_indices):
        states, actions, rewards, next_states, dones = replay_buffers[self.agent_id].sample(batch_indices)
        actions_local = self.actor_network_local(states)
        actions_next_states = self.actor_network_target(next_states)

        own_batch = Batch()
        own_batch.states = states
        own_batch.actions = actions
        own_batch.rewards = rewards
        own_batch.next_states = next_states
        own_batch.dones = dones
        own_batch.actions_local = actions_local
        own_batch.actions_next_states = actions_next_states
        return own_batch

    def _update_actor(self, combined_batch: Batch):
        actor_loss = -self.critic_network_local(combined_batch.states, combined_batch.actions_local).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def _update_critic(self, own_batch: Batch, combined_batch: Batch):
        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.critic_network_target(combined_batch.next_states, combined_batch.actions_next_states)
        # Compute Q targets for current states
        q_targets = own_batch.rewards + (self.gamma * q_targets_next * (1 - own_batch.dones))
        # Get expected Q values from local model
        q_expected = self.critic_network_local(combined_batch.states, combined_batch.actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_network_local.parameters(), 1)
        self.critic_optimizer.step()

    def _soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _copy_weights(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
