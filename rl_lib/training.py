import torch
import numpy as np
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_ddpg(env, brain_name, agent, n_episodes=2000, max_t=1000, min_avg_score=13.0, continue_learning=False,
               filename='checkpoint'):
    """
    @param env: unity environment
    @param brain_name: name of the brain that we control from Python
    @param agent: the RL agent that we train
    @param n_episodes: maximum number of training episodes
    @param max_t: maximum number of timesteps per episode
    @param min_avg_score: minimum average score over 100 episodes that the agent must achieve to consider the task fulfilled
    @param continue_learning: if true, the agent continues to learn after reaching min_avg_score until reaching n_episodes
    @param filename: name for the file that contains the trained network parameters
    @return:
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    min_score_achieved = False

    for i_episode in range(1, n_episodes + 1):
        # reset the environment and agent
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        # get current states
        states = env_info.vector_observations
        # initialize scores
        score = np.zeros(len(env_info.agents))
        for t in range(max_t):
            # select action
            actions = agent.act(states)
            # simulate
            env_info = env.step(actions)[brain_name]
            # get next state, reward, and check if episode has finished
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_state, dones)

            # update state and reward
            states = next_state
            score += rewards

            if np.any(dones):
                break

        scores_window.append(np.max(score))  # save most recent score
        scores.append(np.max(score))  # save most recent score

        print('\rEpisode {}\tAverage Max Score: {:.6f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Max Score: {:.6f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= min_avg_score and not min_score_achieved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Max Score: {:.6f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            agent.save_networks(filename + 'min_score_achieved_')
            min_score_achieved = True
            if not continue_learning:
                break
    agent.save_networks(filename + 'final_')
    return scores
