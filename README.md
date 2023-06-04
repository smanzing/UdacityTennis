# Collaboration and Competition

## Project Details

In this project, we train two agents to play tennis. 

### Environment

**Number of Agents**: There are two agents in the environment.

**Reward**: Each agent receives a reward of 
 - +0.1 for hitting the ball over the net. 
 - -0.01 for letting the ball hit the ground or letting the ball out of bounds.

**Observation Space**: For each agent, we observe 8 variables that are the position and velocity of the ball and racket.

**Action Space**: For each agent, we have two actions corresponding to the horizontal movement towards and away
from the net, and the vertical movement of jumping.

**Duration**: The task is episodic.

**Success Criteria**: The environment is solved, if the average maximum score of the agents is at least +0.5 over 100 
consecutive episodes. This means that we sum the reward of each agent after each episode. We then pick the maximum score, 
i.e., the accumulated reward of one agent. This procedure is repeated over 100 consecutive episodes and the resulting scores
are then averaged over the 100 episodes.


## Getting Started

Follow the instructions below to set up your python environment to run the code in this repository.
Note that the installation was only tested for __Mac__.

1. Download the repository and change to the root directory of the repository.

2. Initialize the git submodules in the root folder of this repository. 

    ```bash
    git submodule init
    git submodule update
    ```
 
3. Create (and activate) a new conda environment with Python 3.6.

    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
	
4. Install the base Gym library and the **box2d** environment group:

    ```bash
    pip install gym
    pip install Box2D gym
    ```

5. Navigate to the `external/deep-reinforcement-learning/python/` folder.  Then, install several dependencies.

    ```bash
    cd external/deep-reinforcement-learning/python
    pip install .
    ```
    **Note**: If an error appears during installation regarding the torch version, please remove the torch version 0.4.0 in
    external/deep-reinforcement-learning/python/requirements.txt and try again.

6. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
    **Note**: Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

7. Download the [Unity environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip) in the **root folder** of this repository.

    
## Instructions

After you have successfully installed the dependencies, open the jupyter notebook [Tennis.ipynb](Tennis.ipynb) 
and follow the instructions to learn how to train the agents.

Below, you can find an overview of the files in the repository.

### Repository Overview

The folder **external** contains the repository [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
as submodule. The repository is only used for installation purposes.

The folder **rl_lib** contains the library that implements the agents and the algorithms for training the agents: 

- [training.py](rl_lib/training.py): The file contains the main function **train_ddpg** for training the agents. 
- [multi_agent.py](rl_lib/multi_agent.py): Orchestration of learning process for multiple agents in the environment.
- [agent.py](rl_lib/agent.py): Implementation of a single agent that interacts and learns from the environment. 
                               The member function **step** implements the DDPG algorithm.
- [replay.py](rl_lib/replay.py): Implementation of a replay buffer to store and extract experiences for learning.
- [noise.py](rl_lib/noise.py): Implementation of Ornstein-Uhlenbeck process noise.
- [model.py](rl_lib/model.py): Neural network models for the actor and critic.

The jupyter notebook [Tennis.ipynb](Tennis.ipynb) shows how to train the agents using the MADDPG algorithm.

## References 

The implementation is based on the research paper [[1]](#1) and the code provided by Udacity, see 
[deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning#dependencies).

Following repository served as inspiration for the implementation:
- https://github.com/vivekthota16/Project-Collaboration-and-Competition-Udacity-Deep-Reinforcement-Learning/tree/master

<a id="1">[1]</a> 
Lowe, R. *et al.* 
**Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.**
*arXiv* **1706.02275**, (2020).
