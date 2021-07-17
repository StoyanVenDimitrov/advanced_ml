import math
import random
from collections import deque, namedtuple
from itertools import count
from gym_minigrid.wrappers import *

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from dqn import DQN, FlatDQN
from transition import ReplayMemory, Transition

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

NUM_EPISODES = 300
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1
HIDDEN_STATES = 100
MEMORY_SIZE = 10000

env = gym.make('MiniGrid-Empty-5x5-v0').unwrapped

n_actions = env.action_space.n
n_actions = 3

# ! for flat
# env = FlatObsWrapper(env)
# n_states = env.observation_space.shape[0]
# policy_net = FlatDQN(n_states , HIDDEN_STATES, n_actions)
# target_net = FlatDQN(n_states , HIDDEN_STATES, n_actions)
# optimizer = optim.Adam(policy_net.parameters(), lr=0.0002)

env.reset()

memory = ReplayMemory(MEMORY_SIZE)
# resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
# def get_screen(obs):
#     screen = env.render(mode='rgb_array').transpose((2, 0, 1))
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#     screen = torch.from_numpy(screen)
#     screen = resize(screen).unsqueeze(0) # make 40x40 pictures from 160x160
#     return screen
# obs = env.reset()
# init_screen = get_screen(obs)
# _,_,screen_height, screen_width= init_screen.shape
policy_net = DQN(n_actions)
target_net = DQN(n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
criterion = nn.SmoothL1Loss()
#! optimizer = optim.SGD(policy_net.parameters(), lr=0.001, momentum=0.9)
#! criterion = nn.CrossEntropyLoss()

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # eps_threshold = 0.1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(policy_net(state)).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_output = policy_net(state_batch)
    state_action_values = policy_output.gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

for i_episode in range(NUM_EPISODES):
    state = env.reset()['image']
    for t in count():
        # Select and perform an action
        action = select_action(state)
        #reward = torch.tensor([reward], device=device)
        next_state, reward, done, _ = env.step(action.item())
        torch.as_tensor(next_state['image'], dtype=torch.float32)
        memory.push(
            torch.as_tensor(state, dtype=torch.float32),
            action, 
            torch.as_tensor(next_state['image'], dtype=torch.float32), 
            torch.tensor([reward])
        ) 
        state = next_state['image']      
        optimize_model() #! optimization for BATCH after a single observation? 
        env.render()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
# Update the target network, copying all weights and biases in DQN
# if i_episode % TARGET_UPDATE == 0:
#     target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

# done:<Actions.done: 6>
# drop:<Actions.drop: 4>
# forward:<Actions.forward: 2>
# left:<Actions.left: 0>
# pickup:<Actions.pickup: 3>
# right:<Actions.right: 1>
# toggle:<Actions.toggle: 5>
