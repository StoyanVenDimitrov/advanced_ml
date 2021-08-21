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
from torch._C import AggregationType
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

ENV = "MiniGrid-FourRooms-v0" # "MiniGrid-MultiRoom-N2-S4-v0" # 'MiniGrid-Empty-5x5-v0'
NUM_EPISODES = 100
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.1
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1
HIDDEN_STATES = 100
MEMORY_SIZE = 10000

env = gym.make(ENV).unwrapped

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

policy_net = DQN(n_actions)
target_net = DQN(n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
criterion = nn.SmoothL1Loss()
# optimizer = optim.RMSprop(policy_net.parameters())
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
    plt.savefig('_env_reward_eps_01.png')
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

# def get_potential():
#     """get the potential Phi in the current env state
#         for 8x8 empty grid world
#     """
#     agent_pos = env.agent_pos
#     goal_pos = (8,8)
#     mannh_dist = goal_pos[0] - agent_pos[0] + goal_pos[1] - agent_pos[1]
#     return (-1)*mannh_dist

def _get_dir_to_goal(agent, goal):
    """get the direction to the goal from the current agent position
    """
    directions = []
    if agent[0]-goal[0]<0:
        directions.append(0)
    if agent[0]-goal[0]>0:
        directions.append(2)
    
    if agent[1]-goal[1]<0:
        directions.append(1)
    if agent[1]-goal[1]>0:
        directions.append(3)
    return directions

def get_potential():
    """get the potential Phi in the current env state
        for FourRooms environment
    """
    walls = [[i,9] for i in range(1,18)] + [[9,i] for i in range(1,18)]
    rooms = {
        1:([0,9],[0,9]), 
        2:([0,9], [10,18]),
        3:([10,18], [10,18]),
        4:([10,18], [0,9])
    }
    walls.remove([9,9])
    agent_pos = env.agent_pos
    for i in range(1,env.grid.width-1):
        for j in range(1,env.grid.height-1):
            cell = env.grid.get(i, j)
            if cell:
                if cell.type!='goal':
                    walls.remove([i,j])
                if cell.type=='goal':
                    goal = cell.cur_pos
    doors = {1:walls[0], 2:walls[3], 3:walls[1], 4:walls[2]}
    directions = _get_dir_to_goal(agent_pos, goal)
    # consider changes in agent orientation to the value
    phi = 1 if env.agent_dir in directions else -1
    # check which doors the agent has to pass 
    for room in rooms.items(): 
        # determine agents current room:
        if room[1][0][0] <= agent_pos[0] <= room[1][0][1] and room[1][1][0] <= agent_pos[1] <= room[1][1][1]:
            agent_room = room[0]
        if room[1][0][0] <= goal[0] <= room[1][0][1] and room[1][1][0] <= goal[1] <= room[1][1][1]:
            goal_room = room[0]
    if agent_room == goal_room:
        mannh_dist = abs(goal[0] - agent_pos[0]) + abs(goal[1] - agent_pos[1])
        return phi + (-1)*mannh_dist

    points = [] 
    # order doors, goal and agent in a circle:
    for door in doors.items():
        if agent_room == door[0]:
            points.append(agent_pos)
        if goal_room == door[0]:
            points.append(goal)
        points.append(door[1])
    goal_and_agent = [i for i,el in enumerate(points) if isinstance(el, (np.ndarray) )]
    traj_1 = points[goal_and_agent[0]: goal_and_agent[1]+1]
    traj_2 = points[goal_and_agent[1]:] + points[0: goal_and_agent[0]+1]
    mannh_dist1 = sum([abs(traj_1[i][0]-traj_1[i+1][0])+abs(traj_1[i][1]-traj_1[i+1][1]) for i in range(len(traj_1)-1)])
    mannh_dist2 = sum([abs(traj_2[i][0]-traj_2[i+1][0])+abs(traj_2[i][1]-traj_2[i+1][1]) for i in range(len(traj_2)-1)])
    mannh_dist = min(mannh_dist1, mannh_dist2)
    if mannh_dist==mannh_dist1:
        next_door = traj_1[1] if (traj_1[0]==agent_pos).all() else traj_1[-2]
    if mannh_dist==mannh_dist2:
        next_door = traj_2[1] if (traj_2[0]==agent_pos).all() else traj_2[-2]
    directions = _get_dir_to_goal(agent_pos, next_door)
    phi = 1 if env.agent_dir in directions else -1
    return phi + (-1)*mannh_dist


for i_episode in range(NUM_EPISODES):
    state = env.reset()['image']
    for t in count():
        # Select and perform an action
        f_s = get_potential()
        action = select_action(state)
        next_state, env_reward, done, _ = env.step(action.item())
        # update the reward:
        f_s_prime = get_potential()
        reward = env_reward + (f_s_prime - f_s )
        torch.as_tensor(next_state['image'], dtype=torch.float32)
        memory.push(
            torch.as_tensor(state, dtype=torch.float32),
            action, 
            torch.as_tensor(next_state['image'], dtype=torch.float32), 
            torch.tensor([env_reward],dtype=torch.float32)
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
