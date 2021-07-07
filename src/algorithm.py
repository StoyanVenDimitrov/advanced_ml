import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import *
from dqn import DQN

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


env = gym.make('MiniGrid-Empty-5x5-v0')
env = FlatObsWrapper(env)
env.reset()
before_img = env.render('rgb_array')

# take an action and render the resulting state
action = env.actions.forward
obs, reward, done, info = env.step(action)
after_img = env.render('rgb_array')

plt.imshow(np.concatenate([before_img, after_img], 1))
plt.show()

# for i in range(100):
#     action = env.action_space
#     obs, reward, done, info = env.step(action.sample())

#     env.render()
#     if done:
#         obs = env.reset()