import gym
import numpy as np

# Init environment

env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

# (0-left 1-down 2-right 3-up)

def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # one iteration:
    stopping_reached = False
    iteration = 0
    while not stopping_reached:
        delta = 0.0
        for state, v_state in enumerate(V_states):
            best_value = 0.0
            for action in range(n_actions):
                value = 0.0
                for dynamics in env.P[state][action]:
                    value += dynamics[0]*(dynamics[2] + gamma*V_states[dynamics[1]])
                best_value = max(best_value, value)
            delta = max(delta, abs(v_state - best_value))
            V_states[state] = best_value
        iteration += 1
        if delta < theta:
            stopping_reached = True

    action_values = np.zeros((n_states, n_actions))
    for state, v_state in enumerate(V_states):
        for action in range(n_actions):
            value = 0.0
            for dynamics in env.P[state][action]:
                value += dynamics[0] * (dynamics[2] + gamma * V_states[dynamics[1]])
            action_values[state][action] = value
    print('Iterations: ', iteration)
    print('Optimal value function: ', V_states)
    optimal_policy = np.argmax(action_values, axis=1)
    # best_policies = []
    # for row in action_values:
    #     winners = np.argwhere(row == np.amax(row))
    #     # skip adding policies where every values are all 0.0, like terminal states
    #     if np.amax(row) != 0.0:
    #         best_policies.append(winners.flatten().tolist())
    #     else:
    #         best_policies.append([0])

    return optimal_policy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    # print ("rollout policy:")
    # maxiter = 100
    # state = env.reset()
    # for i in range(maxiter):
    #     new_state, reward, done, info = env.step(policy[state])
    #     env.render()
    #     state=new_state
    #     if done:
    #         print ("Finished episode")
    #         break


if __name__ == "__main__":
    main()
