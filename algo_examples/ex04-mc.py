import gym
import numpy as np
import matplotlib.pyplot as plt


def mc_predictions():
    """
    First-visit MC prediction, for estimating V of the policy below
    :return: value for each state
    """
    env = gym.make('Blackjack-v0')
    gamma = 1.0
    values = np.zeros((10, 10, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable)
    times_visited = np.zeros((10, 10, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable)
    for i in range(10000):
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        episode = []
        while not done:
            # don't add episodes when hit is the only choice: at sum<12
            if obs[0] < 12:
                obs, reward, done, _ = env.step(1)
                continue
                # use the observation numbers as states
                # and let the states directly be the indices of the values
            #print("observation:", obs)
            state = (obs[0] - 12, obs[1] - 1, int(obs[2]))
            if obs[0] >= 20:
                #print("stick")
                action = 0
                obs, reward, done, _ = env.step(0)
                # find position of state value from the new state:
            else:
                #print("hit")
                action = 1
                # don't add episodes when hit is the only choice: at sum<12
                obs, reward, done, _ = env.step(1)
            #print("reward:", reward)
            #print("")
            if state is not None:
                times_visited[state[0]][state[1]][state[2]] += 1
                episode.append([state, action, reward])
        G = 0
        for step in reversed(episode):
            G = gamma*G + step[2]  # update G with the reward
            state = step[0]
            # use the state as indices
            old_value = values[state[0]][state[1]][state[2]]
            visited = times_visited[state[0]][state[1]][state[2]]
            values[state[0]][state[1]][state[2]] = old_value + (G - old_value)/visited

        i += 1
    '------ Projection ------'
    no_usable_ace_values = np.take(values, [0], axis=2)
    usable_ace_values = np.take(values, [1], axis=2)

    x, y = np.arange(1, 11), np.arange(12, 22)
    X, Y = np.meshgrid(x, y)
    Z_1 = np.squeeze(usable_ace_values, axis=2)
    Z_2 = np.squeeze(no_usable_ace_values, axis=2)

    fig = plt.figure(figsize=(120, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, Z_1)
    ax1.set_title('Usable Ace')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_wireframe(X, Y, Z_2)
    ax2.set_title('No usable Ace')
    plt.show()
    return no_usable_ace_values, usable_ace_values


def mc_exploring_starts():
    """
    :return: optimal policy
    """
    env = gym.make('Blackjack-v0')
    gamma = 1.0
    # initialize with the policy for stick if sum >20, else hit
    policy_hit = np.ones((8, 10, 2), dtype=np.int8)  # len(11<sum<20) x len(dealer_card) x (usable, not_usable)
    policy_stick = np.zeros((2, 10, 2), dtype=np.int8)  # len(sum=>20) x len(dealer_card) x (usable, not_usable)
    policy = np.append(policy_hit, policy_stick, axis=0)
    # policy = np.zeros((10, 10, 2), dtype=np.int8)  # len(sum>11) x len(dealer_card) x (usable, not_usable)
    action_values = np.zeros((10, 10, 2, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable) x [stick, hit]
    times_visited = np.zeros((10, 10, 2, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable) x [stick, hit]
    for i in range(500000):
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        episode = []
        random_start = True
        while not done:
            # for all sums < 12: hit. Cannot loose.
            if obs[0] < 12:
                obs, reward, done, _ = env.step(1)
                continue
            # print("observation:", obs)
            # use the observation numbers as states
            # and let the states directly be the indices of the values
            state = (obs[0] - 12, obs[1] - 1, int(obs[2]))
            action = policy[state[0]][state[1]][state[2]]
            if random_start:
                action = np.random.randint(2)
                random_start = False
            obs, reward, done, _ = env.step(action)
            # print("reward:", reward)
            # print("")
            times_visited[state[0]][state[1]][state[2]] += 1
            episode.append([state, action, reward])
        G = 0
        for step in reversed(episode):
            G = gamma * G + step[2]  # update G with the reward
            state = step[0]
            action = step[1]
            # use the states as indices:
            old_value = action_values[state[0]][state[1]][state[2]][action]
            visited = times_visited[state[0]][state[1]][state[2]][action]
            action_values[state[0]][state[1]][state[2]][action] = old_value + (G - old_value) / visited
            policy[state[0]][state[1]][state[2]] = np.argmax(action_values[state[0]][state[1]][state[2]])
        i += 1
    no_usable_ace_policy = np.take(policy, [0], axis=2)
    usable_ace_policy = np.take(policy, [1], axis=2)

    action_values_no_usable_ace = np.squeeze(np.take(action_values, [0], axis=2), axis=2)
    action_values_usable_ace = np.squeeze(np.take(action_values, [1], axis=2), axis=2)

    # take the action value for the chosen action for each state
    # to get the state value, because the policy is deterministic:
    values_no_usable_ace = np.take_along_axis(action_values_no_usable_ace, no_usable_ace_policy, axis=2)
    values_usable_ace = np.take_along_axis(action_values_usable_ace, usable_ace_policy, axis=2)

    print('After 10000 iterations and with gamma=1:')
    print('No usable Ace:')
    print(np.squeeze(no_usable_ace_policy))
    print('Usable ace:')
    print(np.squeeze(usable_ace_policy))


    return no_usable_ace_policy, usable_ace_policy, values_no_usable_ace, values_usable_ace


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    while not done:
        print("observation:", obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)
        print("reward:", reward)
        print("")


if __name__ == "__main__":
    # main()
    mc_exploring_starts()
