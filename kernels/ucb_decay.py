import numpy as np

decay = 0.97
total_reward = 0
bandit = None

def agent(observation, configuration):
    global reward_sums, n_selections, total_reward, bandit
    
    n_bandits = configuration.banditCount

    if observation.step == 0:
        n_selections = np.full(n_bandits, 1e-32)
        reward_sums = np.full(n_bandits, 1e-32)
    else:
        reward_sums[bandit] += decay * (observation.reward - total_reward)
        total_reward = observation.reward

    avg_reward = reward_sums / n_selections    
    delta_i = np.sqrt(2 * np.log(observation.step + 1) / n_selections)
    bandit = int(np.argmax(avg_reward + delta_i))

    n_selections[bandit] += 1

    return bandit