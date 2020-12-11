
import numpy as np

post_a = None
post_b = None
bandit = None
total_reward = 0
c = 3


def agent(observation, configuration):
    global reward_sums, total_reward, bandit, post_a, post_b, c
    
    n_bandits = configuration.banditCount

    if observation.step == 0:
        post_a = np.ones(n_bandits)
        post_b = np.ones(n_bandits)
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward

        # Update Gaussian posterior
        post_a[bandit] += r
        post_b[bandit] += (1 - r)

    samples = np.random.beta(post_a, post_b)
    bandit = int(np.argmax(samples))
    
    return bandit