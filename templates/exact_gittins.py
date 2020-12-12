from abc import ABCMeta, abstractmethod
import numpy as np

class AbstractAgent(metaclass=ABCMeta):   
    def init_actions(self, n_actions):
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._rival_moves = np.zeros(n_actions)
        self._total_pulls = 0
    
    @abstractmethod
    def get_action(self):
        '''
        Get current best action
        :rtype: int
        '''
        pass
    
    def update(self, action, reward, rival_move=None):
        '''
        Observe reward from action and update agent's internal parameters
        :type action: int
        :type reward: int
        '''
        self._total_pulls += 1
        if reward > 0:
            self._successes[action] += 1
        else:
            self._failures[action] += 1
            
        if rival_move is not None:
            self._rival_moves[rival_move] += 1
    
    @property
    def name(self):
        return self.__class__.__name__
    
from collections import defaultdict

class ExactGittinsIndexAgent(AbstractAgent):
    def __init__(self):
        alpha = 0.125
        distortion_horizon = 1.01
        eps = 0
        p = 1
        q = 1
        decay = 0
        rival_drift = 0
        {}
        self.horizon = 2000 # fixed by game
        self.distortion_horizon = distortion_horizon
        self.alpha = alpha
        self.eps = eps
        self.p = p
        self.q = q
        self._decay = decay
        self._rival_drift = rival_drift
        
    def get_gittins(self):
        '''
        Exact algorithm
        '''
        p = self._successes + self.p
        q = self._failures + self.q
        n = p + q
        
        m = max((self.distortion_horizon * self.horizon) - self._total_pulls + 1, 1)
        mu = float(m) / n

        gittins = p / n  + np.sqrt((2. * self.alpha) / n * np.log(mu / np.sqrt(np.maximum(1e-9, np.log(mu)))))
        
        gittins[n < 1] = float('+inf')
        
        {}
      
        return gittins
    
    def update(self, action, reward, rival_move=None):
        super().update(action, reward, rival_move)
        
    def get_action(self):
        f = self.get_gittins()
        if np.random.random() < self.eps:  # eps greedy
            return np.random.randint(f.shape[0])
        f += np.random.random(f.shape) * 1e-12
        return np.argmax(f)
    
agent = ExactGittinsIndexAgent()
last_bandit = None
total_reward = 0
sums_of_reward = None
numbers_of_selections = None

def exec(observation, configuration):
    global agent, last_bandit, total_reward, sums_of_reward, numbers_of_selections

    if observation.step == 0:
        agent.init_actions(configuration["banditCount"])

    if last_bandit is not None:
        reward = observation.reward - total_reward
        total_reward += reward
        rival_id = 1 - observation.agentIndex
        rival_move = observation.lastActions[rival_id]
        agent.update(last_bandit, reward, rival_move)

    last_bandit = agent.get_action()
    
    return int(last_bandit)