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

class GittinsIndexAgent(AbstractAgent):
    def __init__(self):
        beta = 0.9
        p = 1
        q = 1
        decay = 0.03
        rival_drift = 0.0001
        eps = 0
        {}
        self.beta = beta
        self.p = p
        self.q = q
        self.c = -np.log(self.beta)
        self.eps = eps
        self._decay = decay
        self._rival_drift = rival_drift
        
    def get_gittins(self):
        '''
        Whittle's approximation
        '''
        p = self.p + self._successes
        q = self.q + self._failures
        n = p + q
        mu = p / n
        
        gittins = mu + mu * (1 - mu) / \
                        (n * np.sqrt((2 * self.c + 1 / n) * mu * (1 - mu)) + mu - 1/2)
        {}            
        return gittins
        
    def get_action(self):
        f = self.get_gittins()
        if np.random.random() < self.eps:  # eps greedy
            return np.random.randint(f.shape[0])
        {}
        return np.argmax(f)
    
gittinsAgent = GittinsIndexAgent()
last_bandit = None
total_reward = 0
sums_of_reward = None
numbers_of_selections = None

def agent(observation, configuration):
    global gittinsAgent, last_bandit, total_reward, sums_of_reward, numbers_of_selections

    if observation.step == 0:
        gittinsAgent.init_actions(configuration["banditCount"])

    if last_bandit is not None:
        reward = observation.reward - total_reward
        total_reward += reward
        #rival_id = 1 - observation.agentIndex  !!! REMOVE COMMENT WHEN KAGGLE UPDATE ENV
        rival_move = None #observation.lastActions[rival_id]
        gittinsAgent.update(last_bandit, reward, rival_move)

    last_bandit = gittinsAgent.get_action()
    
    return int(last_bandit)