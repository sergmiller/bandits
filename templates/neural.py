import numpy as np
import torch
from torch import nn
import torch.optim as optim

from abc import ABCMeta, abstractmethod

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


class NeuralAgent(AbstractAgent):
    def __init__(self):
        # exact gittins params
        alpha = 0.125
        distortion_horizon = 1.01
        # model params
        lr = 1e-3
        hidden = 128
        # gittins params
        beta = 0.4613730907562291
        # gittins & exact params
        p = 0.5451383211748958
        q = 0.5821590019751394
        # common bandit params
        drift = -0.0011519703027092387
        eps = 0.04165963371245352
        decay = 0.03
        input_f = 11 + 5 + 5 + 5
        {}
        self.horizon = 2000 # fixed by game
        self.input_f = input_f
        self.hidden = hidden
        self.out = 1
        self.lr = lr
        self.model = nn.Sequential(
            nn.Softsign(),
            nn.Linear(self.input_f, self.hidden),
            nn.Sigmoid(),
            nn.Linear(self.hidden, self.out)
        )
        self.alpha = alpha
        self.distortion_horizon = distortion_horizon
        self.p = p
        self.q = q
        self.beta = beta
        self._rival_drift = drift
        self._decay = decay
        self.eps = eps
        self.c = -np.log(self.beta)
        self.loss = nn.BCEWithLogitsLoss()
        self.last_prediction = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.five_last_moves_queue = []
    
    def get_gittins(self):
        p = self.p + self._successes
        q = self.q + self._failures
        n = p + q
        mu = p / n
        
        gittins = mu + mu * (1 - mu) / \
                        (n * np.sqrt((2 * self.c + 1 / n) * mu * (1 - mu)) + mu - 1/2)
        
        return gittins
    
    
    def get_exact_gittins(self):
        p = self._successes + self.p
        q = self._failures + self.q
        n = p + q
        
        m = max((self.distortion_horizon * self.horizon) - self._total_pulls + 1, 1)
        mu = float(m) / n

        gittins = p / n  + np.sqrt((2. * self.alpha) / n * np.log(mu / np.sqrt(np.maximum(1e-9, np.log(mu)))))
        
        gittins[n < 1] = float('+inf')
      
        return gittins
    
    def add_linear_atoms(self, v):
        return v + self._rival_drift * self._rival_moves - self._decay * self._successes
        

    def get_action(self):
        p = self.p + self._successes
        q = self.q + self._failures
        n = p + q
        mu = p / n
        
        gittins = self.get_gittins()
        exact_gittins = self.get_exact_gittins()
                        
        gittins = self.add_linear_atoms(gittins)
        
        total_pulls = np.ones_like(p) * self._total_pulls
        
        thompson = np.random.beta(p, q)
        
        remaining = self.horizon - total_pulls
        
        remap = lambda stream : [x.reshape(-1, 1) for x in stream]
        
        dense_five_last_rival_moves = np.zeros((p.shape[0], 5))
        dense_five_last_my_moves = np.zeros((p.shape[0], 5))
        dense_five_last_my_rewards = np.zeros((p.shape[0], 5))
        
        for i, e in enumerate(self.five_last_moves_queue[::-1]):
            dense_five_last_rival_moves[e[0]][i] = 1
            dense_five_last_my_moves[e[1]][i] = 1
            dense_five_last_my_rewards[e[2]][i] = 1

        
        vectors = np.concatenate(remap((p, q, n, mu, gittins, exact_gittins, total_pulls, remaining, self._successes, self._failures, self._rival_moves)) +  
            [dense_five_last_rival_moves, dense_five_last_my_moves, dense_five_last_my_rewards],axis=1)
        
        {}
         
        self.optimizer.zero_grad()
        vectors = torch.Tensor(vectors)
        out = self.model(vectors)
        out += torch.Tensor(gittins).reshape(out.shape) # baseline
        out += torch.rand(out.shape) * 1e-12  # random
        prediction = torch.argmax(out, dim=0)
        if np.random.random() < self.eps:  # eps greedy
            prediction = np.random.randint(out.shape[0])
        self.last_prediction = out[prediction]
        return prediction
    
    def update(self, action, reward, rival_move=None):
        super().update(action, reward, rival_move)
        if len(self.five_last_moves_queue) >= 5:
            self.five_last_moves_queue = self.five_last_moves_queue[1:]
        self.five_last_moves_queue.append([rival_move, action, reward])
        
        out = self.loss(self.last_prediction.reshape(-1,1), torch.FloatTensor([reward]).reshape(-1,1))
        out.backward() 
        self.optimizer.step()
        

agent = NeuralAgent()
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
