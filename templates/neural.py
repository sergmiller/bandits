import numpy as np
import torch
from torch import nn
import torch.optim as optim

from scipy.stats import beta

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


class CustomBilinearLayer(nn.Module):
    def __init__(self, f_in, f_out, inner=1, list_a=None, list_b=None, list_bias=None):
        super().__init__()
        if list_a is None:
            assert list_b is None and list_bias is None
            list_a = torch.nn.init.xavier_uniform_(torch.empty(f_in, inner))
            list_b = torch.nn.init.xavier_uniform_(torch.empty(f_out, inner))
            list_bias = torch.nn.init.normal_(torch.empty(f_out))
        self.a = self._create_grad_tensor_from_list(list_a).reshape(f_in, inner)
        self.b = self._create_grad_tensor_from_list(list_b).reshape(f_out, inner)
        self.bias = self._create_grad_tensor_from_list(list_bias).reshape(f_out)
        self.a = nn.Parameter(self.a, requires_grad=True)
        self.b = nn.Parameter(self.b, requires_grad=True)
        self.bias = nn.Parameter(self.bias, requires_grad=True)
    def forward(self, x):
        return torch.matmul(x, torch.matmul(self.a, self.b.T)) + self.bias
    def _create_grad_tensor_from_list(self, l):
        a = torch.FloatTensor(l)
        return a
    
class CustomLinearLayer(nn.Module):
    def __init__(self, f_in, f_out, list_a=None, list_bias=None):
        super().__init__()
        if list_a is None:
            assert list_bias is None
            list_a = torch.nn.init.xavier_uniform_(torch.empty(f_in, f_out))
            list_bias = torch.nn.init.normal_(torch.empty(f_out))
        self.a = self._create_grad_tensor_from_list(list_a).reshape(f_in, f_out)
        self.bias = self._create_grad_tensor_from_list(list_bias).reshape(f_out)
        self.a = nn.Parameter(self.a, requires_grad=True)
        self.bias = nn.Parameter(self.bias, requires_grad=True)
    def forward(self, x):
        return torch.matmul(x, self.a) + self.bias
    def _create_grad_tensor_from_list(self, l):
        a = torch.FloatTensor(l)
        return a


class NeuralAgent(AbstractAgent):
    def _make_nn_wrapper(self):
        self.weights_dict_serialized = None
        if self.seed is not None:
            torch.random.manual_seed(self.seed)
        if self.use_sep_nn:
            self._make_nn_sep()
        else:
            self._make_nn_dense()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def _make_nn_sep(self):
        import json
        weights = dict()
        if self.weights_dict_serialized is not None:
            weights = json.loads(self.weights_dict_serialized)
        
        if len(weights) == 0:
            self.model = nn.Sequential(
                nn.Softsign(),
                CustomLinearLayer(self.input_f, self.hidden),
                nn.Sigmoid(),
                CustomLinearLayer(self.hidden, self.out)
            )
            return 
        self.model = nn.Sequential(
                nn.Softsign(),
                CustomLinearLayer(self.input_f, self.hidden, weights['1_0'], weights['1_1']),
                nn.Sigmoid(),
                CustomLinearLayer(self.hidden, self.out, weights['3_0'], weights['3_1'])
        )
        
    def _make_nn_dense(self):
        self.model = nn.Sequential(
            nn.Softsign(),
            nn.Linear(self.input_f, self.hidden),
            nn.Sigmoid(),
            nn.Linear(self.hidden, self.out)
        )
    
    def __init__(self):
        # exact gittins params
        alpha = 0.125
        distortion_horizon = 1.01
        # model params
        use_sep_nn = False
        sep_components = 1
        lr = 1e-3
        hidden = 128
        top_negatives = 0
        negatives_weight = 1
        # gittins params
        beta = 0.4613730907562291
        # gittins & exact params
        p = 0.5451383211748958
        q = 0.5821590019751394
        # common bandit params
        drift = -0.0011519703027092387
        eps = 0.04165963371245352
        decay = 0.03
        input_f = 14 + 5 + 5 + 5
        batch_size = 1
        use_reinforce = False
        no_reward = 0.3
        seed = None
        loss = nn.BCELoss()
        # bayesian
        c = 3
        
        {}
        self.use_sep_nn = use_sep_nn
        self.sep_components = sep_components
        self.horizon = 2000 # fixed by game
        self.input_f = input_f
        self.hidden = hidden
        self.out = 1
        self.lr = lr
        self.c = c
        self.top_negatives = top_negatives
        self.negatives_weight = negatives_weight
        self.negatives = None
        self.seed = seed
        self.alpha = alpha
        self.distortion_horizon = distortion_horizon
        self.p = p
        self.q = q
        self.beta = beta
        self._rival_drift = drift
        self._decay = decay
        self.eps = eps
        self.c = -np.log(self.beta)
        self.loss = loss
        self.use_reinforce = use_reinforce
        self.last_prediction = None
        self.five_last_moves_queue = []
        self.loss_sum = 0
        self.log_policy_sum = 0
        self.reward_sum = 0
        self.batch_size = batch_size
        self._no_reward = no_reward
        self.thompson_state = None
        self._make_nn_wrapper()
    
    def get_gittins(self):
        p = self.p + self._successes
        q = self.q + self._failures
        n = p + q
        mu = p / n
        
        gittins = mu + mu * (1 - mu) / \
                        (n * np.sqrt((2 * self.c + 1 / n) * mu * (1 - mu)) + mu - 1/2)
        
        return gittins
    
    
    def get_thompson(self):
        p = self.p + self._successes
        q = self.q + self._failures
        
        thompson = np.random.beta(p, q)
        
        return thompson
    
    
    def get_thompson_with_decay_my_and_rival(self):        
        if self.thompson_state is None:
            self.thompson_state = np.ones((self._successes.shape[0], 2))
        
        best_proba = -1
        best_agent = None
        probs = np.zeros_like(self._successes)
        for k in range(self.thompson_state.shape[0]):
            probs[k] = np.random.beta(self.thompson_state[k][0], self.thompson_state[k][1])

        return probs
    
    
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
    
    def get_bayesian_ucb(self):
        p = self._successes + self.p
        q = self._failures + self.q
        n = p + q
        bucb = p / n.astype(float) + beta.std(p, q) * self.c
        return bucb

    def get_action(self):
        p = self.p + self._successes
        q = self.q + self._failures
        n = p + q
        mu = p / n
        
        # bandits
        gittins = self.get_gittins()
        gittins = self.add_linear_atoms(gittins)
        
        exact_gittins = self.get_exact_gittins()
        #exact_gittins = self.add_linear_atoms(exact_gittins)
        
        bucb = self.get_bayesian_ucb()
        
        thompson = self.get_thompson()
        
        thompson_with_decay = self.get_thompson_with_decay_my_and_rival()
        
        # stats
        total_pulls = np.ones_like(p) * self._total_pulls
        
        remaining = self.horizon - total_pulls
        
        remap = lambda stream : [x.reshape(-1, 1) for x in stream]
        
        # history
        dense_five_last_rival_moves = np.zeros((p.shape[0], 5))
        dense_five_last_my_moves = np.zeros((p.shape[0], 5))
        dense_five_last_my_rewards = np.zeros((p.shape[0], 5))

        for i, e in enumerate(self.five_last_moves_queue[::-1]):
#             print(e, p)
            dense_five_last_rival_moves[e[0]][i] = 1
            dense_five_last_my_moves[e[1]][i] = 1
            dense_five_last_my_rewards[e[2]][i] = 1

        
        vectors = np.concatenate(remap((p, q, n, mu, gittins, exact_gittins, thompson, thompson_with_decay, total_pulls, remaining, bucb, self._successes, self._failures, self._rival_moves)) +  
            [dense_five_last_rival_moves, dense_five_last_my_moves, dense_five_last_my_rewards],axis=1)
        
        baseline = gittins
        self.policy_baseline = mu
        {}

        vectors = torch.Tensor(vectors)
        out = self.model(vectors)
        out += torch.Tensor(baseline).reshape(out.shape) # baseline
        out += torch.rand(out.shape) * 1e-12  # random
        prediction = torch.argmax(out, dim=0)
        if self.top_negatives:
            self.negatives = out[torch.argsort(out)[:self.top_negatives]]
        if self.use_reinforce:
            probs = torch.nn.Softmax(dim=0)(out).detach().numpy().reshape(-1)
            prediction = np.random.choice(np.arange(probs.shape[0]), 1, p=probs)[0]
        if np.random.random() < self.eps:  # eps greedy
            prediction = np.random.randint(out.shape[0])
        self.last_prediction = out[prediction]
        self.last_out = out
        
        T_PRINT = 25

        if self._total_pulls == self.horizon - T_PRINT:
            from copy import deepcopy
            self.frozen = deepcopy(self.model)
        
        if self.use_sep_nn and self._total_pulls >= self.horizon - T_PRINT:
            print_f = lambda x : list(x.detach().reshape(-1).numpy())
            order = 0
            for i in [1,3]:
                for j, x in enumerate([self.frozen[i].a, self.frozen[i].bias]):
                    if self._total_pulls == self.horizon - T_PRINT + order:
#                         np.set_printoptions(precision=3)
                        print('layer_' + str(i) + '_' + str(j) + '_' + ','.join(map(lambda x : '%.3f' % x,np.array(print_f(x)))) + '$')
                    order += 1
        
        return prediction
    
    def update(self, action, reward, rival_move=None):
        super().update(action, reward, rival_move)
        
        if reward > 0:
            self.thompson_state[action][0] += 1
        else:
            self.thompson_state[action][1] += self._no_reward
            
        self.thompson_state[action][0] = (self.thompson_state[action][0] - 1) * (1 - self._decay) + 1
        if rival_move is not None:
            self.thompson_state[rival_move][0] = (self.thompson_state[rival_move][0] - 1) * (1 - self._decay) + 1
        
        if len(self.five_last_moves_queue) >= 5:
            self.five_last_moves_queue = self.five_last_moves_queue[1:]
        self.five_last_moves_queue.append([rival_move, action, reward])
        
        if self.use_reinforce:
            act_policy = torch.nn.Softmax(dim=0)(self.last_out)[action]
            self.log_policy_sum += torch.log(act_policy).reshape(-1)
            self.reward_sum += (reward - self.policy_baseline[action]) # baseline
#             self.loss_sum += loss.reshape(-1)
        else:
            good_probs = torch.sigmoid(self.last_prediction.reshape(-1,1))
            good_probs = good_probs * (1 - self._decay)  # remove bias
            loss = self.loss(good_probs, torch.FloatTensor([reward]).reshape(-1,1))
            if self.top_negatives:
                negative_probs = torch.sigmoid(self.negatives.reshape(-1, 1))
                negative_targets = torch.zeros(self.top_negatives).reshape(-1,1)
                negatives_loss = self.loss(negative_probs, negative_targets)
                negatives_loss *= self.negatives_weight / self.top_negatives
                loss += negatives_loss
            self.loss_sum += loss.reshape(-1)
        if self._total_pulls % self.batch_size == 0:
            if self.use_reinforce:
                self.loss_sum = -self.log_policy_sum * self.reward_sum  # policy gradient
                self.reward_sum = 0
                self.log_policy_sum = 0
            self.optimizer.zero_grad()
            self.loss_sum.backward() 
            self.optimizer.step()
            self.loss_sum = 0
        

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
