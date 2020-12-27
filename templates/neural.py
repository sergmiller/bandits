import numpy as np
import torch
from torch import nn
import torch.optim as optim

from scipy.stats import beta

from scipy import integrate

def pdf(p, weights, rewards, normalization=1):
    s = 1
    for weight, reward in zip(weights, rewards):
        if reward == 1:
            s *= (weight * p)
        else:
            s *= (1 - weight * p)
    return s / normalization

GLOBAL_CACHE = dict()

def get_expected_mean_std(weights, rewards):
    global GLOBAL_CACHE
    key = (tuple(weights), tuple(rewards), 'bayes')
    if key in GLOBAL_CACHE:
        return GLOBAL_CACHE[key]
    normalization = integrate.quad(
        lambda x: pdf(x, weights, rewards), 0, 1
    )[0]
    first_order = integrate.quad(
        lambda x: x * pdf(x, weights, rewards, normalization), 0, 1
    )[0]
    second_order = integrate.quad(
        lambda x: x * x * pdf(x, weights, rewards, normalization), 0, 1
    )[0]
    GLOBAL_CACHE[key] = (first_order, (second_order - first_order ** 2) ** 0.5)
    return GLOBAL_CACHE[key]

def beta_estimation(_, rewards):
    global GLOBAL_CACHE
    key = (tuple(rewards), 'beta')
    if key in GLOBAL_CACHE:
        return GLOBAL_CACHE[key]
    trials = len(rewards)
    GLOBAL_CACHE[key] = (beta.mean(1 + sum(rewards), trials+1 - sum(rewards)),
        beta.std(1 + sum(rewards), trials+1 - sum(rewards)))
    return GLOBAL_CACHE[key]

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
    

class NNWithCustomFeatures(nn.Module):
    def __init__(self, INPUT_F, DROP_P, H):
        super().__init__()
        INPUT_F_C = INPUT_F + 2 * INPUT_F
        self.model_ff =  nn.Sequential(
            nn.BatchNorm1d(INPUT_F_C),
            nn.Dropout(DROP_P),
            nn.Linear(INPUT_F_C, H),
            nn.Sigmoid(),
            nn.Dropout(DROP_P),
            nn.Linear(H, 1)
        )

    def forward(self, x):
        lg = torch.log(1 + torch.abs(x))
        sn = torch.sin(x)
        input_x = torch.cat([x, lg, sn], axis=1)
        return self.model_ff(input_x)
    

import sys
import os

# sys.path.append("/kaggle_simulations/agent")
# working_dir = "/kaggle_simulations/agent"
working_dir = "models"
model_name = "nagiss_v2"

nagiss_model = NNWithCustomFeatures(36, 0.05, 256) # 72 for models from v4, before were - 36
nagiss_model.load_state_dict(torch.load(os.path.join(working_dir, model_name)))
nagiss_model.eval()

def save_wrap(f):
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Got exception: ' + str(e))
            return (0, 0)  
    return g
        

L = 45

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
        
        self.model = nn.Sequential(
            nn.Softsign(),
            CustomLinearLayer(self.input_f, self.hidden, weights.get('1_0', None), weights.get('1_1', None)),
            nn.Sigmoid(),
            CustomLinearLayer(self.hidden, self.hidden, weights.get('3_0', None), weights.get('3_1', None)),
            nn.Sigmoid(),
            CustomLinearLayer(self.hidden, self.out, weights.get('5_0', None), weights.get('5_1', None))
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
        input_f = 14 + 5 + 5 + 5 + 4 + 3 #- 8 # saved features
        batch_size = 1
        use_reinforce = False
        no_reward = 0.3
        seed = None
        loss = nn.BCELoss()

        use_mean = False
        # bayesian
        c = 3
        not_learn = False
        use_only_nagiss = False
        use_feature_nagiss = False
        
        {}
        if use_mean:
            input_f *= 2
        if use_feature_nagiss:
            input_f += 1
        self.use_feature_nagiss = use_feature_nagiss
        self.not_learn = not_learn
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
        
        self.use_mean = use_mean
        
        self.discountings_estimation = None
        self.rewards = None
        self.prior_enemy = None
        
        self.use_only_nagiss = use_only_nagiss

    
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
    
    def get_custom_ucb_bandits(self):
        p = self._successes + self.p
        q = self._failures + self.q
        n = p + q
        mu = p / n
        sigma2 = mu * (1 - mu)
        C = 1
        c = 1
        xi = 1
        out = np.zeros((n.shape[0], 4))
        lnt_divide_t_k = np.log(1 + self._total_pulls) / n
        out[:, 0] = np.sqrt(C * lnt_divide_t_k)
        out[:, 1] = np.sqrt(lnt_divide_t_k * np.minimum(0.25, np.sqrt(sigma2) + np.sqrt(2 * lnt_divide_t_k)))
        out[:, 2] = np.sqrt(16 * sigma2 * (1 + 1 / n) * lnt_divide_t_k)
        out[:, 3] = np.sqrt(2 * sigma2 * xi * lnt_divide_t_k) + c * 3 * xi * lnt_divide_t_k
        out += mu.reshape(-1, 1)
        return out
    
    
    def enemy_p_estimate(self, discountings_est, rewards):
#         return get_expected_mean_std(discountings_est, rewards)[0] * (1 - prior_enemy) + \
#             beta_estimation(discountings_est, rewards)[0] * prior_enemy
#         return  beta_estimation(discountings_est, rewards)[0]
        return save_wrap(get_expected_mean_std)(discountings_est, rewards)[0]
    
    def make_update_initial_probs_estimations(self, action, reward, rival_move):
        def up_discount(d, r):
            if len(d) == 0:
                d.append(1)
            else:
                d.append(d[-1] * (1 - self._decay * r))
                
        p_enemy = self.enemy_p_estimate(self.discountings_estimation[rival_move][:L], self.rewards[rival_move][:L])
        if p_enemy is None or np.isnan(p_enemy):
            p_enemy = 0.5
        else:
            p_enemy=np.clip(p_enemy, 0, 1)
        enemy_reward_sample = int(np.random.choice([0, 1], p=(1-p_enemy, p_enemy)))
        enemy_reward_decay_proxy = (1 - np.exp(np.log(1 - self._decay) * p_enemy)) / self._decay

        self.rewards[action].append(reward)
        self.rewards[rival_move].append(enemy_reward_sample)
        
        up_discount(self.discountings_estimation[action], reward)
        up_discount(self.discountings_estimation[rival_move], enemy_reward_decay_proxy)
        

    
    def get_initial_probs_estimations(self):
        N = self._successes.shape[0]
        out = np.zeros((N, 3))
        
        if self.discountings_estimation is None:
            self.discountings_estimation = [[] for _ in range(N)]
            self.rewards = [[] for _ in range(N)]
        
        p = self._successes + self.p
        q = self._failures + self.q
        n = p + q
        prior_enemy = self._rival_moves / (n + self._rival_moves)

        for i in range(N):
            out[i, 0] = save_wrap(get_expected_mean_std)(self.discountings_estimation[i][:L], self.rewards[i][:L])[0]
            out[i, 1] = save_wrap(beta_estimation)(self.discountings_estimation[i][:L], self.rewards[i][:L])[0]
        
        out[:, 2] = out[:, 0] * (1 - prior_enemy) + out[:, 1] * prior_enemy
        return out

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
        
        custom_ucb_bandits = self.get_custom_ucb_bandits()
        
        thompson_with_decay = self.get_thompson_with_decay_my_and_rival()
        
        # stats
        prior_estimations = self.get_initial_probs_estimations()
        
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
        
        # saved : p, q, n, total_pulls, remaining, self._successes, self._failures, self._rival_moves
        
        vectors = np.concatenate(remap((p, q, n, total_pulls, remaining, self._successes, self._failures, self._rival_moves, mu, gittins, exact_gittins, thompson, thompson_with_decay, bucb)) +  
            [dense_five_last_rival_moves, dense_five_last_my_moves, dense_five_last_my_rewards, custom_ucb_bandits, prior_estimations],axis=1)
        
        vectors_mean = np.mean(vectors, axis=0, keepdims=True)
        vectors_mean = np.repeat(vectors_mean, vectors.shape[0],axis=0)
        if self.use_mean:
            vectors = np.concatenate([vectors, vectors - vectors_mean], axis=1)
        
        baseline = gittins
        self.policy_baseline = mu
        self.vectors = vectors
        {}

        vectors = torch.Tensor(vectors)
        
        if self.use_feature_nagiss:
            f = nagiss_model(vectors)
            vectors = torch.cat([vectors, f], dim=1)
        
        if self.use_only_nagiss:
            out = nagiss_model(vectors)
            out += torch.rand(out.shape) * 1e-12  # random
            prediction = torch.argmax(out, dim=0)
            return prediction

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
            for i in [1,3,5]:
                for j, x in enumerate([self.frozen[i].a, self.frozen[i].bias]):
                    if self._total_pulls == self.horizon - T_PRINT + order:
                        print('layer_' + str(i) + '_' + str(j) + '_' + ','.join(map(lambda x : '%.2f' % x,np.array(print_f(x)))) + '$')
                    order += 1
        
        return prediction
    
    def get_features_dump(self, action):
        return self.vectors[action]
    
    def update(self, action, reward, rival_move=None):
        super().update(action, reward, rival_move)
        self.make_update_initial_probs_estimations(action, reward, rival_move)
        
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
        
        if self.use_only_nagiss:
            return
        
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
            if self.not_learn:
                return
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
