from kaggle_environments import make
from scipy.stats import norm

import numpy as np

import hyperopt
import os
if not os.path.exists('tmp'):
    os.mkdir('tmp')
    
base_path = '/Users/sergmiller/Documents/my/bandits'

def p_val(x):
    return 2 * min(norm.cdf(-x), norm.cdf(x))

class Agent:
    def __init__(self, text=None, file=None):
        self.text = text
        self.file = file
    
def init_agent(a : Agent):
    if a.file is None:
        a.file = "tmp/b_{}.py".format(np.random.random())
        assert a.text is not None
        write_to(a.file, a.text)

def write_to(f_name, text):
    with open(f_name, "w") as f:
        f.write(text)

def compare(t1 : Agent, t2 : Agent, T=10):
    init_agent(t1)
    init_agent(t2)
    res1 = np.zeros(T)
    res2 = np.zeros(T)
    for i in range(T):
        env = make("mab", debug=True)
        res = env.run([t1.file, t2.file])
        res1[i] = res[-1][0]['reward']
        res2[i] = res[-1][1]['reward']
    delta = res1 - res2
    mu_z = np.mean(delta)
    sigma = np.std(delta)
    z = mu_z / sigma * T ** 0.5
    p = p_val(z)
    return (p, mu_z, sigma, res1, res2)

with open(base_path + '/templates/gittins.py', 'r') as f:
    gittins = f.read()
    

with open(base_path + '/templates/exact_gittins.py', 'r') as f: 
    exact_gittins = f.read()
        
gittins_with_random = gittins.format("{}", "{}", "f += np.random.random(f.shape) * 1e-12")
gittins_with_count_my = gittins_with_random.format("{}", "gittins -= self._decay * self._successes")
gittins_with_count_my_exp = gittins_with_random.format("{}", "gittins -= self._decay * self._successes")
gittins_with_count_rival = gittins_with_random.format("{}", "gittins *= (1 - self._decay * self._rival_moves)")
gittins_with_count_rival_mu = gittins_with_random.format("{}", "gittins -= self._decay * self._rival_moves * mu")
gittins_with_count_my_and_rival =  gittins_with_random.format("{}", 
    "gittins = (gittins - self._decay * self._successes) * (1 - self._decay * self._rival_moves)")
gittins_with_count_my_and_rival_mu =  gittins_with_random.format("{}", 
    "gittins -= (self._decay * self._successes + self._decay * self._rival_moves * mu)")
gittins_with_count_rival_drift = gittins_with_random.format("{}", "gittins += self._rival_drift * self._rival_moves")
gittins_with_my_and_count_rival_drift = gittins_with_random.format("{}", 
    "gittins += (self._rival_drift * self._rival_moves - self._decay * self._successes)")
# gittins_with_random_and_custom_params = gittins_with_random.format("beta,p,q=0.5, 0.5, 1", "{}")

exact_gittins_with_my_and_count_rival_drift = gittins_with_random.format("{}", 
    "gittins += (self._rival_drift * self._rival_moves - self._decay * self._successes)")

    
bb = {
    'beta': 0.3648259880510159,
    'rival_drift': -0.0014398709738894131,
    'eps': 0.04635953447584462,
    'p': 0.31270631588673564,
    'q': 0.9205017213361296}

bb_delta = {
    'beta': 0.4613730907562291,
    'rival_drift': -0.0011519703027092387,
    'eps': 0.04165963371245352,
    'p': 0.5451383211748958,
    'q': 0.5821590019751394}


with open(base_path + '/templates/neural.py', 'r') as f:
    neural = f.read()
    
neural_with_new_feature = neural.format(
    "{}\n        input_f += 1", "vectors = np.concatenate([vectors, remap([thompson])[0]], axis=1)")
    

def init_template(tmpl : str, params : dict) -> str:

    s = ",".join(params.keys())
    s += '='
    closures = ",{}" * len(params)
    closures = closures[1:]
    s += closures
    for_insert = s.format(*params.values())
    return tmpl.format(for_insert)

gittins_bb = init_template(gittins_with_my_and_count_rival_drift, bb)
gittins_bb_delta = init_template(gittins_with_my_and_count_rival_drift, bb_delta)


def bench(a : Agent) -> list:
    known_agents = [
        ('gittins', Agent(text=gittins)),
        ('gittins_with_random', Agent(text=gittins_with_random)),
        ('gittins_with_count_my', Agent(text=gittins_with_count_my)),
        ('gittins_with_count_rival_drift', Agent(text=gittins_with_count_rival_drift)),
        ('gittins_with_my_and_count_rival_drift', Agent(text=gittins_with_my_and_count_rival_drift)),
        ('gittins_bb', Agent(text=gittins_bb)),
        ('gittins_bb_delta', Agent(text=gittins_bb_delta)),
        ('softmax_ucb', Agent(file=base_path + '/kernels/softmax_ucb.py')),
        ('multiarmed_bandit_agent', Agent(file=base_path + '/kernels/multiarmed_bandit_agent.py')),
        ('upper_confidence', Agent(file=base_path + '/kernels/upper_confidence.py')),
        ('ucb_decay', Agent(file=base_path + '/kernels/ucb_decay.py')),
        ('bayesian_ucb', Agent(file=base_path + '/kernels/bayesian_ucb.py')),
        ('thompson', Agent(file=base_path + '/kernels/thompson.py')),
        ('neural', Agent(text=neural)),
#         ('max_likelihood', Agent(file=path + 'kernels/max_likelihood.py')), works too long
        ('optimized_ucb', Agent(file=base_path + '/kernels/optimized_ucb.py')),
        ('exact_gittins', Agent(text=exact_gittins)),
    ]
    
    res = []
    for k in known_agents:
        res.append([k[0], compare(a, k[1], T=10)[:3]])
        print(res[-1])
    return res