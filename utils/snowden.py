import numpy as np
import json

import utils.bandits as bandits


class Event:
    def __init__(self, reward, action, rival_move, thresholds):
        self.reward = reward
        self.action = action
        self.rival_move = rival_move
        self.thresholds = thresholds

class Session:
    def __init__(self, d : dict):
        self.d = d
        self._validate(self.d)

        make_event = lambda x, r: Event(
            r,
            x['lastActions'][x['agentIndex']],
            x['lastActions'][1 - x['agentIndex']],
            x['thresholds'])

        total_reward = 0

        self.events = []

        for x in self.d['steps'][1:]:  # skip zero step(no info there)
            reward = x[0]['reward'] - total_reward
            total_reward += reward
            self.events.append(make_event(x[0]['observation'], reward))


    def _validate(self, d):
        assert 'steps' in d
        assert len(d['steps']) == 2000
        assert 'observation' in d['steps'][0][0]
        s = d['steps'][0][0]['observation']
        assert 'reward' in s
        assert 'thresholds' in s
        assert 'step' in s
        assert 'agentIndex' in s
        assert 'lastActions' in s
        assert len(s['thresholds']) == 100

def parse_json_session(file : str) -> Session:
    with open(file, 'r') as f:
        j = json.load(f)
    return Session(j)


def collect_dataset(agent, s : Session):
    '''
        agent also should provide methods:
            get_action() - to make actions(and some precalcs), result not used
            get_features_dump() -  to get sample for current action(1D np.array)
            init_actions(N) - N is number of bandits,
            update(last_action, reward, rival_move) - to make internal update
    '''

    agent.init_actions(100)  # default number of bandits
    X = []
    y = []

    for e in s.events:
        agent.get_action()
        f = agent.get_features_dump(e.action)
        agent.update(e.action, e.reward, e.rival_move)
        X.append(f)
        y.append(e.reward)
    X = np.array(X)
    y = np.array(y)
    return (X,y)
