import numpy as np
import json

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
        make_event = lambda x : Event(
            x['reward'],
            x['lastActions'][x['agentIndex']],
            x['lastActions'][1 - x['agentIndex']],
            x['thresholds'])
        self.events = [make_event(x[0]['observation']) for x in self.d['steps'][1:]]  # skip zero step(no info there)


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
