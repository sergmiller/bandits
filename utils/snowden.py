import numpy as np
import json
import os
import tqdm

import utils.bandits as bandits


class Event:
    def __init__(self, reward, action, rival_move, thresholds):
        self.reward = reward
        self.action = action
        self.rival_move = rival_move
        self.thresholds = thresholds

class Dataset:
    def __init__(self, X, y, sessions,actions=None):
        self.X = X
        self.y = y
        self.actions=actions
        self.sessions = sessions
    def save_dataset(self, file : str):
        r = dict()
        r['X'] = self.X
        r['y'] = self.y
        r['sessions'] = self.sessions
        r['actions'] = self.actions
        np.save(file, r)

    @classmethod
    def load_dataset(cls, file : str):
        r = np.load(file, allow_pickle=True).item()
        return cls(r['X'], r['y'], r['sessions'], r.get('actions', None))

class Session:
    def __init__(self, d : dict, file : str):
        self.d = d
        self.file = file
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

    def __str__(self):
        return self.file


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
    return Session(j, file)


def get_all_nested_sessions_in_dir(dir : str)->list:
    ffiles = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-5:] != '.json':
                continue
            ffiles.append(os.path.join(root, file))
    return ffiles

def read_sessions_in_dir(dir : str):
    files = get_all_nested_sessions_in_dir(dir)
    print(dir, files)
    sessions = []
    for f in files:
        try:
            s = parse_json_session(f)
            sessions.append(s)
        except Exception as e:
            print('error while parse session ' + str(f) + ' : ' + str(e))
            continue
    return sessions


def make_dataset(agent_class, sessions : list, n_jobs = 1):
    from joblib import Parallel, delayed

    with Parallel(n_jobs=n_jobs) as parallel:
        res = parallel(delayed(collect_dataset)(agent_class, s) for s in tqdm.tqdm(sessions))

    X,Y,S,A = [],[],[],[]
    for s,r in zip(sessions,res):
        x,y,a = r
        X.append(x)
        Y.append(y)
        S.append(s.file)
        A.append(a)

    F = X[0].shape[-1]

    X = np.array(X).reshape(-1, F)
    Y = np.array(Y).reshape(-1)
    S = np.array(S)
    A = np.array(A)
    return Dataset(X,Y,S,A)


def collect_dataset_from_dir(agent_class, dir : str, val_ratio = 0, n_jobs=1):
    sessions = read_sessions_in_dir(dir)

    np.random.seed(0)
    np.random.shuffle(sessions)

    split = int(len(sessions) * val_ratio)
    val_sessions = sessions[:split]
    train_sessions = sessions[split:]

    t = make_dataset(agent_class, train_sessions, n_jobs)
    v = make_dataset(agent_class, val_sessions, n_jobs)
    return t, v

def resample_eq(d, size=None):
    if size is None:
        size = int(d.y.shape[0] * np.mean(d.y))
    print('mean before resampling: {}'.format(np.mean(d.y)))
    samples_1 = np.random.choice(np.where(d.y)[0], size=size)
    samples_0 = np.random.choice(np.where(1 - d.y)[0], size=size)
    X = np.concatenate([d.X[samples_1],d.X[samples_0]], axis=0)
    y = np.concatenate([d.y[samples_1],d.y[samples_0]], axis=0)
    print('mean after resampling: {}'.format(np.mean(y)))
    return Dataset(X,y,None)


def collect_dataset(agent_class, s : Session):
    '''
        agent also should provide methods:
            get_action() - to make actions(and some precalcs), result not used
            get_features_dump() -  to get sample for current action(1D np.array)
            init_actions(N) - N is number of bandits,
            update(last_action, reward, rival_move) - to make internal update
    '''

    agent = agent_class()

    agent.init_actions(100)  # default number of bandits
    X = []
    y = []
    a = []

    for e in s.events:
        agent.get_action()
        f = agent.get_features_dump(e.action)
        agent.update(e.action, e.reward, e.rival_move)
        a.append(e.action)
        X.append(f)
        y.append(e.reward)
    X = np.array(X)
    y = np.array(y)
    a = np.array(a)
    return (X,y,a)
