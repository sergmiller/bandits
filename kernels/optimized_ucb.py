import random
import numpy as np

class UCBAgent:
    # Optimal Settings:
    # exploration=12,  opp_reward=0.6, warmup=1, choose='max'
    # exploration=0.1, opp_reward=0.1, warmup=3, choose='random'
    def __init__(self, exploration=0.2,  opp_reward=0.2, warmup=2, winrate=0.8, choose='max', verbose=False):
        self.exploration = exploration
        self.choose      = choose
        self.opp_reward  = opp_reward
        self.warmup      = warmup
        self.winrate     = winrate
        self.verbose     = verbose
        self.history     = None
        self.state       = None

        
    def init_state(self, observation, configuration, force=False):
        if self.state is None or force:
            self.history = {
                "actions":  [],
                "opponent": [],
                "reward":   [],
            }
            self.state = {
                "our_rewards":  np.zeros(configuration.banditCount, dtype=np.float),
                "opp_rewards":  np.zeros(configuration.banditCount, dtype=np.float),
                "our_visits":   np.zeros(configuration.banditCount, dtype=np.float),
                "opp_visits":   np.zeros(configuration.banditCount, dtype=np.float),                
                "total_visits": np.zeros(configuration.banditCount, dtype=np.float),                
            }        
        
        
    def update_state(self, observation, configuration):
        if self.state is None:
            self.init_state(observation, configuration)
        
        self.history['reward'].append( observation.reward )
        if len(self.history['actions']):
            # observation.reward is cumulative reward
            our_reward      = int(self.history['reward'][-1] > self.history['reward'][-2])
            our_last_action = self.history['actions'][-1]
            if len( set(observation.lastActions) ) == 1:
                opp_last_action = our_last_action
            else:
                opp_last_action = list( set(observation.lastActions) - {our_last_action} )[0]
            self.history['opponent'].append(opp_last_action)

            self.state['our_rewards'][  our_last_action ] += our_reward
            self.state['opp_rewards'][  opp_last_action ] += self.opp_reward
            self.state['our_visits'][   our_last_action ] += 1
            self.state['opp_visits'][   opp_last_action ] += 1
            self.state['total_visits'][ our_last_action ] += 1
            self.state['total_visits'][ opp_last_action ] += 1
            
    
        
    def scores(self, observation, configuration):
        total_visits = np.sum(self.state['our_visits']) + 1
        our_visits   = np.max([ self.state['our_visits'], np.ones(len(self.state['our_visits'])) ])
        scores = (
            (self.state['our_rewards'] + self.state['opp_rewards']) / our_visits 
            + np.sqrt( self.exploration * np.log(total_visits) / our_visits )
        )
        scores *= configuration.decayRate ** self.state['total_visits']
        return scores

        
    # observation   {'remainingOverageTime': 60, 'step': 1, 'reward': 1, 'lastActions': [54, 94]}
    # configuration {'episodeSteps': 2000, 'actTimeout': 0.25, 'runTimeout': 1200, 'banditCount': 100, 'decayRate': 0.97, 'sampleResolution': 100}
    def agent(self, observation, configuration):

        self.update_state(observation, configuration)

        scores = self.scores(observation, configuration)

        winners  = np.argwhere( (self.state['our_visits'] != 0) 
                              & ( 
                                    (self.state['our_visits'] <= self.state['our_rewards'] + (self.warmup - 1)) 
                                  | (
                                      np.nan_to_num(self.state['our_rewards'] / self.state['our_visits']) 
                                      >= self.winrate * configuration.decayRate ** (observation.step/configuration.banditCount)
                                    )
                                )
                              ).flatten()
        untried  = np.argwhere( self.state['our_visits'] == 0).flatten()
        
        if self.warmup and len(winners):
            action = np.random.choice(winners)  # keep trying winners until we lose
        elif self.warmup and len(untried):
            action = np.random.choice(untried)
        else:
            if self.choose == 'random':
                action = random.choices( population=np.arange(len(scores)), weights=scores, k=1 )[0]        
            elif self.choose == 'max':
                action = np.argmax(scores)
            else:
                assert False, self.choose 
                
        if self.verbose:
            if True or observation.step < configuration.banditCount:
                print()
                print('observation = ', observation)
                print(f'scores = {list(scores.round(2))}')
                for key, values in self.state.items():
                    print(f'self.state["{key}"] = {list(values)}')
                print(f'action = {action}')

        self.history['actions'].append(action)
        return int(action)

    
    def __call__(self, observation, configuration):
        return self.agent(observation, configuration)
    
ucb_instance = UCBAgent() 
def ucb_agent(observation, configuration):
    return ucb_instance.agent(observation, configuration)