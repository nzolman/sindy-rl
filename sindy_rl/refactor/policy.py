from gym.spaces.box import Box

class BasePolicy:
    '''Parent class for policies'''
    def __init__(self):
        raise NotImplementedError
    def compute_action(self, obs):
        raise NotImplementedError
    

class RLlibPolicyWrapper(BasePolicy):
    '''Wraps an RLlib algorithm into a BasePolicy class'''
    def __init__(self, algo, mode = 'algo'):
        self.algo = algo
        self.mode = mode
    def compute_action(self, obs, explore = False):
        res = self.algo.compute_single_action(obs, explore=explore)
        if self.mode == 'policy':
            res = res[0]
        return res
        

class RandomPolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self,action_space = None, low=None, high=None, seed=0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        if action_space: 
            self.action_space = action_space
        else:
            self.action_space = Box(low=low, high=high) #action_space
        self.action_space.seed(seed)
        self.magnitude = 1.0
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        return self.magnitude * self.action_space.sample()
    
    def set_magnitude_(self, mag):
        self.magnitude = mag

