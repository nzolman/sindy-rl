from gymnasium.spaces.box import Box
import numpy as np

class BasePolicy:
    '''Parent class for policies'''
    def __init__(self):
        raise NotImplementedError
    def compute_action(self, obs):
        '''given observation, output action'''
        raise NotImplementedError

class FixedPolicy(BasePolicy): 
    '''
    Deterministic policy that provides feedforward control
    from a prescribed sequence of actions
    '''
    def __init__(self, fixed_actions):
        self.fixed_actions = fixed_actions
        self.n_step = 0
        self.n_acts = len(fixed_actions)
    def compute_action(self, obs):
        u = self.fixed_actions[self.n_step % self.n_acts]
        self.n_step += 1
        return u

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



class SparseEnsemblePolicy(BasePolicy):
    '''
    Sparse ensemble dictionary model of the form
    Y = \Theta(X) \Xi
    where the labels Y are control values depending on the states u(x)
    '''
    def __init__(self, optimizer, feature_library, min_bounds = None, max_bounds = None):

        # bounds for the action space
        self.min_bounds = min_bounds 
        self.max_bounds = max_bounds

        self.optimizer = optimizer
        self.feature_library = feature_library
        
    def compute_action(self, obs):
        ThetaX = self.feature_library.transform(obs)
        u = self.optimizer.coef_ @ ThetaX
        
        # clip action
        if self.min_bounds is not None:
            u = np.clip(u, self.min_bounds, self.max_bounds)
        return np.array(u, dtype=np.float32)
        
    def _init_features(self, X_concat):
        '''compute Theta(X)'''
        X = self.feature_library.reshape_samples_to_spatial_grid(X_concat)
        self.ThetaX = self.feature_library.fit_transform(X)
        return self.ThetaX
    
    def fit(self, data_trajs, action_trajs):
        '''Fit ensemble models'''
        X_concat = np.concatenate(data_trajs)
        Y_concat = np.concatenate(action_trajs)
        ThetaX = self._init_features(X_concat)
        self.optimizer.fit(ThetaX, Y_concat)
        return self.optimizer.coef_list
    
    def get_coef_list(self):
        ''''
        Get list of model coefficients.
        
        (Wrapper for pysindy optimizer `coef_list` attribute.)
        '''
        return self.optimizer.coef_list
    
    def set_mean_coef_(self, valid=False):
        '''
        Set the model coefficients to be the ensemble mean.
        
        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs: 
            `coef_`: the ensemble mean coefficients
        '''
        coef_list = np.array(self.get_coef_list())
        if valid:
            coef_list = coef_list[self.safe_idx]
        self.optimizer.coef_ = np.mean(coef_list, axis=0)
        return self.optimizer.coef_

    def set_median_coef_(self, valid=False):
        '''
        Set the model coefficients to be the ensemble median.
        
        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs: 
            `coef_`: the ensemble median coefficients
        '''
        coef_list = np.array(self.get_coef_list())
        if valid:
            coef_list = coef_list[self.safe_idx]
        self.optimizer.coef_ = np.median(coef_list, axis=0)
        
        return self.optimizer.coef_
        
    def set_idx_coef_(self, idx):
        '''
        Set the model coefficients to be the `idx`-th ensemble coefficient
        
        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs: 
            `coef_`: the ensemble `idx`-th ensemble coefficient
        '''
        self.optimizer.coef_ = self.optimizer.coef_list[idx]
        return self.optimizer.coef_
    
    def print(self, input_features=None, precision=3):
        '''
        Analagous to SINDy model print function
        Inputs:
            input_features: (list)
                List of strings for each state/control feature
            precision: (int)
                Floating point precision for printing.
        '''
        lib = self.feature_library
        feature_names = lib.get_feature_names(input_features=input_features)
        coefs = self.optimizer.coef_
        for idx, eq in enumerate(coefs): 
            print_str = f'u{idx} = '
            
            for c, name in zip(eq, feature_names):
                c_round = np.round(c, precision)
                if c_round != 0:
                    print_str += f'{c_round:.{precision}f} {name} + '
            
            print(print_str[:-2])

class OpenLoopSinusoidPolicy(BasePolicy):
    '''Feedforward control outputting a sinewave'''
    def __init__(self, dt=1, amp=1, phase=0, offset=0, f0=1, k=1):
        '''
        Amp * sin(freq * t - phase) + offset
        '''
        self.amp = amp      # amplitude
        self.phase = phase  # phase
        self.offset = offset# offset
        self.f0 = f0        # fundamental frequency
        self.k = k          # wave number
        
        self.dt = dt        # used for updating the time 
        self.t = 0
        
        self.freq = 2 * np.pi * self.k/self.f0
        
        
    def compute_action(self, obs):
        '''
        Return deterministic sine output
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Sinusoidal output depending on the number of calls
            to the policy. 
        '''
        self.t += self.dt
        
        u = self.amp * np.sin(self.freq * self.t -self.phase) + self.offset
        return np.array([u])
    

class OpenLoopSinRest(OpenLoopSinusoidPolicy):
    '''Feedforward Sine wave for some amount of time, then do nothing.
    Used for generating data for Hydrogym environments and geting the decay
    response.
    '''
    def __init__(self, t_rest, **kwargs):
        super().__init__(**kwargs)
        self.t_rest = t_rest
        
    def compute_action(self, obs):
        '''
        Return deterministic sine output, then
        do nothing
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Sinusoidal output depending on the number of calls
            to the policy. 
        '''
        u = super().compute_action(obs)
        
        if self.t >= self.t_rest:
            u = u * 0.0
        return u
    
    
class OpenLoopRandRest(RandomPolicy):
    '''
    Feedforward random actions, then null actions after some
    amount of time.
    '''
    def __init__(self, steps_rest, **kwargs):
        super().__init__(**kwargs)
        self.steps_rest = steps_rest
        self.n_steps = 0
        
    def compute_action(self, obs):
        self.n_steps += 1
        u = super().compute_action(obs)
        
        if self.n_steps >= self.steps_rest:
            u = 0.0 * u
        return u

class SwitchPolicy(BasePolicy):
    '''
    A wrapper that switches between generic policies
    '''
    def __init__(self, policies):
        self.policies = policies
        self.policy = policies[0]
    
    def switch_criteria(self):
        '''Determine when to swap between policies'''
        pass
    
    def compute_action(self, obs):
        policy = self.switch_criteria()
        return policy.compute_action(obs)
        
class SwitchAfterT(SwitchPolicy):
    '''Switch between 2 policies after some amount of time'''
    def __init__(self, t_switch, policies):
        super().__init__(policies)
        self.t_switch = t_switch
        self.n_steps = 0
        
    def switch_criteria(self):
        '''switch policies after some amount of time'''
        self.n_steps += 1
        
        if self.n_steps < self.t_switch: 
            policy_idx = 0
        else: 
            policy_idx = 1
        
        return self.policies[policy_idx]
    
    

class SignPolicy(BasePolicy):
    '''Wrapper for creating a symmetric bang-bang controller from a given policy'''
    def __init__(self,policy, mag=1.0, thresh = 0):
        self.wrapper = policy
        self.mag = mag
        self.thresh = thresh
        
    def compute_action(self, obs):
        # compute action from policy
        action = self.wrapper.compute_action(obs)
        
        # compute whether the action meets a threshold
        mask = np.abs(action) < self.thresh
        
        return np.sign(action)*self.mag * mask
    
    def set_mean_coef_(self):
        self.wrapper.set_mean_coef_()