import numpy as np
from scipy.integrate import solve_ivp
from gym.spaces import Box
from gym import Env

def gt_dyn_pend(t, x, u=0.0, m = 0.1, g = 9.8, l = 0.5, k = 0.1): 
    '''
    Ground Truth pendulum dynamics
    (`scipy.integrate.solve_ivp` compatible)
    '''
    x1, x2 = x
    dx1 = -x2
    dx2 = -(g/l)*np.sin(x1) - (k/m) * x2 + (1/(m*l**2))*u
    return dx1, dx2

def gt_cartpole(t,z, u=None, M=1, m=0.1, L = 0.8, g = 9.8):
    '''
    dynamics for cartpole with force as state
    
    This is a hacky way of approximately keeping track of the force being applied at each 
    step. The other way would be to pass propagated states back through the NN in post
    and calculate the force. 

    Inputs:
    - z: state
    - t: time
    - *params: NN parameters
    '''
    # position (cart), velocity, pole angle, angular velocity
    x1, x2, x3, x4  = z

    dx1 = x2
    dx2 = (1.0 / (L*(M + m * np.sin(x1)**2))) * (-u*np.cos(x1) - m*L*(x2**2)*np.sin(x1)*np.cos(x1) + (M + m)*g*np.sin(x1))
    dx3 = x4
    dx4 = (1.0/(M+ m*np.sin(x1)**2)) * (u + m*np.sin(x1) * (L*(x2**2) - g* np.cos(x1)))
    
    return dx1, dx2, dx3, dx4

class DynEnv(Env): 
    '''
    Basic Gym environment for dynamical systems defined by 
        `scipy.integrate.solve_ivp` compatible dynamics functions
        By default, control terms need to be passible via `u` kwarg
        and control is implemented as zero-hold control over the dt
        interval.
    '''
    def __init__(self, config):
        self.dyn_fn = config['dyn_fn']
        self.dyn_params = config['dyn_params']
        self.dt = config['dt']
        self.n_state = config['n_state']
        self.n_control = config['n_control']
        self.n_steps = 0
        
        self.action_space = Box(low  = -np.inf*np.ones(self.n_control), 
                                high =  np.inf*np.ones(self.n_control))
        self.observation_space = Box(low  = -np.inf*np.ones(self.n_state), 
                                     high =  np.inf*np.ones(self.n_state))
    def is_done(self):
        return False
    
    def get_reward(self):
        return 0

    def _get_obs(self, u):
        self.obs = solve_ivp(self.dyn_fn, 
                             y0 = self.obs, 
                             t_span = [0, self.dt],
                             args = (u, *self.dyn_params,)).y[:,-1]
        return self.obs

    def step(self, u):
        # zero-hold control 
        self.obs = self._get_obs(u)
        done = self.is_done()
        rew = self.get_reward()
        self.n_steps += 1
        return self.obs, rew, done, {}
        
    def reset(self, seed = 0): 
        self.obs = np.zeros(self.n_state)
        self.n_steps = 0
        return self.obs
    
class PendEnv(DynEnv):
    def __init__(self):
        self.m = 0.1
        self.g = 9.8
        self.l = 0.5
        self.k = 0.1
        config = {}
        config['dyn_fn'] = gt_dyn_pend
        config['dyn_params'] = (self.m, self.g, self.l, self.k)
        config['dt'] = 1.0 / 200
        config['n_state'] = 2
        config['n_control'] = 1
        
        super().__init__(config)
        self.max_steps = int(5/self.dt)
        self.low_bounds = np.array([-10,-10])
        self.high_bounds = np.array([10,10])
        
        self.action_space = Box(low  = -10*np.array([1]),
                                high = 10*np.array([1]),
                               )
        self.observation_space = Box(low  = np.array([-np.pi, -np.inf]), 
                                     high =  np.array([np.pi, np.inf]))
        
    def is_done(self):
        if ((self.n_steps >= self.max_steps) 
            or np.any(self.obs < self.low_bounds) 
            or np.any(self.obs > self.high_bounds)
           ):
            return True
        else:
            return False

    def _get_obs(self, u): 
        self.obs = super()._get_obs(u)
        self.obs[0] = ((self.obs[0] + np.pi) % (2*np.pi)) - np.pi
        return self.obs
        
    def reset(self,seed=None):
        if seed is not None:
            self.observation_space.seed(seed)
        self.obs = self.observation_space.sample()
        self.n_steps = 0
        return self.obs
    
    
class Cartpole(DynEnv):
    def __init__(self):
        self.M = 1.0
        self.m = 0.1
        self.L = 0.5 # urban uses 0.5 0.8
        self.g = 9.8
        config = {}
        config['dyn_fn'] = gt_cartpole
        config['dyn_params'] = (self.M, self.m, self.L, self.g )
        config['dt'] = 1.0 / 1000
        config['n_state'] = 2
        config['n_control'] = 1
        
        super().__init__(config)
        self.max_steps = int(5/self.dt)
        self.low_bounds = np.array([-100, -100, -100, -100])
        self.high_bounds =  np.array([100, 100, 100, 100])
        
        self.action_space = Box(low  = -100*np.array([1]),
                                high = 100*np.array([1]),
                               )
        self.observation_space = Box(low  = np.array([-np.pi, -np.inf, -np.inf, -np.inf]), 
                                     high =  np.array([np.pi, np.inf, np.inf, np.inf]))
        
    def is_done(self):
        if ((self.n_steps >= self.max_steps) 
            or np.any(self.obs < self.low_bounds) 
            or np.any(self.obs > self.high_bounds)
           ):
            return True
        else:
            return False

    def _get_obs(self, u): 
        self.obs = super()._get_obs(u)
        self.obs[0] = ((self.obs[0] + np.pi) % (2*np.pi)) - np.pi
        return self.obs
        
    def reset(self,seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.obs = 0.1* self.observation_space.sample()
        # self.obs = 0.1*np.random.randn(4)
        # self.obs[0] += np.pi
        # self.obs[0] = ((self.obs[0] + np.pi) % (2*np.pi)) - np.pi
        self.n_steps = 0
        return self.obs