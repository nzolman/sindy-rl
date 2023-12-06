import numpy as np
import warnings
from scipy.optimize import minimize
from tqdm import tqdm
from sindy_rl.refactor.env import safe_step

from sindy_rl.refactor.policy import RandomPolicy, BasePolicy
from sindy_rl.refactor.env import rollout_env
from sindy_rl.refactor.dynamics import EnsembleSINDyDynamicsModel


def sparse_variance(coefs, feature_library, X, state_size=5):
    '''TO-DO: fix the number of'''
    phi_X = feature_library.transform(X)
    covs = np.array([np.cov(coefs[:,i,:].T) for i in range(state_size)])
    var_tot = np.einsum('nd,edf,fn->n',phi_X, covs, phi_X.T)
    return var_tot


def mpc_explore(
        dyn_model, 
        x0=None,
        horizon=10,
        u_dim = 1,
        n_hold =1, # zero-hold control
        gamma = 0.99, # discount factor
        **min_kwargs):
    
    '''
    TO-DO:
        - Do something about integration blowup
    '''
    x_dim = x0.shape[0]
    feature_library = dyn_model.model.feature_library
    coefs = np.array(dyn_model.get_coef_list())
    covs = np.array([np.cov(coefs[:,i,:].T) for i in range(x_dim)])
    gammas = gamma ** np.arange(horizon*n_hold)
    
    def mpc_obj(u_t_flat, control_pen = 0.001):
        '''this is the MPC objective to optimize'''
        
        u_t = u_t_flat.reshape(horizon,u_dim)
        # u_t = np.clip(u_t, u_bounds[0], u_bounds[1])
        
        # do something about resetting!
        obs_list = [x0]
        u_list = []
        for u in u_t:
            # zero-hold control
            for k in range(n_hold): 
                try: 
                    x_t1 = dyn_model.predict(obs_list[-1], u)
                except ValueError as e:
                    warnings.warn(str(e))
                    return np.inf
                obs_list.append(x_t1)
                u_list.append(u)

        obs_list.pop(-1)
        obs_list = np.array(obs_list)  
        u_list = np.array(u_list) 

        X_stack = np.concatenate([obs_list, u_list], axis=1)
        phi_X = feature_library.transform(X_stack)
        var_tot = np.einsum('nd,edf,fn->n',phi_X, covs, phi_X.T)
        return -1*np.dot(var_tot, gammas) 
    
    # choose the first value
    u0 = np.random.uniform(-1, 1, size =horizon * u_dim)
    res = minimize(mpc_obj, u0, **min_kwargs)
    return res


def mpc(
        dyn_model, 
        x0=None,
        horizon=10,
        u_dim = 1,
        n_hold =1, # zero-hold control
        gamma = 0.99, # discount factor,
        rew_fn = None,
        uq_pen = 0.001,
        **min_kwargs):
    '''
    TO-DO:
        - Do something about integration blowup
    '''
    x_dim = x0.shape[0]
    gammas = gamma ** np.arange(horizon*n_hold)
    
    # there's probably an argument for passing the covariance matrix directly. 
    feature_library = dyn_model.model.feature_library
    coefs = np.array(dyn_model.get_coef_list())
    covs = np.array([np.cov(coefs[:,i,:].T) for i in range(x_dim)])
    
    def mpc_obj(u_t_flat):
        '''this is the MPC objective to optimize'''
        
        u_t = u_t_flat.reshape(horizon,u_dim)
        # u_t = np.clip(u_t, u_bounds[0], u_bounds[1])
        
        # do something about resetting!
        obs_list = [x0]
        u_list = []
        for u in u_t:
            # zero-hold control
            for k in range(n_hold):
                try: 
                    x_t1 = dyn_model.predict(obs_list[-1], u)
                except ValueError as e:
                    warnings.warn(str(e))
                    return np.inf
                obs_list.append(x_t1)
                u_list.append(u)

        rew_list = np.array([rew_fn(x,u) for x,u in zip(obs_list[1:], u_list)])

        obs_list.pop(-1)
        obs_list = np.array(obs_list)  
        u_list = np.array(u_list) 

        X_stack = np.concatenate([obs_list, u_list], axis=1)
        phi_X = feature_library.transform(X_stack)
        var_tot = np.einsum('nd,edf,fn->n',phi_X, covs, phi_X.T)
        return uq_pen*np.dot(var_tot, gammas)  - np.dot(rew_list, gammas)
    
    # choose the first value
    u0 = np.random.uniform(-1, 1, size =horizon * u_dim)
    res = minimize(mpc_obj, u0, **min_kwargs)
    return res



class SINDyMPCExplorePolicy(BasePolicy):
    def __init__(self, dyn_model, horizon=1, u_dim=1, minimizer_kwargs=None, 
                 n_hold=1, gamma=0.99):
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.u_dim = u_dim
        self.n_hold = n_hold
        self.gamma = gamma
        self.minimizer_kwargs = minimizer_kwargs or {}
        self.n_steps = 0
        
    def compute_action(self, obs):
        # only query the MPC every n_hold steps.
        if self.n_steps % self.n_hold == 0:
            self.res = mpc_explore(self.dyn_model,
                        x0 = obs,
                        horizon=self.horizon,
                        u_dim = self.u_dim,
                        n_hold=self.n_hold,
                        gamma = self.gamma,
                        **self.minimizer_kwargs
                        )
        
        self.n_steps += 1
        return np.array(self.res.x[:self.u_dim], dtype=np.float32).reshape(self.u_dim)
    
    


class SINDyMPCExploitPolicy(BasePolicy):
    def __init__(self, dyn_model, horizon=1, u_dim=1, minimizer_kwargs=None, 
                 n_hold=1, gamma=0.99, rew_fn=None, uq_pen = 0.001,):
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.u_dim = u_dim
        self.n_hold = n_hold
        self.gamma = gamma
        self.minimizer_kwargs = minimizer_kwargs or {}
        self.n_steps = 0
        self.rew_fn = rew_fn
        self.uq_pen = uq_pen
        
    def compute_action(self, obs):
        # only query the MPC every n_hold steps.
        if self.n_steps % self.n_hold == 0:
            self.res = mpc(self.dyn_model,
                        x0 = obs,
                        horizon=self.horizon,
                        u_dim = self.u_dim,
                        n_hold=self.n_hold,
                        gamma = self.gamma,
                        uq_pen = self.uq_pen,
                        rew_fn = self.rew_fn,
                        **self.minimizer_kwargs
                        )
        
        self.n_steps += 1
        return np.array(self.res.x[:self.u_dim], dtype=np.float32).reshape(self.u_dim)
    
    

def rollout_env_no_reset(env, x0, policy, n_steps, verbose = False, env_callback = None):
    '''
    returns lists of obs, acts, rews trajectories
    '''
       
    obs_list = [x0]

    act_list = []
    rew_list = []
    
    trajs_obs = []
    trajs_acts = []
    trajs_rews = []

    for i in tqdm(range(n_steps), disable=not verbose):
        action = policy.compute_action(obs_list[-1])
        obs, rew, done, info = safe_step(env.step(action))
        act_list.append(action)
        obs_list.append(obs)
        rew_list.append(rew)
        
        if env_callback:
            env_callback(i, env)
            
    if len(act_list) != 0:
        obs_list.pop(-1)
        trajs_obs.append(np.array(obs_list))
        trajs_acts.append(np.array(act_list))
        trajs_rews.append(np.array(rew_list))
    return trajs_obs, trajs_acts, trajs_rews



def active_learning_vanilla(env, dyn_config, n_refits = 10, n_rollout = 100, 
                            horizon=5, n_hold = 5, u_dim=1, other_min_kwargs=None, ignore_reset=False, 
                            always_random = False):
    other_min_kwargs = other_min_kwargs or {}
    min_kwargs = {'method' :'Nelder-Mead', 
            'bounds' : [(-2,2) for i in range(horizon)],
            }
    min_kwargs.update(other_min_kwargs)
    big_coefs = []
    obs_list = []
    act_list = []
    rew_list = []

    for i in range(n_refits+1):
        
        # take random actions at the beginning
        if i ==0:
            dyn_model = EnsembleSINDyDynamicsModel(dyn_config)
            policy = RandomPolicy(env.action_space)
            x0 = env.reset()[0]

        # explore otherwise
        else:
            dyn_model.set_median_coef_()
            if not always_random:
                policy = SINDyMPCExplorePolicy(dyn_model, 
                                        horizon=horizon, 
                                        u_dim = u_dim, 
                                        n_hold = n_hold,
                                        minimizer_kwargs=min_kwargs)
            x0 = new_obs[-1][-1]
        
        # collect experience
        if not ignore_reset:
            new_obs, new_acts, new_rews = rollout_env(env, 
                                                    policy, 
                                                    n_steps=n_rollout, 
                                                    verbose=True)
        else:
            new_obs, new_acts, new_rews = rollout_env_no_reset(env, 
                                                                x0=x0,
                                                                policy=policy, 
                                                                n_steps=n_rollout, 
                                                                verbose=True)
        obs_list += new_obs
        act_list += new_acts
        rew_list += new_rews
        
        # refit
        dyn_model.fit(obs_list,
                    act_list
        )
        dyn_model.set_median_coef_()
        coefs = np.array(dyn_model.get_coef_list())
        big_coefs.append(coefs)
        
    return {'coefs': big_coefs, 'obs': obs_list, 
            'act': act_list, 'rew': rew_list, 'dyn': dyn_model}