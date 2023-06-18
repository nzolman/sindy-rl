import gym
import numpy as np
import pysindy as ps
import warnings
from tqdm import tqdm
import os
from scipy.integrate import solve_ivp

from sindy_rl.refactor.sindy_utils import build_optimizer, build_feature_library
from sindy_rl.refactor import dynamics_callbacks

class BaseDynamicsModel:
    '''
    Parent class for dynamics models. Each subclass must implement it's own 
        predict and fit functions. 
    '''
    def __init__(self, config):
        raise NotImplementedError
    def predict(self, x, u):
        raise NotImplementedError
    def fit(self, X, U):
        raise NotImplementedError

class EnsembleSINDyDynamicsModel(BaseDynamicsModel):
    '''
    An ensemble SINDy Dynamcis Model. Note that all SINDy models can be an ensemble 
        of just 1 model. 
    '''
    def __init__(self, config): 
        # TO DO: BUILD FEATURE LIBRARY AND OPTIMIZER
        self.config = config or {}
        self.dt = self.config.get('dt', 1)
        self.discrete = self.config.get('discrete', True)
        self.callbacks = self.config.get('callbacks', None)
        
        if self.callbacks is not None:
            self.callbacks = getattr(dynamics_callbacks, self.callbacks)
        
        # init optimizer
        optimizer = self.config.get('optimizer')
        if isinstance(optimizer, ps.BaseOptimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = build_optimizer(optimizer)
        
        # init feature library
        self.feature_library = self.config.get('feature_library', ps.PolynomialLibrary())
        
        if isinstance(self.feature_library, dict):
            self.feature_library = build_feature_library(self.feature_library)

        self.model = ps.SINDy(discrete_time= self.discrete, 
                              optimizer=self.optimizer, 
                              feature_library=self.feature_library)
        self.n_models = self.model.optimizer.n_models
        
        self.safe_idx = None

    def reset_safe_list(self):
        self.safe_idx = np.ones(self.n_models, dtype=bool)
        
    def fit(self, observations, actions, **sindy_fit_kwargs):
        '''
        Inputs:
            `observations': list of observations (state variables) of the form [traj_1, traj_2, ... traj_N]
                where each trajectory is an (n_time, n_state) dimensional array
                (this is in the form of pysindy's `multiple_trajecotries=True` keyword)
            `actions': list of actions (control inputs) of the form [traj_1, traj_2, ... traj_N]
                where each trajectory is an (n_time, n_inputs) dimensionla array
                (this is in the form of pysindy's `multiple_trajecotries=True` keyword)
            `sindy_fit_kwargs': keyword arguments to be passed to pysindy's `model.fit' function.
        Returns: 
            `model': Fitted PySINDy model
        '''
        self.optimizer.coef_list = []
        self.model.fit(observations, u=actions, multiple_trajectories=True, t=self.dt, **sindy_fit_kwargs)
        self.safe_idx = np.ones(self.n_models, dtype=bool)
        return self.model
    
    def validate_ensemble(self, x0, u_data, targets, thresh = None, verbose = True, **sim_kwargs):
        '''
        Validates which members of the ensemble are usuable with respect to a given test data.
        
        Inputs: 
            `x0': initial state for a trajectory `ndarray' of shape (n_state,)
            `u_data': the associated control inputs (actions) to apply during the simulation.
            `targets`: the associated expected predictions
            `thresh`: (float) scalar threshold for MSE. If MSE exceeds threshold, report that the
                dynamics model is not safe.
            `verbose': whether to use a tqdm progress bar.
        Ouptus:
            `pred_list': ndarray of shape (n_ensmble, n_time, n_state) -ish
                The set of predictions. A given ensemble index reports none if that member fails
                to be safe.
            `safe_idx': bool ndarray of shape (n_ensemble,). If an index is `True`: the corresponding
                model is assumed to be safe. If it is `False`, the corresponding model is assumed to be
                not safe.
        '''
        coef_list = self.get_coef_list()
        pred_list = []
        for coef_idx, coef in enumerate(tqdm(coef_list, disable=(not verbose), position=0, leave=True)):
            try:
                self.model.optimizer.coef_ = coef
                preds = self.simulate(x0, u=u_data, t = len(u_data), **sim_kwargs)
                mse = np.mean((targets - preds)**2)
                pred_list.append(preds)
                if thresh and mse >= thresh:
                    self.safe_idx[coef_idx] = False
            except Exception as e:
                warnings.warn(str(e))
                self.safe_idx[coef_idx] = False
                pred_list.append(None)
        return np.array(pred_list), self.safe_idx
    
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
        self.model.optimizer.coef_ = np.mean(coef_list, axis=0)
        return self.model.optimizer.coef_

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
        self.model.optimizer.coef_ = np.median(coef_list, axis=0)
        
        return self.model.optimizer.coef_
        
    def set_idx_coef_(self, idx):
        '''
        Set the model coefficients to be the `idx`-th ensemble coefficient
        
        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs: 
            `coef_`: the ensemble `idx`-th ensemble coefficient
        '''
        self.model.optimizer.coef_ = self.model.optimizer.coef_list[idx]
        return self.model.optimizer.coef_
    
    def set_rand_coef_(self, valid = True):
        '''
        Set the model coefficients to be a random element of the ensemble.
        
        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs: 
            `coef_`: the random ensemble coefficient
        '''
        idx_list = np.arange(self.n_models)
        if valid:
            idx_list = idx_list[self.safe_idx]
        
        idx = np.random.choice(idx_list)
        
        return self.set_idx_coef_(idx)
    
    def simulate(self, x0, u,t=None, upper_bound=1e6, **kwargs):
        '''
        Discrete: 
            Wrapper for pysindy model simulator
        Continuous: 
            faster implementation using zero-hold control
            (reduced accuracy, increased chance of divergence)
        
        Inputs: 
            x0: ndarray with shape (n_state,) or (1, n_state)
                initial state to simulate forward
            u: ndarray with shape (n_time, n_inputs)
                control inputs (actions) to simulate forward
            kwargs: key word arguments for pysindy model.simulate
            
        Note: 
            must pass a `t` kwarg <= len(u).
        Returns: 
            ndarray of states with shape (n_time, n_state)
        '''
        if self.discrete:
            x_list = self.model.simulate(x0, u=u, t = t, **kwargs)
            if np.any(np.abs(x_list) > upper_bound):
                raise ValueError('Bound exceeded. Likely integration blowup')
        else:
            # zero-hold control
            x_list = [x0]
            for i, ui in enumerate(u):
                update = solve_ivp(self._dyn_fn,
                                y0 = x_list[-1],
                                t_span = [0, self.dt],
                                args=(np.array([ui]),),
                                **kwargs
                                ).y.T[-1]
                x_list.append(update)
                if np.any(np.abs(update) > upper_bound):
                    raise ValueError('Bound exceeded. Likely integration blowup')
            x_list.pop(-1)
        return np.array(x_list)
            # return NotImplementedError

    def _dyn_fn(self, t, x, u=None):
        return self.model.predict(x.reshape(1,-1), u=u.reshape(1,-1))
    
    def get_coef_list(self):
        ''''
        Get list of model coefficients.
        
        (Wrapper for pysindy optimizer `coef_list` attribute.)
        '''
        return self.model.optimizer.coef_list
    
    def predict(self, x, u):
        '''
        The one-step predictor (wrapper for pysindy simulator)
        '''
        if self.discrete:
            update = (self.simulate(x, np.array([u]), t =2))[-1]
        else:
            update = solve_ivp(self._dyn_fn,
                                y0 = x,
                                t_span = [0, self.dt],
                                args=(np.array([u]),)).y.T[-1]
        if self.callbacks is not None:    
            update = self.callbacks(update)
        return update
    def print(self):
        '''wrapper for pysindy print'''
        self.model.print()
        
    def set_ensemble_coefs_(self, weight_list):
        self.model.optimizer.coef_list = weight_list
        self.model.optimizer.coef_ = self.set_idx_coef_(0)