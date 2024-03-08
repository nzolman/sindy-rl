import numpy as np
import pysindy as ps
import pickle


from sindy_rl.sindy_utils import build_optimizer, build_feature_library
from sindy_rl import reward_fns

class BaseRewardModel:
    '''
    Base Class for the reward model, R = \Theta(X,U) \Xi
    '''
    def __init__(self, config):
        raise NotImplementedError
    def predict(self, x, u):
        raise NotImplementedError
    
    def fit(self, X, U, Y):
        raise NotImplementedError
    
    
class FunctionalRewardModel(BaseRewardModel):
    '''Reward Model when there is an analytic expression'''
    def __init__(self, config):
        '''
        Attributes:
            rew_fn: function
                signature: rew_fn(x, u, **kwargs)
                returns scalar reward for 1-d arrays x,u
        '''
        self.config = config
        self.rew_fn = getattr(reward_fns, config['name'])
        self.rew_kwargs = config.get('kwargs', {})
        self.can_update = False

    def predict(self, x, u):
        return self.rew_fn(x, u, **self.rew_kwargs)

    def fit(self, X, U, Y):
        return None
    
    def save(self, save_path):
        '''EXPERIMENTAL'''
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.config, f)
            
    def load(self, load_path):
        '''Experimental'''
        with open(load_path, 'rb') as f:
            config = pickle.load(f)
        
        self.__init__(config)
        

class EnsembleSparseRewardModel(BaseRewardModel):
    '''
    An ensemble sparse model. Analagous to an Ensemble SINDy model,
        this fits a scalar valued reward function using sparse symbolic
        regression. In theory, this should be compatible with all the
        pysindy optimizers, though not all have been tested. 
        
    R = Theta(X,U) @ \Xi
    '''
    def __init__(self, config):
        self.can_update = True
        
        self.config = config
        self.use_control = config.get('use_control', False)
        
        optimizer = self.config.get('optimizer')
        
        if isinstance(optimizer, ps.BaseOptimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = build_optimizer(optimizer)
            
        # init feature library
        self.feature_library = self.config.get('feature_library', ps.PolynomialLibrary())
        
        if isinstance(self.feature_library, dict):
            self.feature_library = build_feature_library(self.feature_library)
        self._is_fitted = False
        self.n_models = self.optimizer.n_models

    def fit(self, X, Y, U=None, init=False):
        '''
        Fit the Ensemble Model
        
        Inputs: 
            X: list of trajectories [traj_1, traj_2, ... traj_N]
                where each traj contains an ndarray of state variables
                of shape (n_time, n_state) (where n_time depends on the traj)
                (this is in the form of pysindy's `multiple_trajecotries=True` keyword)
            Y: list of trajectories [traj_1, traj_2, ... traj_N]
                where each traj contains an ndarray of scalar reward variables
                of shape (n_time,) (where n_time depends on the traj)
                (this is in the form of pysindy's `multiple_trajecotries=True` keyword)
            U: OPTIONAL list of trajectories [traj_1, traj_2, ... traj_N]
                where each traj contains an ndarray of control inputs. If the inputs X
                are concatenated with U for the regression.
                of shape (n_time, n_state) (where n_time depends on the traj)
        '''
        self.optimizer.coef_list = []
        Y_tmp = np.concatenate(Y)
        X_flat = np.concatenate(X)
        
        if self.use_control and U is None:
            raise ValueError('use_control is True, but U is None')
        elif self.use_control:
            # add u as features
            U_flat = np.concatenate(U)
            X_tmp = np.concatenate([np.array(X_flat), np.array(U_flat)], axis=-1)
        else: 
            X_tmp = X_flat

        X_tmp = self.feature_library.reshape_samples_to_spatial_grid(X_tmp)
        ThetaX = self.feature_library.fit_transform(X_tmp)
        
        self.optimizer.fit(ThetaX, Y_tmp)
        self._is_fitted = (True and init)
        self.safe_idx = np.ones(self.n_models, dtype=bool)
        
        self.n_state = X[0][0].shape[0]
        if self.use_control:
            self.n_control = U[0][0].shape[0]
        else: 
            self.n_control = 1
        
    def reset_safe_list(self):
        '''TO-DO: probably don't need safe_idx'''
        self.safe_idx = np.ones(self.n_models, dtype=bool)
        
    def get_coef_list(self):
        ''''
        Get list of model coefficients.
        
        (Wrapper for pysinder optimizer `coef_list` attribute.)
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
        coef_list = self.get_coef_list()
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
        coef_list = self.get_coef_list()
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
        
    def predict(self, x, u):
        '''
        Inputs:
            x: ndarray of shape (n_pts, n_state) of state variabels
            u: ndarray of shape (n_pts, n_inputs) of inputs
        '''
        if self.use_control:    
            X_tmp = np.concatenate([np.array(x), np.array(u)], axis=-1)
        else:
            X_tmp = x
        X_tmp = self.feature_library.reshape_samples_to_spatial_grid(X_tmp)
        ThetaX = self.feature_library.transform(X_tmp)
        return (ThetaX @ self.optimizer.coef_.T)[0]
    
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
            print_str = f'r{idx} = '
            
            for c, name in zip(eq, feature_names):
                c_round = np.round(c, precision)
                if c_round != 0:
                    print_str += f'{c_round:.{precision}f} {name} + '
            
            print(print_str[:-2])
            
    def set_ensemble_coefs_(self, weight_list):
        self.optimizer.coef_list = weight_list
        self.optimizer.coef_ = self.set_idx_coef_(0)
        
    def save(self, save_path):
        '''EXPERIMENTAL'''
        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, load_path):
        '''EXPERIMENTAL'''
        
        with open(load_path, 'rb') as f:
            model = pickle.load(f)

            self.__init__(model.config)
            
            x_tmp = np.ones((10, model.n_state))
            u_tmp = np.ones((10, model.n_control))
            r_tmp = np.ones((10, 1))
            self.fit([x_tmp], U=[u_tmp], Y=[r_tmp])
            
            self.optimizer.coef_list = model.optimizer.coef_list
            self.optimizer.coef_ = model.optimizer.coef_