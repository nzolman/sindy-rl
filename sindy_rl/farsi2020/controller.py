import numpy as np
from scipy.integrate import solve_ivp

from sindy_rl.farsi2020.utils import lib_jac_from_feat, uOpt, pDot
from sindy_rl.sindy_utils import get_affine_lib_from_base
from sindy_rl.refactor.policy import BasePolicy

class Farsi2020Controller(BasePolicy):
    '''
    Policy class for Farsi 2020 paper [1]. Where a dynamics function is approximated
    by a linear combination of dictionary terms:
        $$ \dot{x}(t) = W \Phi(x) + \sum_j W_j \Phi(x) u $$
    
    And a value function is given by 
        $$ V = \Phi(x)^T P \Phi(x) $$
    
    Should extend naturally to tracking [2]. 
    
    TO-DO: Write convenient subclass for tracking
    
    [1] Farsi, Milad, and Jun Liu. 
        "Structured online learning-based control of continuous-time nonlinear systems." 
        IFAC-PapersOnLine 53.2 (2020): 8142-8149.
        
    [2] Farsi, Milad, and Jun Liu. 
        "A structured online learning approach to nonlinear tracking with unknown dynamics." 
        2021 American Control Conference (ACC). IEEE, 2021.
    '''
    def __init__(self, config):
        '''
        Parameters:
            `n_state` : (int) 
                number of state variables
            `n_control` : (int) 
                number of control variables
            `n_phi_lib`: (int)
                number of elements in phi lib
            `state_features`: list(ints)
                names of the features used in the affine library
            `phi_lib` : (pysindy feature library)
                feature library used for both the state
                and affine control terms
            `affine_lib`: (pysindy feature library)
                full affine dynamics feature library. 
            `phi_fn` : (function)
                pointwise evaluation of Phi library
            `dphi_fn`: (function)
                pointwise evaluations of Jacobian of Phi library
            `W`: (ndarray), shape: (n_state, n_phi_lib)
                Weights on the state-space portion of the
                affine library
            `Wj: (ndarray), shape: (n_control, n_state, n_phi_lib)
                Weights on the control portion of the 
                affine library
            `P`: (ndarray), shape: (n_phi_lib, n_phi_lib)
                symmetric matrix in value function.
            `r`: (ndarray), shape (n_control, )
                diagonal of control objective function matrix, R
            `Q`: (ndarray), shape (n_phi_lib, n_phi_lib)
                state objective function matrix, Q
            `dt`: (float)
                expected delta t 
            `gamma`: (float) >=0
                discount factor for infinite time objective
            `control_data_store`: list of ndarrays of shape (n_time, n_control)
                trajectories of control data (companion to the state trajectories)
            `state_data_store`: list of ndarrays of shape (n_time, n_state)
                trajectories of state data (companion to the control trajectories)
        '''
        self.config = config
        
        self.n_state = config['n_state']
        self.n_control = config['n_control']
        self.var_names = ([f'x{i}' for i in range(self.n_state)]
                             +[f'u{i}' for i in range(self.n_control)]
                         )
        self.affine_lib = config['affine_lib']
        
        # Functions used for point-wise evaluations of the
        # P and optimal control terms.
        self.phi_fn = None
        self.dphi_fn = None
        self._init_phi()
        
        
        self.P = config.get('P0', np.zeros((self.n_phi_lib, 
                                            self.n_phi_lib)))
        
        self.r = config.get('r', np.ones(self.n_control))
        self.Q = config.get('Q', np.diag([*np.ones(self.n_state),
                                          *np.zeros(self.n_phi_lib - self.n_state)
                                         ])
                           )

        self.dt = config.get('dt', 1) # Only needed when calling update_P
        self.gamma = config.get('gamma', 0)
    
    def _init_phi(self):
        '''
        Initialize `phi_fn` and `dph_fn`
        '''
        self.affine_lib.fit(np.ones(self.n_state+self.n_control))
        
        feature_names = np.array(self.affine_lib.get_feature_names(self.var_names))
        
        phi_idx = np.array(['u' not in feat for feat in feature_names])
        
        self.n_phi_lib = np.sum(phi_idx)
        
        self.phi_feature_names = feature_names[phi_idx]
        
        self.phi_fn, self.dphi_fn = lib_jac_from_feat(self.phi_feature_names, 
                                                      self.n_state)

    def set_weights_(self, dyn_model):
        '''
        Set the model weights (after the dynamics model has be fit).
        
        Inputs:
            `dyn_model`: sindy_rl dynamics model
                dynamics model using affine_lib
        '''
        self.W = dyn_model.model.optimizer.coef_[:, :self.n_phi_lib]
        
        # need to be careful with the shape on this one.
        # shape (n_state, n_control * n_phi_lib)
        Wj = dyn_model.model.optimizer.coef_[:, self.n_phi_lib:]
        
        # pre-T: shape (n_state, n_phi_lib, n_control)
        Wj = Wj.reshape(self.n_state, self.n_phi_lib, self.n_control)

        # post-T: shape (n_control, n_state, n_phi_lib)
        self.Wj = Wj.transpose(2,0,1)

        
    def dP(self, t, flat_P, x=None):
        
        return pDot(x, 
                    flat_P.reshape(self.Q.shape), 
                    phi_fn=self.phi_fn, 
                    dphi_fn=self.dphi_fn, 
                    Q=self.Q, 
                    r = self.r, 
                    gamma=self.gamma,
                    W = self.W, 
                    Wj=self.Wj).flatten()
    
    def update_P(self, x, dt, mode='scipy'):
        '''
        x: state variable
        dt: timestep to integrate over
        mode: 
            "scipy": use solve_ivp RK45 integrator
            "euler": naive euler step integrator
        '''
        
        if mode not in ['scipy', 'euler']:
            raise NotImplementedError(f'mode {mode} is not implemented')
        if mode == 'scipy':
            self.P = solve_ivp(self.dP, 
                            y0=self.P.flatten(),
                            t_span = [0, dt], 
                            args=(x,)
                            ).y.T[-1].reshape(self.Q.shape)
        
        elif mode == 'euler': 
            self.P = self.P + dt*self.dP(None, self.P, x=x).reshape(self.Q.shape)
        return self.P
        
    def compute_action(self, obs, update_P=True):
        x = obs
        
        if update_P:
            self.P = self.update_P(x, self.dt)
        u = uOpt(x, self.P, phi_fn=self.phi_fn, dphi_fn=self.dphi_fn, r = self.r, Wj=self.Wj)
        
        return u