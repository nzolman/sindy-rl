from tqdm import tqdm
import numpy as np
import pysindy as ps
from pysindy.feature_library import CustomLibrary

from scipy.integrate import solve_ivp

from gym import Env
from sympy import symbols, diff, sympify, Matrix, lambdify, diag


def lib_jac_from_feat(feature_names, n_state):
    '''
    Compute jacobian function.
    
    Input:
        `lib': PySINDy library (assumes no control)
    Output:
        `jac': lambda function that evaluates the pointwise jacobian
            i.e.: jac(x)
    '''
    feature_names = [feat.replace(' ', '*') for feat in feature_names]
    state_names = [f"x{i}" for i in range(n_state)]
    sym_state = [sympify(state) for state in state_names]
    sym_features = [sympify(feature) for feature in feature_names]
    sym_lib = Matrix([sym_features]).T
    sym_jac = sym_lib.jacobian(sym_state)
    jac = lambdify(sym_state, sym_jac, "numpy")
    jac_fn = lambda x: jac(*x)
    lib = lambdify(sym_state, sym_lib, "numpy")
    lib_fn = lambda x: lib(*x)
    return lib_fn, jac_fn


def pDot(x, P, phi_fn=None, dphi_fn=None, Q=None, r = None, gamma=0, W = None, Wj=None):
    '''
    Calculates dynamics for symmertric matrix ODE, -P:
        $$
        -\dot{P} = Q 
                    + P \frac{\partial \Phi}{\partial x}(x) W  
                    + \left(P \frac{\partial \Phi}{\partial x}(x) W \right)^T 
                    - \gamma P 
            - \left( P \frac{\partial \Phi}{\partial x}(x) \right)
                \left( \sum_{j=1}^m W_j \Phi(x) r^{-1}_j \Phi(x)^T W_j ^T \right)
            \left( P \frac{\partial \Phi}{\partial x}(x) \right)^T
        $$
        
    Notation: 
        Dynamics: 
            $\dot{x} = W \Phi(x) + \sum_j W_j \Phi(x) u_j$
        Optimization: 
            Minimize w.r.t $u$:  
                $J = \int e^{-\gamma t} \left( \Phi^T(x) Q \Phi(x) + u^T R u dt \right)$
        n_lib = dim(Phi)
        n_state = dim(x)
    Inputs:
        x: (ndarray), shape (n_state,)
            the state variable
        P: (ndarray), shape (n_lib, n_lib)
            the symmetric matrix defining the quadratic form value function $V = \Phi(x)^T P \Phi(x)$
        phi_fn: (python function)
            the function Phi(x)
        dphi_fn: (python function)
            the jacobian (matrix valued function) dPhi(x)/dx
        Q: (ndarray), shape: (n_lib, n_lib)
            the state penality objective matrix for the library
        r: (ndarray)
            diagonal of penalty matrix (R) for the control.
        gamma: (float >= 0)
            discount factor
        W: (ndarray), shape: (n_state, n_lib)
            state library coefficients
        Wj: (ndarray), shape: (n_control, n_state, n_lib)
            control library coefficients
    '''
    Phi_x = phi_fn(x)
    dPhi_x = dphi_fn(x)
    
    P_dPhi = P @ dPhi_x
    P_dPhi_W = P_dPhi @ W
    
    w_sum = np.einsum('an,jnp,pk,j,qk,jmq,bm->ab',P_dPhi, Wj,Phi_x, 1/r, Phi_x, Wj, P_dPhi)
    
    rhs = Q + P_dPhi_W + P_dPhi_W.T - gamma * P - w_sum 
    return rhs

def uOpt(x, P, phi_fn=None, dphi_fn=None, r = None, Wj=None):
    '''
    Calculates optimal control output, u.
    
        $$
        u^*_j = - \Phi(x)^T r_j^{-1} P \frac{\partial \Phi}{\partial x}(x) W_j \Phi(x)
        $$
    
    Notation: 
        Dynamics: 
            $\dot{x} = W \Phi(x) + \sum_j W_j \Phi(x) u_j$
        Optimization: 
            Minimize w.r.t $u$:  
                $J = \int e^{-\gamma t} \left( \Phi^T(x) Q \Phi(x) + u^T R u dt \right)$
        n_lib = dim(Phi)
        n_state = dim(x)
    Inputs:
        x: (ndarray), shape (n_state,)
            the state variable
        P: (ndarray), shape (n_lib, n_lib)
            the symmetric matrix defining the quadratic form value function $V = \Phi(x)^T P \Phi(x)$
        phi_fn: (python function)
            the function Phi(x)
        dphi_fn: (python function)
            the jacobian (matrix valued function) dPhi(x)/dx
        r: (ndarray)
            diagonal of penalty matrix (R) for the control. 
        Wj: (ndarray), shape: (n_control, n_state, n_lib)
            control library coefficients
    '''
    Phi_x = phi_fn(x)
    dPhi_x = dphi_fn(x)
    u = -1*np.einsum('kp,j,pq,qn,jnr,rk->j',Phi_x.T, 1/r, P, dPhi_x, Wj, Phi_x)
    return u


