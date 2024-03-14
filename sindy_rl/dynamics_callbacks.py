# --------------------------------------------------------
# Intended to be used during dynamics models predict(x, u)
# --------------------------------------------------------

import numpy as np

def project_cartpole(z):
    '''Projecting the dm_control swingup task back onto the circle'''
    cart_pos, cos_th, sin_th, dx, dth = z
    
    u = np.array([cos_th, sin_th])
    new_cos, new_sin = u/np.linalg.norm(u)
    return np.array([cart_pos, new_cos, new_sin, dx, dth])


# only works for (d,)-shaped vectors
def project_cartpole_n(z, n=2): 
    '''
    USE WITH CAUTION.
    n-length pendulum observation project.
    
    Cart Obs appears to be of the form:
        [x, cos(t_1), sin(t_1), ..., cos(t_k), sin(t_k)]
        concatenated with derivatives:
        [dx, dt_t, dt_2, \dots dt_n]
    '''
    new_obs = z.copy()
    for i in range(n):
        idx = 1 + 2*i
        u = z[idx:idx+2]
        new_cos, new_sin = u/np.linalg.norm(u)
        new_obs[idx:idx+2] = [new_cos, new_sin]
    return new_obs

def project_pend(z): 
    cos_th, sin_th, dth = z
    
    u = np.array([cos_th, sin_th])
    new_cos, new_sin = u/np.linalg.norm(u)
    return np.array([new_cos, new_sin, dth])
