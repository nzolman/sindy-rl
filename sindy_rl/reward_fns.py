import numpy as np
from dm_control.utils.rewards import tolerance

def cart_reward(z, u):
    '''dmc reward for cartpole'''
    cart_pos, cos_th, sin_th, dx, dth = z
    
    upright = (cos_th + 1.0)/2.0
    centered = tolerance(cart_pos, margin=2)
    centered = (1.0 + centered)/2.0
    small_control = tolerance(u, margin=1, 
                              value_at_margin=0, 
                              sigmoid='quadratic')[0]
    small_control = (4.0 + small_control)/5.0
    small_velocity = tolerance(dth, margin=5)
    small_velocity = (1.0 + small_velocity)/2.0
    return upright * small_control * small_velocity * centered

def double_cart_reward(z, u):
    '''
    EXPERIMENTAL
    dmc reward for double-arm cartpole'''
    cart_pos, cos_th1, sin_th1, cos_th2, sin_th2, dx, dth1, dth2 = z
    cos_th = np.array([cos_th1, cos_th2])
    dth = np.array([dth1, dth2])
    
    upright = (cos_th + 1.0)/2.0
    centered = tolerance(cart_pos, margin=2)
    centered = (1.0 + centered)/2.0
    small_control = tolerance(u, margin=1, 
                              value_at_margin=0, 
                              sigmoid='quadratic')[0]
    small_control = (4.0 + small_control)/5.0
    small_velocity = tolerance(dth, margin=5).min()
    small_velocity = (1.0 + small_velocity)/2.0
    return upright.mean() * small_control * small_velocity * centered

def swimmer_reward(z,u, ctrl_cost_weight= 1e-4, forward_reward_weight=1): 
    '''
    surrogate reward for swimmer defined by the head x_vel 
    (instead of the body x_vel)
    '''
    
    x_vel = z[3]
    forward_rew = x_vel * forward_reward_weight
    control_cost = ctrl_cost_weight * np.sum(np.square(u))
    return forward_rew - control_cost


_REF_PIN_CL = np.array([0.0, 1.0, -1.0])
def pinball_lift_track(z, u,  dt = 0.1):
    '''
    EXPERIMENTAL
    Experimental analytic reward for a pinball env'''
    CL = z[:3]
    diff_sq = (CL - _REF_PIN_CL)**2
    return -dt * np.sum(diff_sq)