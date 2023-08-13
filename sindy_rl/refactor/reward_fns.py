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

def swimmer_reward(z,u, ctrl_cost_weight= 1e-4, forward_reward_weight=1): 
    '''
    surrogate reward for swimmer defined by the head x_vel 
    (instead of the body x_vel)
    '''
    
    x_vel = z[3]
    forward_rew = x_vel * forward_reward_weight
    control_cost = ctrl_cost_weight * np.sum(np.square(u))
    return forward_rew - control_cost