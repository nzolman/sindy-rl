from dm_control.utils.rewards import tolerance

def cart_reward(z, u):
    cart_pos, cos_th, sin_th, dx, dth = z
    
    upright = (cos_th + 1.0)/2.0
    centered = tolerance(cart_pos, margin=2)
    centered = (1.0 + centered)/2.0
    small_control = tolerance(u, margin=1, 
                              value_at_margin=0, 
                              sigmoid='quadratic')
    small_control = (4.0 + small_control)/5.0
    small_velocity = tolerance(dth, margin=5)
    small_velocity = (1.0 + small_velocity)/2.0
    return 2 * upright * small_control * small_velocity * centered