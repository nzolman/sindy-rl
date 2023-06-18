import numpy as np

def project_cartpole(z):
    cart_pos, cos_th, sin_th, dx, dth = z
    
    u = np.array([cos_th, sin_th])
    new_cos, new_sin = u/np.linalg.norm(u)
    return np.array([cart_pos, new_cos, new_sin, dx, dth])